# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023 The vLLM team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/llm.py

from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

import os
import logging
import random
import tqdm
import torch
import torch.nn as nn
import multiprocessing
from torch.nn.utils.rnn import pad_sequence
from vllm import LLM, TokensPrompt
from vllm.outputs import PoolingRequestOutput, RequestOutput
from vllm.sampling_params import RequestOutputKind
from vllm.utils import Counter


logger = logging.getLogger(__name__)

EOS_ID = 151643
THINK_END_ID = 151649
IS_TRAINING = True
TRAIN_SPLIT_POINTS = [8000]
TRAIN_BUDGET_PROBS = [1.0]
VAL_SPLIT_POINTS = [2000, 4000, 6000, 8000]
VAL_BUDGET_PROBS = [0.25] * 4
VARIANCE_REDUCTION = "brpo"  # brpo, v1only, v2only, none
SUMMARY_METHOD = "brpo"  # brpo, grpo
MAX_ANSWER_LEN = 128
COT_NUM = 8
FINAL_ANSWER_DELIMITER = "**Final Answer**\n"
DELIMITER = "\n\n" + FINAL_ANSWER_DELIMITER  # + "\\boxed{"

def get_think_end_id():
    return THINK_END_ID

def set_is_training(value):
    global IS_TRAINING
    print(f"set is_training from {IS_TRAINING} to {value}")
    IS_TRAINING = value

def set_variance_reduction(value):
    # assert value in ["brpo", "v1only", "v2only", "none", "notrain"]
    global VARIANCE_REDUCTION
    VARIANCE_REDUCTION = value

def get_variance_reduction():
    global VARIANCE_REDUCTION
    return VARIANCE_REDUCTION


def set_summary_method(value):
    global SUMMARY_METHOD
    SUMMARY_METHOD = value

def get_summary_method():
    global SUMMARY_METHOD
    return SUMMARY_METHOD

def set_training_split_plan(max_gen_len: int, n_budget_support: int, budget_probs: "str"):
    assert max_gen_len in [8000, 16000]
    assert budget_probs in ["uniform", "linear", "base"]
    global TRAIN_SPLIT_POINTS, TRAIN_BUDGET_PROBS, VAL_SPLIT_POINTS, VAL_BUDGET_PROBS
    if max_gen_len == 8000:
        val_n = 32
        # VAL_SPLIT_POINTS = [8000 // val_n * i for i in range(1, val_n + 1)]
        # VAL_BUDGET_PROBS = [1.0 / val_n] * val_n
        # VAL_SPLIT_POINTS = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000]
        # VAL_BUDGET_PROBS = [0.125] * 8
        VAL_SPLIT_POINTS = [2000, 4000, 6000, 8000]
        VAL_BUDGET_PROBS = [0.25] * 4
        assert n_budget_support in [1, 2, 4, 8]
        TRAIN_SPLIT_POINTS = [8000 // n_budget_support * i for i in range(1, n_budget_support + 1)]
    elif max_gen_len == 16000:
        VAL_SPLIT_POINTS = [4000, 8000, 12000, 16000]
        VAL_BUDGET_PROBS = [0.25] * 4
        assert n_budget_support in [1, 2, 4, 8]
        TRAIN_SPLIT_POINTS = [16000 // n_budget_support * i for i in range(1, n_budget_support + 1)]
    else:
        raise ValueError(f"Unexpected max_gen_len: {max_gen_len}")
    if budget_probs == "uniform":
        TRAIN_BUDGET_PROBS = [1.0 / n_budget_support] * n_budget_support
    elif budget_probs == "linear":
        budget_sum = sum(TRAIN_SPLIT_POINTS)
        TRAIN_BUDGET_PROBS = [l / budget_sum for l in TRAIN_SPLIT_POINTS]
    elif budget_probs == "base":
        TRAIN_BUDGET_PROBS = [0.0] * (n_budget_support - 1) + [1.0 - 0.0 * (n_budget_support - 1)]
    else:
        raise ValueError(f"Unexpected budget_probs: {budget_probs}")
    print(">" * 50)
    print(f"max_gen_len {max_gen_len}, n_budget_support {n_budget_support}, budget_probs {budget_probs}")
    print(f"TRAIN_BUDGETS: {TRAIN_SPLIT_POINTS}")
    print(f"TRAIN_BUDGET_PROBS: {TRAIN_BUDGET_PROBS}")
    print(f"VAL_BUDGETS: {VAL_SPLIT_POINTS}")
    print(f"VAL_BUDGET_PROBS: {VAL_BUDGET_PROBS}")
    print("<" * 50)

def get_is_training():
    global IS_TRAINING
    return IS_TRAINING

def get_max_answer_len():
    return MAX_ANSWER_LEN

def get_split_points():
    if get_is_training():
        return TRAIN_SPLIT_POINTS
    return VAL_SPLIT_POINTS

def get_budget_probs():
    if get_is_training():
        return TRAIN_BUDGET_PROBS
    else:
        return VAL_BUDGET_PROBS

def get_split_num():
    return len(get_split_points())

def set_cot_num(value):
    global COT_NUM
    COT_NUM = value

def get_cot_num():
    global COT_NUM
    return COT_NUM

def find_all_index(resp):
    indexes = []
    i = 0
    while i < len(resp) - 1:
        if resp[i] == '\n' and resp[i + 1] == '\n':
            indexes.append(i)
            i += 1
        i += 1
    return indexes

def sample_partial_cot(tokenizer, responses: RequestOutput):
    question_ids = responses.prompt_token_ids
    new_line = tokenizer("\n\n", add_special_tokens=False).input_ids
    ellipsis_new_line = tokenizer("...\n\n...\n\n</think>", add_special_tokens=False).input_ids
    final_answer = tokenizer(DELIMITER, add_special_tokens=False).input_ids
    # print(f"ellipsis_new_line: {len(ellipsis_new_line)}, final_answer: {len(final_answer)}")  # 3 5

    def remove_final_answer(token_ids):
        for i in range(len(token_ids)):
            match = 0
            for j in range(len(final_answer)):
                if i + j >= len(token_ids):
                    break
                if token_ids[i + j] != final_answer[j]:
                    break
                match += 1
            if match >= len(final_answer):
                return token_ids[:i], match
        return token_ids, 0

    sampled_len = get_split_points()
    prompt_with_lens = []
    for output in responses.outputs:
        cot_ids = list(output.token_ids)
        if cot_ids[-1] == EOS_ID:
            print(f"last token {cot_ids[-1]}")
            print("!" * 100)
        no_think_end = (output.token_ids[-1] != get_think_end_id())
        partial_lens = defaultdict(int)
        for plen in sampled_len:
            if len(cot_ids) > plen:
                partial_lens[plen] += 1
            else:
                partial_lens[len(cot_ids)] += 1

        prompts = []
        for plen in sorted(partial_lens.keys()):
            num = partial_lens[plen]
            if len(cot_ids) > plen or no_think_end:  # truncated or cot reach max length limit
                partial_output = cot_ids[:plen]
                valid_len = plen
                prompt = question_ids + partial_output + ellipsis_new_line + final_answer
            else:  # cot with think end
                valid_len = len(cot_ids)
                prompt = question_ids + cot_ids + final_answer

            # vllm v1 has a bug: https://github.com/vllm-project/vllm/issues/13175
            max_input_id = max(prompt)
            if max_input_id > tokenizer.max_token_id:
                print(f"ERROR: {max_input_id} > {tokenizer.max_token_id}. question: {max(question_ids)}, cot: {max(cot_ids)}. len: {len(prompt)}, index: {list(prompt).index(max_input_id)}")
                print("!" * 100)
                prompt = [x if x <= tokenizer.max_token_id else 271 for x in prompt]

            prompts.append((prompt, num, valid_len, no_think_end))
        prompt_with_lens.append(prompts)
    return prompt_with_lens


class LLM(LLM):

    def __run_engine(
            self, *, use_tqdm: bool
    ) -> List[Union[RequestOutput, PoolingRequestOutput]]:
        # Disable tqdm.
        del use_tqdm

        # Run the engine.
        outputs: List[Union[RequestOutput, PoolingRequestOutput]] = []
        while self.llm_engine.has_unfinished_requests():
            step_outputs = self.llm_engine.step()
            for output in step_outputs:
                if output.finished:
                    outputs.append(output)
                    if "-" not in output.request_id:
                        prompt_with_lens = sample_partial_cot(self.llm_engine.tokenizer.tokenizer, output)
                        for groupd_id in range(len(prompt_with_lens)):
                            for i, (prompt, num, valid_len, no_think_end) in enumerate(prompt_with_lens[groupd_id]):
                                request_id = "-".join([output.request_id, str(groupd_id), str(i), str(valid_len), str(int(no_think_end))])
                                sample_params = self.sampling_params.clone()
                                used_len = len(prompt) - len(output.prompt_token_ids)
                                sample_params.max_tokens = get_max_answer_len() - 10
                                sample_params.n *= num
                                sample_params.output_kind = RequestOutputKind.FINAL_ONLY
                                self.llm_engine.add_request(
                                    request_id,
                                    TokensPrompt(prompt_token_ids=prompt),
                                    sample_params,
                                    lora_request=None,
                                    prompt_adapter_request=None,
                                    priority=0,
                                )
                        continue
        return outputs

    def _run_engine(self, *, use_tqdm: bool) -> List[Union[RequestOutput, PoolingRequestOutput]]:
        outputs = self.__run_engine(use_tqdm=use_tqdm)
        return self.__post_process(outputs)

    def __post_process(self, request_outputs: List[RequestOutput]):
        from verl.models.transformers.flex_attn import BLOCK_SIZE
        outputs_group = defaultdict(list)
        request_id2outputs = {}
        for request_output in request_outputs:
            request_id = request_output.request_id
            if "-" not in request_id:
                request_id2outputs[int(request_id)] = request_output
                continue
            req_ids = [int(_id) for _id in request_id.split("-")]
            assert len(req_ids) == 5
            # req_id, group_id, idx, valid_len, no_think_end
            outputs_group[(req_ids[0], req_ids[1])].append((req_ids[2], req_ids[3], req_ids[4], request_output))
        request_id2ground_truth = {}
        for i, request_id in enumerate(sorted(request_id2outputs.keys())):
            request_id2ground_truth[request_id] = self.ground_truth_list[i]

        output_token_ids = []
        attention_mask = []
        response_position_ids = []
        tree_invalid_start, tree_invalid_end = [], []
        summary_scores = []
        summary_formatted = []
        prompt_len = []
        split_id = []
        no_think_end_list = []
        cot_len_list = []
        summary_len_list = []
        # Sort the outputs by (request ID, group ID, split ID).
        # This is necessary because some requests may be finished earlier than
        # its previous requests.
        for r_id, g_id in sorted(outputs_group.keys()):
            id_outputs = outputs_group[(r_id, g_id)]
            outputs = sorted(id_outputs, key=lambda x: x[0])
            ground_truth = request_id2ground_truth[r_id]
            question = request_id2outputs[r_id].prompt_token_ids
            question_len = len(question)
            response_tree = list(request_id2outputs[r_id].outputs[g_id].token_ids)
            response_for_sample = [x if x <= self.llm_engine.tokenizer.tokenizer.max_token_id else 271 for x in response_tree]
            position_ids = [_ for _ in range(len(response_tree))]
            invalid_start, invalid_end = [0] * len(response_tree), [0] * len(response_tree)
            tree_lens, tree_ids, tree_scores, tree_formatted = [], [], [], []
            last_valid_len = 0
            correct_answer_steps = []
            tree_no_think, tree_cot_len, tree_summary_len = [], [], []
            for i, valid_len, no_think_end, g_output in outputs:
                assert g_output.prompt_token_ids[:question_len] == question
                assert g_output.prompt_token_ids[question_len: question_len + valid_len] == response_for_sample[:valid_len]
                len_0 = last_valid_len      # start of this split
                len_1 = valid_len           # end of this split
                assert last_valid_len < len_1 or len_1 == 0, f"{last_valid_len} < {len_1}"
                inserted_tokens = list(g_output.prompt_token_ids[question_len + valid_len:])
                has_correct_answer = False
                for j, output in enumerate(g_output.outputs):
                    prompt_resp = inserted_tokens + list(output.token_ids)
                    len_2 = len(response_tree)              # start of inserted tokens
                    len_3 = len_2 + len(inserted_tokens)    # start of summary
                    len_4 = len_2 + len(prompt_resp)        # end of summary
                    score, correct_answer, formatted = compute_reward_and_correct_answer(
                        self.reward_tokenizer,
                        cot=response_tree[:valid_len], summary=prompt_resp,
                        ground_truth=ground_truth, split_id=i,
                    )
                    if correct_answer is not None and formatted:
                        correct_answer_steps.append(correct_answer)
                        has_correct_answer = True
                    # print(f"score: {score}, gt: {ground_truth}, resp: {summary_response_str}")
                    tree_scores.append(score)
                    tree_formatted.append(formatted)
                    invalid_start += [len_2 + _ - len_1 for _ in range(len(prompt_resp))]
                    invalid_end += [_ for _ in range(len(prompt_resp))]
                    response_tree += prompt_resp
                    position_ids += [valid_len + _ for _ in range(len(prompt_resp))]
                    tree_lens.append([len_0, len_1, len_2, len_3, len_4])
                    tree_ids.append(i)
                    tree_summary_len.append(len(prompt_resp))
                last_valid_len = len_1
                tree_no_think.append(no_think_end)
                tree_cot_len.append(valid_len)

            output_token_ids.append(response_tree)
            tree_mask = [True] * len(response_tree)
            while (len(tree_mask) + question_len) % BLOCK_SIZE != 0:
                # we pad attention_mask to be divisible by BLOCK_SIZE because flex_attention has numeric issue
                tree_mask.append(True)
            # assert len(tree_mask) <= self.sampling_params.max_tokens
            attention_mask.append(tree_mask)
            response_position_ids.append(position_ids)
            tree_invalid_start.append(invalid_start)
            tree_invalid_end.append(invalid_end)
            summary_scores.append(tree_scores)
            summary_formatted.append(tree_formatted)
            prompt_len.append(tree_lens)
            split_id.append(tree_ids)
            while len(tree_no_think) < get_split_num():
                tree_no_think.append(tree_no_think[-1])
                tree_cot_len.append(tree_cot_len[-1])
            assert len(tree_no_think) == len(tree_cot_len) == get_split_num()
            no_think_end_list.append(tree_no_think)
            cot_len_list.append(tree_cot_len)
            summary_len_list.append(tree_summary_len)

        # B: #question;  G: #response per question;  S: #split per response;  A: #answer per split
        summary_scores = torch.tensor(summary_scores, dtype=torch.float)        # [B * G, S * A]
        summary_formatted = torch.tensor(summary_formatted, dtype=torch.float)  # [B * G, S * A]
        split_id = torch.tensor(split_id, dtype=torch.long)                     # [B * G, S * A]
        prompt_len = torch.tensor(prompt_len, dtype=torch.long)                 # [B * G, S * A, 5]
        no_think_end_tensor = torch.tensor(no_think_end_list, dtype=torch.float)# [B * G, S]
        cot_len_tensor = torch.tensor(cot_len_list, dtype=torch.float)          # [B * G, S]
        summary_len_tensor = torch.tensor(summary_len_list, dtype=torch.float)  # [B * G, S * A]
        token_level_adv, trainable_mask, cot_metrics = compute_token_level_advantage(
            output_token_ids, summary_scores, summary_formatted, split_id, prompt_len,
            G=get_cot_num(), S=get_split_num()
        )
        budget_metrics = get_budget_metrics(
            summary_scores, summary_formatted, split_id,
            S=get_split_num()
        )
        budget_metrics.update(cot_metrics)
        budget_metrics.update({
            "no_think_end": no_think_end_tensor,
            "cot_length": cot_len_tensor,
            "summary_length": summary_len_tensor,
        })

        return output_token_ids, (
            attention_mask, response_position_ids, 
            tree_invalid_start, tree_invalid_end, 
            token_level_adv, trainable_mask
        ), budget_metrics


def compute_reward_and_correct_answer(tokenizer, cot: List[int], summary: List[int], ground_truth: str, split_id: int):
    if cot[-1] == EOS_ID:
        return 0., None, False
    if len(summary) > 0 and summary[-1] == EOS_ID:
        summary_response_str = tokenizer.decode(summary[:-1])
    else:
        summary_response_str = tokenizer.decode(summary)
    is_correct, correct_ansewr = match_answer_with_ground_truth(summary_response_str, ground_truth)
    think_end = (cot[-1] == THINK_END_ID)
    think_end_count = int(think_end) + summary.count(THINK_END_ID)
    is_formatted = (think_end_count == 1)
    out_of_tokens = (split_id == get_split_num() - 1) and not think_end
    score = float(is_correct and is_formatted)
    return float(score), correct_ansewr, is_formatted


def match_answer_with_ground_truth(model_solution: str, ground_truths):
    from verl.trainer.ppo.math_utils import extract_answer, grade_answer_mathd
    resp_steps = model_solution.split("\n\n")
    final_answer_step = None
    for step_str in resp_steps:
        if step_str.startswith(FINAL_ANSWER_DELIMITER):
            final_answer_step = step_str[len(FINAL_ANSWER_DELIMITER):]
            break
    if final_answer_step is None:
        return False, None

    # print(final_answer_step)
    model_answer = extract_answer(final_answer_step)
    if model_answer is None:
        return False, None

    # Process the ground truth(s)
    if ground_truths is None:
        return False, None

    # Convert single answer to list for uniform processing
    if isinstance(ground_truths, (str, float, int)):
        ground_truths = [ground_truths]

    # Process each ground truth
    processed_ground_truths = []
    for truth in ground_truths:
        truth = str(truth)
        if "\\boxed" in truth:
            processed_truth = extract_answer(truth)
            if processed_truth is not None:
                processed_ground_truths.append(processed_truth)
        else:
            processed_ground_truths.append(truth)

    if not processed_ground_truths:
        return False, None

    # Check against all possible correct answers
    for ground_truth in processed_ground_truths:
        is_correct = grade_answer_mathd(model_answer, ground_truth)
        if is_correct:
            return True, final_answer_step
        is_correct = run_grade_answer_sympy_with_timeout(model_answer, ground_truth, 10.0)
        if is_correct:
            return True, final_answer_step

    return False, None


class ProcessPoolManager:
    _instance = None
    _pool = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ProcessPoolManager, cls).__new__(cls)
            cls._instance._pool = None
        return cls._instance

    def get_pool(self, processes=None):
        if self._pool is None:
            self._pool = multiprocessing.Pool(processes=processes)
        return self._pool

    def terminate(self):
        if self._pool is not None:
            self._pool.terminate()
            self._pool.join()
            self._pool = None

    def __del__(self):
        self.terminate()


def run_grade_answer_sympy_with_timeout(model_answer: str, ground_truth: str, timeout: float = 10.0) -> bool:
    """在后台进程中运行grade_answer_sympy,如果超过timeout秒则终止并返回False"""
    from verl.trainer.ppo.math_utils import grade_answer_sympy
    process_pool_manager = ProcessPoolManager()
    pool = process_pool_manager.get_pool()
    result = pool.apply_async(grade_answer_sympy, (model_answer, ground_truth))
    try:
        return result.get(timeout)
    except multiprocessing.TimeoutError:
        print(f"sympy timeout, model_answer: {model_answer}, ground_truth: {ground_truth}")
        return False


def compute_token_level_advantage(output_token_ids, summary_scores, summary_formatted, split_id, prompt_len, G: int, S: int):
    summary_adv = compute_summary_advantage(summary_scores, split_id, G=G, S=S)         # [B * G, S * A]
    cot_adv, cot_metrics = compute_cot_advantage(summary_scores, split_id, G=G, S=S)    # [B * G, S]
    token_level_adv, trainable_mask = [], []
    for idx, response_tree in enumerate(output_token_ids):
        advantage = [0.0] * len(response_tree)
        trainable = [False] * len(response_tree)
        visited_split_id = {}
        for idy in range(len(summary_scores[idx])):
            bid = int(split_id[idx, idy])
            len_0 = int(prompt_len[idx, idy, 0])
            len_1 = int(prompt_len[idx, idy, 1])
            if bid not in visited_split_id:
                visited_split_id[bid] = (len_0, len_1)
                for i in range(len_0, len_1):
                    advantage[i] = cot_adv[idx, bid]
                    trainable[i] = True
            else:
                assert visited_split_id[bid] == (len_0, len_1)
            len_3 = int(prompt_len[idx, idy, 3])
            len_4 = int(prompt_len[idx, idy, 4])
            for i in range(len_3, len_4):
                advantage[i] = summary_adv[idx, idy]
                trainable[i] = True
        token_level_adv.append(advantage)
        trainable_mask.append(trainable)
    return token_level_adv, trainable_mask, cot_metrics


def compute_cot_advantage(summary_scores, split_id, G: int, S: int):
    B = summary_scores.shape[0] // G
    A = summary_scores.shape[1] // S
    budget_probs = torch.tensor(get_budget_probs(), device=summary_scores.device)
    budget_probs /= budget_probs.sum()
    discount_factor = 2.0
    discount_probs = torch.pow(discount_factor, torch.arange(S, device=summary_scores.device))
    cot_budget_scores = torch.reshape(summary_scores, (B, G, S, A)).mean(dim=-1)  # [B, G, S]
    cot_adv = torch.zeros_like(cot_budget_scores)
    cot_v1 = torch.zeros_like(cot_budget_scores)    # [B, G, S]
    cot_v2 = torch.zeros_like(cot_budget_scores)    # [B, G, S]
    cot_rt = torch.zeros_like(cot_budget_scores)    # [B, G, S]
    cot_mask = torch.zeros_like(cot_budget_scores)  # [B, G, S]
    brpo_adv = torch.zeros_like(cot_budget_scores)  # [B, G, S]
    grpo_adv = torch.zeros_like(cot_budget_scores)  # [B, G, S]
    reshaped_split_id = torch.reshape(split_id, (B, G, S, A))
    for b in range(B):
        for g in range(G):
            for s in range(S):
                cot_return = (cot_budget_scores[b, g, s:] * budget_probs[s:]).sum()
                avg_return = 0
                if G > 1:
                    avg_return = (cot_budget_scores[b, :, s:] * budget_probs[s:]).sum(dim=-1).mean()
                last_cot_return = 0
                if s > 0:
                    # discounted mean
                    discounted_probs = discount_probs[:s]
                    prev = (cot_budget_scores[b, g, :s] * discounted_probs).sum() / discounted_probs.sum()
                    # weighted mean
                    # probs_with_eps = budget_probs[:s] + 1e-4
                    # prev = (cot_budget_scores[b, g, :s] * probs_with_eps).sum() / probs_with_eps.sum()
                    # max
                    # prev = torch.amax(cot_budget_scores[b, g, :s])
                    last_cot_return = prev * budget_probs[s:].sum()
                # baseline = torch.minimum(last_cot_return, avg_return)
                # baseline = (cot_budget_scores[b, :, :] * budget_probs).sum(dim=-1).mean() * budget_probs[s:].sum()
                if VARIANCE_REDUCTION == "brpo":
                    baseline = (s * last_cot_return + (S - s) * avg_return) / S
                elif VARIANCE_REDUCTION == "v1only":
                    baseline = last_cot_return
                elif VARIANCE_REDUCTION == "v2only":
                    baseline = avg_return
                elif VARIANCE_REDUCTION == "none":
                    baseline = 0.0
                elif VARIANCE_REDUCTION == "notrain":
                    baseline = cot_return
                else:
                    raise ValueError(f"Invalid variance reduction method: {VARIANCE_REDUCTION}")
                adv = cot_return - baseline
                cot_adv[b, g, s] = adv
                cot_v1[b, g, s] = last_cot_return
                cot_v2[b, g, s] = avg_return
                cot_rt[b, g, s] = cot_return
                brpo_adv[b, g, s] = cot_return - (s * last_cot_return + (S - s) * avg_return) / S
                grpo_adv[b, g, s] = cot_return - avg_return
                cot_mask[b, g, s] = (reshaped_split_id[b, g, s, 0] == s)
    cot_adv = torch.reshape(cot_adv, (B * G, S))
    cot_v1 = torch.reshape(cot_v1, (B * G, S))
    cot_v2 = torch.reshape(cot_v2, (B * G, S))
    cot_rt = torch.reshape(cot_rt, (B * G, S))
    brpo_adv = torch.reshape(brpo_adv, (B * G, S))
    grpo_adv = torch.reshape(grpo_adv, (B * G, S))
    cot_mask = torch.reshape(cot_mask, (B * G, S))
    return cot_adv, {
        "cot_v1": cot_v1,
        "cot_v2": cot_v2,
        "cot_rt": cot_rt,
        "cot_mask": cot_mask,
        "brpo_adv": brpo_adv,
        "grpo_adv": grpo_adv,
    }


def compute_summary_advantage(summary_scores, split_id, G, S):
    _, grpo_id2mean, _ = get_grpo_metrics_dict(summary_scores, split_id)
    B = summary_scores.shape[0] // G
    A = summary_scores.shape[1] // S
    reshaped_scores = torch.reshape(summary_scores, (B, G, S, A))
    summary_adv = torch.zeros_like(summary_scores)
    budget_probs = torch.tensor(get_budget_probs(), device=summary_scores.device)
    budget_probs /= budget_probs.sum()
    for id_0 in range(summary_scores.shape[0]):
        for id_1 in range(summary_scores.shape[1]):
            sid = int(split_id[id_0, id_1])
            if A == 1:
                # for GRPO baseline
                baseline = reshaped_scores[id_0 // G, :, id_1, 0].mean()
            else:
                baseline = grpo_id2mean[id_0][sid]
            if get_summary_method() == "grpo":
                if id_1 == len(summary_scores[id_0]) - 1:
                    adv = summary_scores[id_0, id_1] - baseline
                else:
                    adv = 0.0
            elif get_summary_method() == "brpo":
                adv = summary_scores[id_0, id_1] - baseline
            else:
                raise ValueError(f"Invalid summary method: {get_summary_method()}")
            summary_adv[id_0, id_1] = adv
    return summary_adv


def get_grpo_metrics_dict(summary_scores, split_id):
    id2score = defaultdict(lambda: defaultdict(list))
    grpo_id2mean = {}
    grpo_id2std = {}
    with torch.no_grad():
        for id_0 in range(summary_scores.shape[0]):
            for id_1 in range(summary_scores.shape[1]):
                id2score[id_0][int(split_id[id_0, id_1])].append(summary_scores[id_0, id_1])
        for idx, split2scores in id2score.items():
            grpo_id2mean[idx] = {}
            grpo_id2std[idx] = {}
            for sid, scores in split2scores.items():
                if len(scores) == 1:
                    grpo_id2mean[idx][sid] = torch.tensor(0.0, device=summary_scores.device)
                    grpo_id2std[idx][sid] = torch.tensor(1.0, device=summary_scores.device)
                elif len(scores) > 1:
                    grpo_id2mean[idx][sid] = torch.mean(torch.tensor(scores))
                    grpo_id2std[idx][sid] = torch.std(torch.tensor(scores))
                else:
                    raise ValueError(f"no score in prompt index: {idx} {sid}")
    return id2score, grpo_id2mean, grpo_id2std

def get_budget_metrics(summary_scores, summary_formatted, split_id, S: int):
    id2score, _, _ = get_grpo_metrics_dict(summary_scores, split_id)
    id2formatted, _, _ = get_grpo_metrics_dict(summary_formatted, split_id)
    BG = summary_scores.shape[0]
    with torch.no_grad():
        budget_scores = torch.zeros((BG, S), device=summary_scores.device)
        final_count = torch.zeros((BG, S), device=summary_scores.device)
        final_scores = torch.zeros((BG, S), device=summary_scores.device)
        is_formatted = torch.zeros((BG, S), device=summary_scores.device)
        for idx in range(BG):
            sid = -1
            for j in range(S):
                if j in id2score[idx]:
                    sid = j
                assert sid >= 0
                is_formatted[idx, j] = torch.mean(torch.tensor(id2formatted[idx][sid], dtype=torch.float32))
                budget_scores[idx, j] = torch.mean(torch.tensor(id2score[idx][sid], dtype=torch.float32))
            final_sid = sid
            final_count[idx, final_sid] = 1
            final_scores[idx, final_sid] = budget_scores[idx, final_sid]
    return {
        "budget_scores": budget_scores,
        "final_count": final_count,
        "final_scores": final_scores,
        "is_formatted": is_formatted,
    }

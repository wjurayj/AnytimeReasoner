# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
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
"""
The vllm_rollout that can be applied in different backend
When working with FSDP:
- Use DTensor weight loader (recommended) or HF weight loader
- Utilize state_dict from the FSDP to synchronize the weights among tp ranks in vLLM
When working with Megatron:
- Use Megatron weight loader
- During training, only the current pp stage holds the parameters
- Before inference, broadcast the parameters of the current pp rank to all other pp ranks (all pp ranks holds all the parameters)
- Bind the parameters to the inference engine
- Do inference in tp. pp is treated as additional dp
- After inference, all the parameters that doesn't belong to this pp rank is freed.
"""
import numpy as np
import os
from typing import List
from contextlib import contextmanager
from omegaconf import DictConfig
import torch
import torch.distributed
from tensordict import TensorDict
from torch import nn
from typing import Any, Union
from verl import DataProto
from verl.utils.torch_functional import pad_2d_list_to_length
from verl.workers.rollout.base import BaseRollout
from vllm.distributed import parallel_state as vllm_ps
from vllm import LLM, SamplingParams
from verl.third_party.vllm import vllm_version
from verl.third_party.vllm import LLM, vllm_version
from verl.trainer.ppo.brpo import get_split_points, set_cot_num, get_think_end_id, set_is_training, set_training_split_plan, set_variance_reduction, set_summary_method
from verl.models.transformers.flex_attn import BLOCK_SIZE

# TODO
# 1. support pp in vllm
# 2. passing tokenizer is not necessary? no encoding/decoding is happending here
# 3. simplify init logics


# NOTE(sgm): add for verl. We can optimize it by making the dataloader yield List[int] without padding.
def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor) -> List[int]:
    # remove the left padding in the prompt token_id
    # pad_token_id = self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id is not None else self.llm_engine.tokenizer.eos_token_id
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    token_ids = prompt_token_ids[non_pad_index:].tolist()
    return token_ids


def _repeat_interleave(value: Union[torch.Tensor, np.ndarray], repeats: int) -> Union[torch.Tensor, List[Any]]:
    if isinstance(value, torch.Tensor):
        return value.repeat_interleave(repeats, dim=0)
    else:
        return np.repeat(value, repeats, axis=0)


class vLLMRollout(BaseRollout):

    def __init__(self, model_path: str, config: DictConfig, tokenizer, model_hf_config, **kwargs):
        """A vLLM rollout. It requires the module is supported by the vllm.

        Args:
            module: module here follows huggingface APIs
            config: DictConfig
            tokenizer: the task/model tokenizer
            model_hf_config: the huggingface config to initiallize the generating model in vllm
            **kwargs: train_tp, for Megatron Backend to initialize hybrid engine (zero redundancy) process group
        """
        super().__init__()
        self.config = config
        assert not (not config.enforce_eager and config.free_cache_engine), \
            "disable CUDA graph (enforce_eager = False) if free cache engine"

        tensor_parallel_size = self.config.get('tensor_model_parallel_size', 1)
        assert tensor_parallel_size <= torch.distributed.get_world_size(), \
            "tensor parallel size should be less than or equal to the world size"
        max_num_batched_tokens = self.config.get('max_num_batched_tokens', 8192)

        if kwargs.get('train_tp', None) is not None:
            # deployed with megatron
            import os
            os.environ['CUDA_TIMER_STREAM_KAFKA_ENABLE'] = '0'
            os.environ['MEGATRON_IMPORT_TIMERS'] = '0'
            train_tp = kwargs.get('train_tp', None)
            num_tp_per_train_tp = train_tp // tensor_parallel_size
            vllm_ps.initialize_parallel_state(tensor_model_parallel_size=tensor_parallel_size,
                                              num_tp_per_train_tp=num_tp_per_train_tp)

        assert model_hf_config.max_position_embeddings >= config.prompt_length + config.response_length, \
            "model context length should be greater than total sequence length"

        self.inference_engine = LLM(
            model=model_path,
            enable_sleep_mode=True,
            tensor_parallel_size=tensor_parallel_size,
            distributed_executor_backend="external_launcher",
            dtype=config.dtype,
            enforce_eager=config.enforce_eager,
            gpu_memory_utilization=config.gpu_memory_utilization,
            disable_custom_all_reduce=True,
            skip_tokenizer_init=False,
            max_model_len=config.prompt_length + config.response_length,
            disable_log_stats=config.disable_log_stats,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=config.enable_chunked_prefill,
            enable_prefix_caching=True,
            seed=config.seed,
        )
        self.inference_engine.reward_tokenizer = tokenizer
        self.inference_engine.apply_model(lambda model: print(model.__class__))

        # Offload vllm model to reduce peak memory usage
        self.inference_engine.sleep(level=1)

        kwargs = dict(
            n=1,
            logprobs=0,  # can be set to 0 and let actor to recompute
            max_tokens=config.response_length,
        )

        # # we may detokenize the result all together later
        if vllm_version != '0.3.1':
            kwargs['detokenize'] = False

        # supporting adding any sampling params from the config file
        for k in config.keys():
            if hasattr(SamplingParams(), str(k)):
                kwargs[k] = config.get(k)

        print(f"kwargs: {kwargs}")
        self.sampling_params = SamplingParams(**kwargs)
        set_training_split_plan(
            max_gen_len=config.max_gen_len,
            n_budget_support=config.n_budget_support,
            budget_probs=config.budget_probs,
        )

        self.pad_token_id = tokenizer.pad_token_id

    @contextmanager
    def update_sampling_params(self, **kwargs):
        old_sampling_params = self.sampling_params.clone()
        # update sampling params
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    setattr(self.sampling_params, key, value)
        self.split_params = self.sampling_params.clone()
        # self.sampling_params.stop = ["</think>"]
        self.sampling_params.stop_token_ids = [get_think_end_id()]
        self.sampling_params.include_stop_str_in_output = True
        set_cot_num(self.config.n)
        set_variance_reduction(self.config.variance_reduction)
        set_summary_method(self.config.summary_method)
        split_points = get_split_points()
        summary_n = self.config.n_summary
        self.split_params.n = summary_n
        self.sampling_params.n = self.config.n
        self.sampling_params.max_tokens = split_points[-1]
        yield
        # roll back to previous sampling params
        # if len(old_sampling_params_args):
        self.sampling_params = old_sampling_params

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        # rebuild vllm cache engine
        if vllm_version in ('0.3.1', '0.4.2', '0.5.4', '0.6.3') and self.config.free_cache_engine:
            self.inference_engine.init_cache_engine()

        idx = prompts.batch['input_ids']  # (bs, prompt_length)
        # left-padded attention_mask
        attention_mask = prompts.batch['attention_mask']
        position_ids = prompts.batch['position_ids']

        batch_size = idx.size(0)

        non_tensor_batch = prompts.non_tensor_batch
        if 'raw_prompt_ids' not in non_tensor_batch:
            non_tensor_batch['raw_prompt_ids'] = np.array(
                [_pre_process_inputs(self.pad_token_id, idx[i]) for i in range(batch_size)], dtype=object)

        if batch_size != len(non_tensor_batch['raw_prompt_ids']):
            raise RuntimeError('vllm sharding manager is not work properly.')

        if 'multi_modal_data' in non_tensor_batch:
            vllm_inputs = []
            for raw_prompt_ids, multi_modal_data in zip(non_tensor_batch.pop('raw_prompt_ids'),
                                                        non_tensor_batch.pop('multi_modal_data')):
                vllm_inputs.append({'prompt_token_ids': raw_prompt_ids, 'multi_modal_data': multi_modal_data})
        else:
            vllm_inputs = [{
                'prompt_token_ids': raw_prompt_ids
            } for raw_prompt_ids in non_tensor_batch.pop('raw_prompt_ids')]

        do_sample = prompts.meta_info.get('do_sample', True)
        is_validate = prompts.meta_info.get('validate', False)
        set_is_training(not is_validate)
        if not do_sample:
            kwargs = {
                # 'best_of': 1,
                'top_p': 1.0,
                'top_k': -1,
                'min_p': 0.0,
                'temperature': 0,
                'n': 1  # if greedy, only 1 response
            }
        elif is_validate:
            # TODO: try **
            kwargs = {
                'top_k': self.config.val_kwargs.top_k,
                'top_p': self.config.val_kwargs.top_p,
                'temperature': self.config.val_kwargs.temperature,
                'n': 1,  # if validate, already repeat in ray_trainer
            }

        # users can customize different sampling_params at different run
        with self.update_sampling_params(**kwargs):
            self.inference_engine.sampling_params = self.split_params
            ground_truth_list = []
            for i in range(len(prompts)):
                ground_truth_list.append(prompts[i].non_tensor_batch['reward_model']['ground_truth'])
            non_tensor_batch.pop("reward_model")
            self.inference_engine.ground_truth_list = ground_truth_list
            response, output_tuple, output_metrics = self.inference_engine.generate(
                prompts=vllm_inputs,  # because we have already convert it to prompt token id
                sampling_params=self.sampling_params,
                use_tqdm=False)

            if is_validate:
                response = [resp[:self.config.response_length] for resp in response]
            response = pad_2d_list_to_length(response, self.pad_token_id, max_length=self.config.response_length)
            response = response.to(idx.device)
            assert response.shape[-1] <= self.config.response_length
            output_list = []
            for out in output_tuple:
                if is_validate:
                    out = [_out[:self.config.response_length] for _out in out]
                padded_out = pad_2d_list_to_length(out, 0, max_length=self.config.response_length)
                padded_out = padded_out.to(idx.device)
                assert padded_out.shape[-1] <= self.config.response_length
                output_list.append(padded_out)
            metrics_dict = {}
            for key, value in output_metrics.items():
                metrics_dict[key] = value.to(idx.device)

            if self.config.n > 1 and do_sample:
                idx = _repeat_interleave(idx, self.config.n)
                attention_mask = _repeat_interleave(attention_mask, self.config.n)
                position_ids = _repeat_interleave(position_ids, self.config.n)
                batch_size = batch_size * self.config.n
                assert response.size(0) == batch_size, f"{response.size(0)} != {batch_size}"
                assert idx.size(0) == batch_size
                if 'multi_modal_inputs' in non_tensor_batch.keys():
                    non_tensor_batch['multi_modal_inputs'] = _repeat_interleave(non_tensor_batch['multi_modal_inputs'],
                                                                                self.config.n)

            seq = torch.cat([idx, response], dim=-1)

        valid_response_mask, response_position_ids, tree_invalid_start, tree_invalid_end, token_level_adv, trainable_mask = tuple(output_list)
        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[:, -1:] + 1 + response_position_ids
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        attention_mask = torch.cat((attention_mask, valid_response_mask), dim=-1)
        response_invalid_slice = torch.stack([tree_invalid_start, tree_invalid_end], dim=-1)
        prompt_invalid_slice = torch.zeros((*idx.shape, 2), dtype=torch.long, device=idx.device)
        tree_invalid_slice = torch.concat([prompt_invalid_slice, response_invalid_slice], dim=1)
        # token_level_adv = torch.cat(
        #     [torch.zeros_like(idx, dtype=torch.float32, device=idx.device), token_level_adv], dim=-1)
        # trainable_mask = torch.cat(
        #     [torch.zeros_like(idx, dtype=torch.long, device=idx.device), trainable_mask], dim=-1)

        # all the tp ranks should contain the same data here. data in all ranks are valid

        tensor_dict = {
                'prompts': idx,
                'responses': response,
                'input_ids': seq,  # here input_ids become the whole sentences
                # 'old_log_probs': log_probs, # we will recompute old log prob with actor
                'attention_mask': attention_mask,
                'position_ids': position_ids,
                'tree_invalid_slice': tree_invalid_slice,
                'advantages': token_level_adv,
                'trainable_mask': trainable_mask,
            }
        tensor_dict.update(metrics_dict)
        batch = TensorDict(
            tensor_dict,
            batch_size=batch_size)

        # free vllm cache engine
        if vllm_version in ('0.3.1', '0.4.2', '0.5.4', '0.6.3') and self.config.free_cache_engine:
            self.inference_engine.free_cache_engine()

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)

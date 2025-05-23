# Copyright 2025 Sea AI Lab.
# Copyright 2025 The HuggingFace Inc. team.
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
# Adapted from https://github.com/huggingface/transformers/blob/v4.50.1/src/transformers/integrations/flex_attention.py

from contextlib import contextmanager
from typing import Optional, Tuple, Union

import torch

from transformers.utils import is_torch_flex_attn_available


if is_torch_flex_attn_available():
    from torch.nn.attention.flex_attention import (
        BlockMask,
        flex_attention,
    )
    from torch.nn.attention.flex_attention import (
        create_block_mask as create_block_causal_mask_flex,
    )

BLOCK_SIZE = 128


class WrappedFlexAttention:
    """
    We are doing a singleton class so that flex attention is compiled once when it's first called.
    """

    _instance = None
    _is_flex_compiled = False
    _compiled_flex_attention = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            # Create a new instance if one doesn't already exist
            cls._instance = super().__new__(cls)
        return cls._instance

    @torch.compiler.disable(recursive=False)
    def __init__(self):
        """
        Initialize or update the singleton instance.
        """
        if self._is_flex_compiled is False:
            ##############################################################################################################
            # set mode="max-autotune-no-cudagraphs" to fix exception, https://github.com/pytorch/pytorch/issues/146260
            # https://github.com/pytorch/pytorch/pull/143299/files
            # https://github.com/pytorch/pytorch/issues/141486
            mode = "max-autotune-no-cudagraphs"
            # mode = "default"
            dynamic = False
            fullgraph = False
            self._compiled_flex_attention = torch.compile(flex_attention, dynamic=dynamic, fullgraph=fullgraph, mode=mode)
            self._compiled_create_block_mask = torch.compile(create_block_causal_mask_flex, dynamic=dynamic, fullgraph=fullgraph, mode=mode)
            ##############################################################################################################
            self._is_flex_compiled = True

    def __call__(self):
        return self._compiled_flex_attention

    @property
    def create_block_mask(self):
        return self._compiled_create_block_mask


@torch.compiler.disable(recursive=False)
def make_flex_block_causal_mask(attention_mask_2d: torch.Tensor, tree_invalid_slice: torch.Tensor) -> "BlockMask":
    """
    Create a block causal document mask for tree-like attention mask.
    BlockMask is essential for performant computation of flex attention.
    See: https://pytorch.org/blog/flexattention/
    """
    device = attention_mask_2d.device

    document_ids = attention_mask_2d
    batch_size, total_seq_len = document_ids.shape

    tree_mask_start, tree_mask_end = tree_invalid_slice[:, :, 0], tree_invalid_slice[:, :, 1]

    def causal_mask_mod(batch_idx, head_idx, q_idx, kv_idx):
        """
        Defines the logic of a block causal mask by combining both a standard causal mask
        and a block diagonal document mask.

        See :func:`~torchtune.modules.attention_utils.create_block_causal_mask`
        for an illustration.
        """
        causal_mask = q_idx >= kv_idx
        document_mask = document_ids[batch_idx, q_idx] == document_ids[batch_idx, kv_idx]
        padding_mask = document_ids[batch_idx, q_idx] > 0
        tree_valid_mask = (kv_idx < q_idx - tree_mask_start[batch_idx, q_idx]) | (kv_idx >= q_idx - tree_mask_end[batch_idx, q_idx])
        return causal_mask & document_mask & padding_mask & tree_valid_mask

    return WrappedFlexAttention().create_block_mask(
        mask_mod=causal_mask_mod,
        B=batch_size,
        H=None,  # attention head
        Q_LEN=total_seq_len,
        KV_LEN=total_seq_len,
        device=device,
        BLOCK_SIZE=BLOCK_SIZE,
    )


@torch.compiler.disable(recursive=False)
def compile_friendly_flex_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    # First call initialise singleton wrapper object, second call invokes the object method to return compiled flex attention
    flex_attention_compiled = WrappedFlexAttention()()
    return flex_attention_compiled(
        query,
        key,
        value,
        **kwargs,
    )


def flex_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Union[torch.Tensor, "BlockMask"],
    scaling: Optional[float] = None,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # print(f"query: {query.shape}, key: {key.shape}")
    attn_output, attention_weights = compile_friendly_flex_attention(
        query,
        key,
        value,
        # score_mod=score_mod,
        block_mask=attention_mask,
        enable_gqa=True,
        scale=scaling,
        # Last time checked on PyTorch == 2.5.1: Flex Attention always computes the lse regardless.
        # For simplification, we thus always return it as no additional computations are introduced.
        return_lse=True,
    )
    # lse is returned in float32
    attention_weights = attention_weights.to(value.dtype)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attention_weights


@contextmanager
def update_flex_attn_impl(permanent: bool = True):
    from transformers.models.qwen2.modeling_qwen2 import Qwen2Model
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    original_impl = ALL_ATTENTION_FUNCTIONS["flex_attention"]
    original_update_causal_mask = Qwen2Model._update_causal_mask
    ALL_ATTENTION_FUNCTIONS["flex_attention"] = flex_attention_forward
    Qwen2Model._update_causal_mask = _update_causal_mask
    yield
    if not permanent:
        ALL_ATTENTION_FUNCTIONS["flex_attention"] = original_impl
        Qwen2Model._update_causal_mask = original_update_causal_mask


def _update_causal_mask(
    self,
    attention_mask: torch.Tensor,
    input_tensor: torch.Tensor,
    cache_position: torch.Tensor,
    past_key_values,
    output_attentions: bool = False,
):
    if self.config._attn_implementation == "flex_attention":
        return attention_mask
    raise ValueError("Not supported")

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Subset-finalizing SM120 direct fp8_ds sparse prefill kernels."""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _finalize_fp8ds_local_slots_subset_multihead_kernel(
    q_ptr,
    k_cache_ptr,
    local_indices_ptr,
    token_to_req_indices_ptr,
    block_table_ptr,
    subset_output_ptr,
    subset_lse_ptr,
    stride_q_t: tl.constexpr,
    stride_q_h: tl.constexpr,
    stride_q_d: tl.constexpr,
    stride_local_t,
    stride_local_c,
    block_table_stride,
    stride_subset_t: tl.constexpr,
    stride_subset_h: tl.constexpr,
    stride_subset_d: tl.constexpr,
    stride_lse_t: tl.constexpr,
    stride_lse_h: tl.constexpr,
    cache_block_size: tl.constexpr,
    token_data_size: tl.constexpr,
    block_stride: tl.constexpr,
    fp8_dim: tl.constexpr,
    scale_dim: tl.constexpr,
    quant_block: tl.constexpr,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    num_candidates,
    scale: tl.constexpr,
    HEAD_BLOCK: tl.constexpr,
    BLOCK_D: tl.constexpr,
    FULL_TILE: tl.constexpr,
):
    token_idx = tl.program_id(0)
    head_block_idx = tl.program_id(1)
    head_offsets = head_block_idx * HEAD_BLOCK + tl.arange(0, HEAD_BLOCK)
    dim_offsets = tl.arange(0, BLOCK_D)
    head_mask = head_offsets < num_heads
    dim_mask = dim_offsets < head_dim
    matrix_mask = head_mask[:, None] & dim_mask[None, :]

    q_offsets = (
        token_idx * stride_q_t
        + head_offsets[:, None] * stride_q_h
        + dim_offsets[None, :] * stride_q_d
    )
    if tl.constexpr(FULL_TILE):
        q = tl.load(q_ptr + q_offsets).to(tl.float32)
    else:
        q = tl.load(
            q_ptr + q_offsets,
            mask=matrix_mask,
            other=0.0,
        ).to(tl.float32)
    running_max = tl.full((HEAD_BLOCK,), -float("inf"), tl.float32)
    running_denom = tl.zeros((HEAD_BLOCK,), tl.float32)
    running_acc = tl.zeros((HEAD_BLOCK, BLOCK_D), tl.float32)

    req_idx = tl.load(token_to_req_indices_ptr + token_idx)
    fp8_mask = dim_offsets < fp8_dim
    rope_mask = (dim_offsets >= fp8_dim) & dim_mask
    rope_offsets = tl.maximum(dim_offsets - fp8_dim, 0)

    for candidate_idx in range(0, num_candidates):
        local_idx = tl.load(
            local_indices_ptr + token_idx * stride_local_t + candidate_idx * stride_local_c
        )
        is_valid = local_idx >= 0

        if is_valid:
            block_indices = local_idx // cache_block_size
            block_numbers = tl.load(
                block_table_ptr + req_idx * block_table_stride + block_indices,
                mask=is_valid,
                other=0,
            )
            pos_in_block = local_idx % cache_block_size
            cache_block_ptr = k_cache_ptr + block_numbers.to(tl.int64) * block_stride
            token_data_ptr = cache_block_ptr + pos_in_block * token_data_size
            token_scale_ptr = (
                cache_block_ptr
                + cache_block_size * token_data_size
                + pos_in_block * scale_dim
            )

            x_uint8 = tl.load(token_data_ptr + dim_offsets, mask=fp8_mask, other=0)
            x_fp8 = x_uint8.to(tl.float8e4nv, bitcast=True)
            x_float = x_fp8.to(tl.float32)
            scale_offsets = dim_offsets // quant_block
            encoded_scale = tl.load(
                token_scale_ptr + scale_offsets,
                mask=fp8_mask,
                other=127,
            )
            dequant_scale = tl.exp2(encoded_scale.to(tl.float32) - 127.0)
            x_dequant = x_float * dequant_scale

            rope_ptr = (token_data_ptr + fp8_dim).to(tl.pointer_type(tl.bfloat16))
            rope = tl.load(rope_ptr + rope_offsets, mask=rope_mask, other=0.0).to(
                tl.float32
            )
            kv = tl.where(fp8_mask, x_dequant, rope)
            kv = tl.where(dim_mask, kv, 0.0)

            score = tl.sum(q * kv[None, :], axis=1) * scale
            next_max = tl.maximum(running_max, score)
            previous_weight = tl.exp(running_max - next_max)
            candidate_weight = tl.exp(score - next_max)
            running_acc = (
                running_acc * previous_weight[:, None]
                + kv[None, :] * candidate_weight[:, None]
            )
            running_denom = running_denom * previous_weight + candidate_weight
            running_max = next_max

    denom_safe = tl.where(running_denom > 0.0, running_denom, 1.0)
    subset = running_acc / denom_safe[:, None]
    lse = tl.where(
        running_denom > 0.0,
        running_max + tl.log(running_denom),
        -float("inf"),
    )
    subset_offsets = (
        token_idx * stride_subset_t
        + head_offsets[:, None] * stride_subset_h
        + dim_offsets[None, :] * stride_subset_d
    )
    lse_offsets = token_idx * stride_lse_t + head_offsets * stride_lse_h
    if tl.constexpr(FULL_TILE):
        tl.store(subset_output_ptr + subset_offsets, subset)
        tl.store(subset_lse_ptr + lse_offsets, lse)
    else:
        tl.store(subset_output_ptr + subset_offsets, subset, mask=matrix_mask)
        tl.store(subset_lse_ptr + lse_offsets, lse, mask=head_mask)


@triton.jit
def _finalize_fp8ds_swa_subset_multihead_kernel(
    q_ptr,
    k_cache_ptr,
    token_to_req_indices_ptr,
    query_start_loc_ptr,
    seq_lens_ptr,
    block_table_ptr,
    subset_output_ptr,
    subset_lse_ptr,
    stride_q_t: tl.constexpr,
    stride_q_h: tl.constexpr,
    stride_q_d: tl.constexpr,
    block_table_stride,
    stride_subset_t: tl.constexpr,
    stride_subset_h: tl.constexpr,
    stride_subset_d: tl.constexpr,
    stride_lse_t: tl.constexpr,
    stride_lse_h: tl.constexpr,
    cache_block_size: tl.constexpr,
    token_data_size: tl.constexpr,
    block_stride: tl.constexpr,
    fp8_dim: tl.constexpr,
    scale_dim: tl.constexpr,
    quant_block: tl.constexpr,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    window_size: tl.constexpr,
    global_token_offset: tl.constexpr,
    scale: tl.constexpr,
    HEAD_BLOCK: tl.constexpr,
    BLOCK_D: tl.constexpr,
    FULL_TILE: tl.constexpr,
):
    token_idx = tl.program_id(0)
    head_block_idx = tl.program_id(1)
    head_offsets = head_block_idx * HEAD_BLOCK + tl.arange(0, HEAD_BLOCK)
    dim_offsets = tl.arange(0, BLOCK_D)
    head_mask = head_offsets < num_heads
    dim_mask = dim_offsets < head_dim
    matrix_mask = head_mask[:, None] & dim_mask[None, :]

    q_offsets = (
        token_idx * stride_q_t
        + head_offsets[:, None] * stride_q_h
        + dim_offsets[None, :] * stride_q_d
    )
    if tl.constexpr(FULL_TILE):
        q = tl.load(q_ptr + q_offsets).to(tl.float32)
    else:
        q = tl.load(
            q_ptr + q_offsets,
            mask=matrix_mask,
            other=0.0,
        ).to(tl.float32)
    running_max = tl.full((HEAD_BLOCK,), -float("inf"), tl.float32)
    running_denom = tl.zeros((HEAD_BLOCK,), tl.float32)
    running_acc = tl.zeros((HEAD_BLOCK, BLOCK_D), tl.float32)

    req_idx = tl.load(token_to_req_indices_ptr + token_idx)
    global_token_idx = token_idx + tl.full((), global_token_offset, tl.int32)
    query_start = tl.load(query_start_loc_ptr + req_idx)
    query_end = tl.load(query_start_loc_ptr + req_idx + 1)
    query_len = query_end - query_start
    seq_len = tl.load(seq_lens_ptr + req_idx)
    prefix_len = seq_len - query_len
    pos = prefix_len + global_token_idx - query_start
    start_pos = tl.where(pos >= (window_size - 1), pos - window_size + 1, 0)
    end_pos = pos + 1
    swa_len = end_pos - start_pos
    fp8_mask = dim_offsets < fp8_dim
    rope_mask = (dim_offsets >= fp8_dim) & dim_mask
    rope_offsets = tl.maximum(dim_offsets - fp8_dim, 0)

    for candidate_idx in range(0, window_size):
        is_valid = candidate_idx < swa_len
        if is_valid:
            pos_offset = start_pos + candidate_idx
            block_indices = pos_offset // cache_block_size
            block_numbers = tl.load(
                block_table_ptr + req_idx * block_table_stride + block_indices,
                mask=is_valid,
                other=0,
            )
            pos_in_block = pos_offset % cache_block_size
            cache_block_ptr = k_cache_ptr + block_numbers.to(tl.int64) * block_stride
            token_data_ptr = cache_block_ptr + pos_in_block * token_data_size
            token_scale_ptr = (
                cache_block_ptr
                + cache_block_size * token_data_size
                + pos_in_block * scale_dim
            )

            x_uint8 = tl.load(token_data_ptr + dim_offsets, mask=fp8_mask, other=0)
            x_fp8 = x_uint8.to(tl.float8e4nv, bitcast=True)
            x_float = x_fp8.to(tl.float32)
            scale_offsets = dim_offsets // quant_block
            encoded_scale = tl.load(
                token_scale_ptr + scale_offsets,
                mask=fp8_mask,
                other=127,
            )
            dequant_scale = tl.exp2(encoded_scale.to(tl.float32) - 127.0)
            x_dequant = x_float * dequant_scale

            rope_ptr = (token_data_ptr + fp8_dim).to(tl.pointer_type(tl.bfloat16))
            rope = tl.load(rope_ptr + rope_offsets, mask=rope_mask, other=0.0).to(
                tl.float32
            )
            kv = tl.where(fp8_mask, x_dequant, rope)
            kv = tl.where(dim_mask, kv, 0.0)

            score = tl.sum(q * kv[None, :], axis=1) * scale
            next_max = tl.maximum(running_max, score)
            previous_weight = tl.exp(running_max - next_max)
            candidate_weight = tl.exp(score - next_max)
            running_acc = (
                running_acc * previous_weight[:, None]
                + kv[None, :] * candidate_weight[:, None]
            )
            running_denom = running_denom * previous_weight + candidate_weight
            running_max = next_max

    denom_safe = tl.where(running_denom > 0.0, running_denom, 1.0)
    subset = running_acc / denom_safe[:, None]
    lse = tl.where(
        running_denom > 0.0,
        running_max + tl.log(running_denom),
        -float("inf"),
    )
    subset_offsets = (
        token_idx * stride_subset_t
        + head_offsets[:, None] * stride_subset_h
        + dim_offsets[None, :] * stride_subset_d
    )
    lse_offsets = token_idx * stride_lse_t + head_offsets * stride_lse_h
    if tl.constexpr(FULL_TILE):
        tl.store(subset_output_ptr + subset_offsets, subset)
        tl.store(subset_lse_ptr + lse_offsets, lse)
    else:
        tl.store(subset_output_ptr + subset_offsets, subset, mask=matrix_mask)
        tl.store(subset_lse_ptr + lse_offsets, lse, mask=head_mask)


def finalize_fp8ds_local_slots_sparse_mla_attention_subset_multihead(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    local_indices: torch.Tensor,
    token_to_req_indices: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    scale: float,
    subset_output: torch.Tensor,
    subset_lse: torch.Tensor,
    head_block_size: int = 16,
) -> None:
    if q.dim() == 4:
        assert q.shape[1] == 1
        q = q[:, 0]
    if local_indices.dim() == 3:
        assert local_indices.shape[1] == 1
        local_indices = local_indices[:, 0]

    token_fp8_dim = 448
    token_bf16_dim = 64
    token_scale_dim = 8
    quant_block_size = 64
    token_data_size = token_fp8_dim + token_bf16_dim * 2

    num_tokens, _, head_dim = q.shape
    num_heads = subset_lse.shape[1]
    num_candidates = local_indices.shape[1]
    block_d = min(1024, triton.next_power_of_2(head_dim))
    full_tile = num_heads % head_block_size == 0 and head_dim == block_d
    grid = (num_tokens, triton.cdiv(num_heads, head_block_size))
    _finalize_fp8ds_local_slots_subset_multihead_kernel[grid](
        q,
        k_cache,
        local_indices,
        token_to_req_indices,
        block_table,
        subset_output,
        subset_lse,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        local_indices.stride(0),
        local_indices.stride(1),
        block_table.stride(0),
        subset_output.stride(0),
        subset_output.stride(1),
        subset_output.stride(2),
        subset_lse.stride(0),
        subset_lse.stride(1),
        block_size,
        token_data_size,
        k_cache.stride(0),
        token_fp8_dim,
        token_scale_dim,
        quant_block_size,
        num_heads,
        head_dim,
        num_candidates,
        scale,
        HEAD_BLOCK=head_block_size,
        BLOCK_D=block_d,
        FULL_TILE=full_tile,
        num_warps=8,
    )


def finalize_fp8ds_swa_slots_sparse_mla_attention_subset_multihead(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    token_to_req_indices: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    window_size: int,
    global_token_offset: int,
    scale: float,
    subset_output: torch.Tensor,
    subset_lse: torch.Tensor,
    head_block_size: int = 16,
) -> None:
    if q.dim() == 4:
        assert q.shape[1] == 1
        q = q[:, 0]

    token_fp8_dim = 448
    token_bf16_dim = 64
    token_scale_dim = 8
    quant_block_size = 64
    token_data_size = token_fp8_dim + token_bf16_dim * 2

    num_tokens, _, head_dim = q.shape
    num_heads = subset_lse.shape[1]
    block_d = min(1024, triton.next_power_of_2(head_dim))
    full_tile = num_heads % head_block_size == 0 and head_dim == block_d
    grid = (num_tokens, triton.cdiv(num_heads, head_block_size))
    _finalize_fp8ds_swa_subset_multihead_kernel[grid](
        q,
        k_cache,
        token_to_req_indices,
        query_start_loc,
        seq_lens,
        block_table,
        subset_output,
        subset_lse,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        block_table.stride(0),
        subset_output.stride(0),
        subset_output.stride(1),
        subset_output.stride(2),
        subset_lse.stride(0),
        subset_lse.stride(1),
        block_size,
        token_data_size,
        k_cache.stride(0),
        token_fp8_dim,
        token_scale_dim,
        quant_block_size,
        num_heads,
        head_dim,
        window_size,
        global_token_offset,
        scale,
        HEAD_BLOCK=head_block_size,
        BLOCK_D=block_d,
        FULL_TILE=full_tile,
        num_warps=8,
    )

# Copyright 2025 XunhaoLai. All rights reserved.
#
# Decode-path sparse GQA attend for spec-decode verify batches
# (decode_query_len > 1): vLLM's ``minimax_m3_sparse_attn_decode``
# (minimax_m3/common/ops/sparse_attn.py) grafted onto sglang's
# ``_gqa_share_sparse_decode_kernel`` (decode/topk_sparse.py).
#
# The only structural change vs the stock sglang decode kernel is the query-token
# -> request mapping. A verify batch flattens ``num_reqs * decode_query_len``
# query tokens request-major (all dq tokens of req0, then req1, ...). Token
# ``pid_b`` belongs to request ``req_id = pid_b // decode_query_len`` at intra-
# request offset ``q_offset``. Its causal horizon is ``kv_len = seq_len -
# decode_query_len + q_offset + 1`` (linear chain). ``seq_lens`` here is
# per-REQUEST and INCLUDES the dq draft tokens (verify convention:
# seq_lens = prefix + dq), matching vLLM's ``seq_len`` convention so the
# position formula transfers verbatim.
#
# Correct only for a LINEAR draft chain (eagle-topk=1). A draft tree (topk > 1)
# needs spec_info.custom_mask and is not implemented.

from typing import Optional

import torch
import triton
import triton.language as tl

from sglang.srt.utils import is_hip

from ..common.utils import robust_allocator

_is_hip = is_hip()


@triton.heuristics(
    {
        "BLOCK_SIZE_H": lambda args: max(
            16, triton.next_power_of_2(args["gqa_group_size"])
        ),
        "BLOCK_SIZE_D": lambda args: triton.next_power_of_2(args["head_dim"]),
        "BLOCK_SIZE_T": lambda args: triton.next_power_of_2(args["max_topk"]),
        "TOTAL_Q_BUCKET": lambda args: triton.next_power_of_2(args["total_q"]),
    }
)
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=nw, num_stages=ns)
        for nw in [4, 8]
        for ns in [2, 3, 4, 5]
    ],
    key=["TOTAL_Q_BUCKET", "gqa_group_size", "head_dim", "block_size"],
)
@triton.jit(do_not_specialize=["decode_query_len"])
def _gqa_share_sparse_decode_qlen_kernel(
    q_ptr,  # Q: total_q x qh x d        (total_q = num_reqs * decode_query_len)
    k_cache_ptr,  # K paged: max_slots x kh x d
    v_cache_ptr,  # V paged: max_slots x kh x d
    req_to_token_ptr,  # req_to_token: max_reqs x max_kv_len
    idx_ptr,  # topk index: qh x total_q x topk (per query token)
    o_ptr,  # O partial: c x total_q x qh x d
    lse_ptr,  # lse partial: c x total_q x qh
    seq_lens,  # [num_reqs]  (per request; = prefix + decode_query_len)
    slot_ids,  # [num_reqs]
    # shape
    max_slots,
    total_q,
    gqa_group_size,
    head_dim,
    max_topk,
    max_kv_len,
    decode_query_len,
    # sm_scale
    sm_scale,
    # stride
    stride_q_b,
    stride_q_h,
    stride_q_d,
    stride_k_s,
    stride_k_h,
    stride_k_d,
    stride_v_s,
    stride_v_h,
    stride_v_d,
    stride_r2t_b,
    stride_ti_h,
    stride_ti_b,
    stride_ti_t,
    stride_o_c,
    stride_o_b,
    stride_o_h,
    stride_o_d,
    stride_l_c,
    stride_l_b,
    stride_l_h,
    # META parameters
    TOTAL_Q_BUCKET: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
    NUM_TOPK_CHUNKS: tl.constexpr,
    IS_FP8: tl.constexpr,
):
    # split-K over the topk dimension; pid(0) folds (query-token, chunk).
    pid_bc, pid_kh = tl.program_id(0), tl.program_id(1)
    pid_b = pid_bc % total_q  # flattened query-token id
    pid_c = pid_bc // total_q
    pid_h = pid_kh * gqa_group_size
    # map flattened query token -> request + intra-request offset
    req_id = pid_b // decode_query_len
    q_offset = pid_b - req_id * decode_query_len
    # per-chunk topk range (runtime, not constexpr)
    chunk_size_topk = (max_topk + NUM_TOPK_CHUNKS - 1) // NUM_TOPK_CHUNKS
    chunk_start_topk = pid_c * chunk_size_topk
    chunk_end_topk_compiletime = chunk_start_topk + chunk_size_topk
    # per-token causal horizon. seq_lens is per-REQUEST and includes the D draft
    # tokens, so query_pos = seq_len - decode_query_len + q_offset (== prefix +
    # q_offset) and this token attends to kv_len = query_pos + 1 positions.
    seq_len = tl.load(seq_lens + req_id)
    query_pos = seq_len - decode_query_len + q_offset
    kv_len = tl.maximum(query_pos + 1, 0)
    kv_len = tl.minimum(kv_len, max_kv_len)
    sid = (
        tl.load(slot_ids + req_id).to(tl.int64) + max_slots
    ) % max_slots  # to avoid bugs when slot_ids is negative
    # get real topk
    off_t = tl.arange(0, BLOCK_SIZE_T)
    idx_base = idx_ptr + pid_kh * stride_ti_h + pid_b * stride_ti_b
    topk_idx = tl.load(idx_base + off_t * stride_ti_t, mask=off_t < max_topk, other=-1)
    valid_idx = tl.where(topk_idx >= 0, off_t, -1)
    real_topk = tl.sum(valid_idx != -1, axis=0)
    chunk_end_topk = tl.minimum(chunk_end_topk_compiletime, real_topk)
    # init pointer
    off_n = tl.arange(0, BLOCK_SIZE_N)
    off_d = tl.arange(0, BLOCK_SIZE_D)
    dim_mask = off_d < head_dim
    # init statistics (no attention sink on the M3 main sparse attend)
    m_i = tl.full((BLOCK_SIZE_H,), float("-inf"), dtype=tl.float32)
    lse_i = tl.full((BLOCK_SIZE_H,), float("-inf"), dtype=tl.float32)
    q_ptrs = tl.make_block_ptr(
        base=q_ptr + pid_b * stride_q_b + pid_h * stride_q_h,
        shape=(gqa_group_size, head_dim),
        strides=(stride_q_h, stride_q_d),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_H, BLOCK_SIZE_D),
        order=(1, 0),
    )
    q = tl.load(q_ptrs, boundary_check=(0, 1), padding_option="zero")
    acc_o = tl.full((BLOCK_SIZE_H, BLOCK_SIZE_D), 0, dtype=tl.float32)
    # only iterate over this chunk's topk slice.
    cur_idx_ptr = idx_base + chunk_start_topk * stride_ti_t
    for _ in tl.range(chunk_start_topk, chunk_end_topk):
        # load index (block id) -> base token position of the block
        c = tl.load(cur_idx_ptr).to(tl.int32) * BLOCK_SIZE_N
        cur_idx_ptr = cur_idx_ptr + stride_ti_t
        # resolve slots for this block via req_to_token
        pos = c + off_n
        pos_mask = pos < kv_len
        slots = tl.load(
            req_to_token_ptr + sid * stride_r2t_b + pos,
            mask=pos_mask,
            other=0,
        ).to(tl.int64)
        slots = (slots + max_slots) % max_slots  # safety against negative
        # load K as (head_dim, BLOCK_SIZE_N) via indirect addressing
        k_off = (
            slots[None, :] * stride_k_s
            + pid_kh * stride_k_h
            + off_d[:, None] * stride_k_d
        )
        k = tl.load(
            k_cache_ptr + k_off,
            mask=dim_mask[:, None] & pos_mask[None, :],
            other=0.0,
        )
        if IS_FP8:
            k = k.to(q.dtype)
        # load V as (BLOCK_SIZE_N, head_dim) via indirect addressing
        v_off = (
            slots[:, None] * stride_v_s
            + pid_kh * stride_v_h
            + off_d[None, :] * stride_v_d
        )
        v = tl.load(
            v_cache_ptr + v_off,
            mask=pos_mask[:, None] & dim_mask[None, :],
            other=0.0,
        )
        if IS_FP8:
            v = v.to(q.dtype)
        # compute qk with per-token causal mask
        qk = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_N), dtype=tl.float32)
        qk += tl.where(off_n[None, :] < kv_len - c, 0, float("-inf"))
        qk += tl.dot(q, k) * sm_scale
        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, axis=1)
        acc_o_scale = tl.exp(m_i - m_ij)
        acc_o = acc_o * acc_o_scale[:, None]
        p = p.to(v.dtype)
        acc_o += tl.dot(p.to(v.dtype), v)
        m_i = m_ij
        lse_i = m_ij + tl.log(tl.exp(lse_i - m_ij) + l_ij)
    # final scale; empty chunks (chunk_start_topk >= real_topk) emit clean zero.
    scale = tl.where(
        lse_i > float("-inf"),
        tl.exp(m_i - lse_i),
        tl.zeros_like(lse_i),
    )
    acc_o = acc_o * scale[:, None]
    o_ptrs = tl.make_block_ptr(
        base=o_ptr + pid_c * stride_o_c + pid_b * stride_o_b + pid_h * stride_o_h,
        shape=(gqa_group_size, head_dim),
        strides=(stride_o_h, stride_o_d),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_H, BLOCK_SIZE_D),
        order=(1, 0),
    )
    tl.store(o_ptrs, acc_o.to(o_ptr.dtype.element_ty), boundary_check=(0, 1))
    lse_ptrs = tl.make_block_ptr(
        base=lse_ptr + pid_c * stride_l_c + pid_b * stride_l_b + pid_h * stride_l_h,
        shape=(gqa_group_size,),
        strides=(stride_l_h,),
        offsets=(0,),
        block_shape=(BLOCK_SIZE_H,),
        order=(0,),
    )
    tl.store(lse_ptrs, lse_i.to(lse_ptr.dtype.element_ty), boundary_check=(0,))


@triton.heuristics(
    {
        "BLOCK_SIZE_D": lambda args: triton.next_power_of_2(args["head_dim"]),
    }
)
@triton.jit
def _merge_topk_attn_out_qlen_kernel(
    o_ptr,  # [NUM_TOPK_CHUNKS, total_q, NQH, D] — partials in, merged out at chunk 0
    lse_ptr,  # [NUM_TOPK_CHUNKS, total_q, NQH]
    head_dim,
    stride_o_c,
    stride_o_b,
    stride_o_h,
    stride_o_d,
    stride_l_c,
    stride_l_b,
    stride_l_h,
    NUM_TOPK_CHUNKS: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    pid_b, pid_h = tl.program_id(0), tl.program_id(1)
    off_c = tl.arange(0, NUM_TOPK_CHUNKS)
    off_d = tl.arange(0, BLOCK_SIZE_D)
    o_ptrs = tl.make_block_ptr(
        base=o_ptr + pid_b * stride_o_b + pid_h * stride_o_h,
        shape=(NUM_TOPK_CHUNKS, head_dim),
        strides=(stride_o_c, stride_o_d),
        offsets=(0, 0),
        block_shape=(NUM_TOPK_CHUNKS, BLOCK_SIZE_D),
        order=(1, 0),
    )
    lse_ptrs = lse_ptr + pid_b * stride_l_b + pid_h * stride_l_h + off_c * stride_l_c
    o = tl.load(o_ptrs, boundary_check=(0, 1), padding_option="zero")
    lse = tl.load(lse_ptrs)  # empty chunks contribute -inf -> weight 0
    lse_max = tl.max(lse, axis=0)
    weights = tl.exp(lse - lse_max)
    weights = weights / tl.sum(weights, axis=0)
    o_merged = tl.sum(o * weights[:, None], axis=0)
    o_out_ptrs = o_ptr + pid_b * stride_o_b + pid_h * stride_o_h + off_d * stride_o_d
    tl.store(o_out_ptrs, o_merged.to(o_ptr.dtype.element_ty), mask=off_d < head_dim)


@torch.no_grad()
def flash_decode_with_gqa_share_sparse_qlen(
    q: torch.Tensor,  # [total_q, num_q_heads, head_dim]  (total_q = num_reqs*dq)
    k_cache: torch.Tensor,  # [max_slots, num_kv_heads, head_dim] (paged)
    v_cache: torch.Tensor,  # [max_slots, num_kv_heads, head_dim] (paged)
    req_to_token: torch.Tensor,  # [max_reqs, max_kv_len]
    seq_lens: torch.Tensor,  # [num_reqs] int (prefix + decode_query_len)
    slot_ids: torch.Tensor,  # [num_reqs]
    block_size: int,
    topk_idx: torch.Tensor,  # [num_kv_heads, total_q, topk]
    decode_query_len: int,
    sm_scale: Optional[float] = None,
) -> torch.Tensor:
    """Block-sparse GQA decode attend for verify (decode_query_len query tokens
    per request, flattened request-major). Linear-causal (eagle-topk=1)."""
    triton.set_allocator(robust_allocator)
    assert q.dtype in (torch.bfloat16, torch.float16)
    _FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2, torch.float8_e4m3fnuz)
    is_fp8 = _is_hip and k_cache.dtype in _FP8_DTYPES
    assert k_cache.dtype == q.dtype or is_fp8, (
        f"sparse decode expects K cache dtype == Q dtype ({q.dtype}) "
        f"or fp8 on HIP, got {k_cache.dtype}"
    )
    assert v_cache.dtype == k_cache.dtype
    total_q, num_q_heads, head_dim = q.shape
    max_slots, num_kv_heads, _ = k_cache.shape
    num_reqs = seq_lens.shape[0]
    assert slot_ids.shape[0] == num_reqs
    assert total_q == num_reqs * decode_query_len, (
        f"total_q ({total_q}) != num_reqs ({num_reqs}) * decode_query_len "
        f"({decode_query_len})"
    )
    assert topk_idx.shape[0] == num_kv_heads
    assert topk_idx.shape[1] == total_q
    assert triton.next_power_of_2(block_size) == block_size
    max_kv_len = req_to_token.shape[1]
    assert num_q_heads % num_kv_heads == 0
    gqa_group_size = num_q_heads // num_kv_heads
    max_topk = topk_idx.shape[2]
    if sm_scale is None:
        sm_scale = head_dim**-0.5
    # NUM_TOPK_CHUNKS: shape-constant (fixed grid within a cuda graph). The
    # split-K chunk count sets the bf16 reduction order, so for the verify path
    # (dq > 1) to be numerically identical to the stock decode path (dq = 1, the
    # spec-off reference) the chunk count must be computed from the number of
    # REQUESTS — what stock decode calls batch_size — not the flattened
    # query-token count.
    TARGET_GRID = 256
    target = max(1, min(max_topk, TARGET_GRID // max(1, num_reqs * num_kv_heads)))
    NUM_TOPK_CHUNKS = 1 << (target.bit_length() - 1)
    o_partial = torch.empty(
        NUM_TOPK_CHUNKS, total_q, num_q_heads, head_dim, dtype=q.dtype, device=q.device
    )
    lse_partial = torch.empty(
        NUM_TOPK_CHUNKS, total_q, num_q_heads, dtype=torch.float32, device=q.device
    )
    grid = (total_q * NUM_TOPK_CHUNKS, num_kv_heads)
    _gqa_share_sparse_decode_qlen_kernel[grid](
        q,
        k_cache,
        v_cache,
        req_to_token,
        topk_idx,
        o_partial,
        lse_partial,
        seq_lens,
        slot_ids,
        max_slots,
        total_q,
        gqa_group_size,
        head_dim,
        max_topk,
        max_kv_len,
        decode_query_len,
        sm_scale,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        v_cache.stride(0),
        v_cache.stride(1),
        v_cache.stride(2),
        req_to_token.stride(0),
        topk_idx.stride(0),
        topk_idx.stride(1),
        topk_idx.stride(2),
        o_partial.stride(0),
        o_partial.stride(1),
        o_partial.stride(2),
        o_partial.stride(3),
        lse_partial.stride(0),
        lse_partial.stride(1),
        lse_partial.stride(2),
        BLOCK_SIZE_N=block_size,
        NUM_TOPK_CHUNKS=NUM_TOPK_CHUNKS,
        IS_FP8=is_fp8,
    )
    merge_grid = (total_q, num_q_heads)
    _merge_topk_attn_out_qlen_kernel[merge_grid](
        o_partial,
        lse_partial,
        head_dim,
        o_partial.stride(0),
        o_partial.stride(1),
        o_partial.stride(2),
        o_partial.stride(3),
        lse_partial.stride(0),
        lse_partial.stride(1),
        lse_partial.stride(2),
        NUM_TOPK_CHUNKS=NUM_TOPK_CHUNKS,
    )
    return o_partial[0].contiguous()

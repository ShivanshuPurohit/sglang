"""Unit tests for the MiniMax-M3 qlen sparse decode attend — no server, no model loading.

flash_decode_with_gqa_share_sparse_qlen serves spec-decode verify batches
(decode_query_len query tokens per request, flattened request-major). Three proofs:

  1. dq=1 equivalence: with decode_query_len=1 the qlen kernel reduces to the stock
     decode kernel (req_id == pid_b, kv_len == seq_len) and must match it.
  2. dq>1 reference correctness: token j of a request attends causally to
     kv_len = prefix + j + 1 positions, restricted to its selected top-k blocks,
     matching a per-token PyTorch reference.
  3. dq>1 stock-decode equivalence: each verify token must match the stock decode
     kernel run on a single-token batch with the same KV cache, req_to_token row,
     selected blocks, and causal length. This is the losslessness gate for the
     speculative verify path: if verify logits differ from what spec-off decode
     would produce, greedy argmax can flip and spec-on output diverges from
     spec-off. (Measured 0.0 on the validated config; the tolerance allows for
     autotune-config differences across arches.)
"""

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=8, stage="base-b", runner_config="1-gpu-small")

import unittest

import torch

from sglang.srt.layers.attention.minimax_sparse_ops.decode.topk_sparse import (
    flash_decode_with_gqa_share_sparse,
)
from sglang.srt.layers.attention.minimax_sparse_ops.decode.topk_sparse_qlen import (
    flash_decode_with_gqa_share_sparse_qlen,
)
from sglang.test.test_utils import CustomTestCase

DEVICE = "cuda"
RTOL = 5e-3
ATOL = 5e-3
# Per-token verify-vs-stock-decode budget: safely below the scale at which
# greedy argmax flips were observed (a mismatched split-K chunk count produced
# 3.9e-3 divergence and occasional flips; the fixed kernel measured 0.0).
STOCK_EQUIV_TOL = 3e-3


def _build_verify_inputs(
    num_reqs, dq, nqh, nkh, hd, prefix_list, blk, topk, dtype=torch.bfloat16
):
    """A spec verify batch: num_reqs requests, each with dq flattened query tokens.

    seq_lens[i] = prefix_list[i] + dq (total KV length INCLUDING the dq drafts).
    With dq=1 and prefix_list = seq_lens - 1 this degenerates to a decode batch.
    """
    seq_lens_list = [p + dq for p in prefix_list]
    total_q = num_reqs * dq
    max_kv_len = max(seq_lens_list)
    max_slots = num_reqs * max_kv_len
    q = torch.randn(total_q, nqh, hd, dtype=dtype, device=DEVICE)
    k_cache = torch.randn(max_slots, nkh, hd, dtype=dtype, device=DEVICE)
    v_cache = torch.randn(max_slots, nkh, hd, dtype=dtype, device=DEVICE)
    req_to_token = torch.zeros(num_reqs, max_kv_len, dtype=torch.int32, device=DEVICE)
    slot_ids = torch.arange(num_reqs, dtype=torch.int64, device=DEVICE)
    seq_lens = torch.tensor(seq_lens_list, dtype=torch.int32, device=DEVICE)
    for i in range(num_reqs):
        base = i * max_kv_len
        req_to_token[i, :max_kv_len] = (
            torch.randperm(max_kv_len, device=DEVICE) + base
        ).to(torch.int32)
    # Per-token top-k: token j of req i attends to kv_len = prefix_i + j + 1
    # positions, so it may only select among the first ceil(kv_len / blk) blocks.
    topk_idx = torch.full((nkh, total_q, topk), -1, dtype=torch.int32, device=DEVICE)
    for kh in range(nkh):
        for i in range(num_reqs):
            for j in range(dq):
                pid_b = i * dq + j
                kv_len = prefix_list[i] + j + 1
                nb = (kv_len + blk - 1) // blk
                ak = min(topk, nb)
                topk_idx[kh, pid_b, :ak] = torch.randperm(nb, device=DEVICE)[:ak].to(
                    torch.int32
                )
    return q, k_cache, v_cache, req_to_token, seq_lens, slot_ids, topk_idx


def _verify_reference(
    q, k_cache, v_cache, req_to_token, prefix_list, dq, blk, topk_idx
):
    """Per-token sparse attention reference: token pid_b = (i, j) attends to
    positions < kv_len = prefix_i + j + 1 within its selected blocks."""
    total_q, nqh, hd = q.shape
    nkh = k_cache.shape[1]
    gqa = nqh // nkh
    topk = topk_idx.shape[2]
    sm_scale = hd**-0.5
    out = torch.zeros(total_q, nqh, hd, dtype=torch.float32, device=q.device)
    for i in range(len(prefix_list)):
        for j in range(dq):
            pid_b = i * dq + j
            kv_len = prefix_list[i] + j + 1
            for kh in range(nkh):
                positions = []
                for t in range(topk):
                    bi = topk_idx[kh, pid_b, t].item()
                    if bi < 0:
                        continue
                    start = bi * blk
                    end = min(start + blk, kv_len)
                    if end > start:
                        positions.extend(range(start, end))
                if not positions:
                    continue
                pos = torch.tensor(positions, device=q.device, dtype=torch.long)
                slots = req_to_token[i, pos].long()
                k = k_cache[slots, kh].float()  # [P, hd]
                v = v_cache[slots, kh].float()
                for g in range(gqa):
                    h = kh * gqa + g
                    qk = (q[pid_b, h].float() @ k.transpose(-1, -2)) * sm_scale
                    out[pid_b, h] = torch.softmax(qk, dim=-1) @ v
    return out


@unittest.skipIf(not torch.cuda.is_available(), "CUDA is required")
class TestMiniMaxSparseQlen(CustomTestCase):
    def test_dq1_matches_stock_decode(self):
        """decode_query_len=1 must reduce to the stock decode kernel."""
        torch.manual_seed(0)
        for bs, nqh, nkh, hd, blk, tk, seqs in [
            (2, 8, 1, 128, 64, 16, [1024, 1024]),
            (4, 8, 1, 128, 64, 32, [513, 1023, 257, 769]),
            (2, 16, 1, 128, 64, 32, [4096, 2048]),
            (3, 32, 8, 128, 64, 16, [1500, 900, 2100]),
        ]:
            with self.subTest(bs=bs, nqh=nqh, nkh=nkh, tk=tk, seqs=seqs):
                prefixes = [s - 1 for s in seqs]
                q, kc, vc, r2t, sl, sid, ti = _build_verify_inputs(
                    bs, 1, nqh, nkh, hd, prefixes, blk, tk
                )
                o_stock = flash_decode_with_gqa_share_sparse(
                    q, None, kc, vc, r2t, sl, sid, blk, ti
                )
                o_qlen = flash_decode_with_gqa_share_sparse_qlen(
                    q, kc, vc, r2t, sl, sid, blk, ti, decode_query_len=1
                )
                torch.testing.assert_close(
                    o_qlen.float(), o_stock.float(), rtol=RTOL, atol=ATOL
                )

    def test_verify_matches_reference(self):
        """decode_query_len>1 must match the per-token causal reference."""
        torch.manual_seed(1)
        for num_reqs, dq, nqh, nkh, hd, blk, tk, prefixes in [
            (2, 4, 8, 1, 128, 64, 16, [500, 1000]),  # eagle chain dq=4
            (3, 4, 16, 1, 128, 64, 32, [300, 800, 1500]),
            (2, 5, 32, 8, 128, 64, 16, [700, 1200]),  # dq=5, GQA 32:8
            (1, 4, 8, 1, 128, 64, 8, [64]),  # short: prefix < blk
            (4, 2, 8, 1, 128, 64, 16, [128, 256, 384, 512]),  # dq=2
        ]:
            with self.subTest(num_reqs=num_reqs, dq=dq, nqh=nqh, nkh=nkh, tk=tk):
                q, kc, vc, r2t, sl, sid, ti = _build_verify_inputs(
                    num_reqs, dq, nqh, nkh, hd, prefixes, blk, tk
                )
                o_kernel = flash_decode_with_gqa_share_sparse_qlen(
                    q, kc, vc, r2t, sl, sid, blk, ti, decode_query_len=dq
                )
                o_ref = _verify_reference(q, kc, vc, r2t, prefixes, dq, blk, ti)
                torch.testing.assert_close(
                    o_kernel.float(), o_ref, rtol=RTOL, atol=ATOL
                )

    def test_verify_per_token_matches_stock_decode(self):
        """The losslessness gate: each verify token vs stock decode on a
        single-token batch with kv_len = prefix + j + 1 and the same KV cache,
        req_to_token row, and selected blocks."""
        torch.manual_seed(0)
        for num_reqs, dq, nqh, nkh, hd, blk, tk, prefixes in [
            (1, 4, 8, 1, 128, 64, 16, [500]),
            (2, 4, 16, 1, 128, 64, 32, [300, 800]),
            (1, 4, 8, 1, 128, 64, 8, [64]),  # short 1-block
            (2, 2, 32, 8, 128, 64, 16, [128, 256]),  # the case that caught the
            # split-K chunk-count bug (GQA 32:8, multi-block)
            (1, 5, 8, 1, 128, 64, 16, [1000]),
        ]:
            with self.subTest(num_reqs=num_reqs, dq=dq, nqh=nqh, nkh=nkh, tk=tk):
                q, kc, vc, r2t, sl, sid, ti = _build_verify_inputs(
                    num_reqs, dq, nqh, nkh, hd, prefixes, blk, tk
                )
                o_qlen = flash_decode_with_gqa_share_sparse_qlen(
                    q, kc, vc, r2t, sl, sid, blk, ti, decode_query_len=dq
                )
                for i in range(num_reqs):
                    for j in range(dq):
                        pid_b = i * dq + j
                        kv_len = prefixes[i] + j + 1
                        o_stock = flash_decode_with_gqa_share_sparse(
                            q[pid_b : pid_b + 1].contiguous(),
                            None,
                            kc,
                            vc,
                            r2t,
                            torch.tensor([kv_len], dtype=torch.int32, device=DEVICE),
                            torch.tensor([i], dtype=torch.int64, device=DEVICE),
                            blk,
                            ti[:, pid_b : pid_b + 1, :].contiguous(),
                        )
                        diff = (
                            (o_qlen[pid_b].float() - o_stock[0].float())
                            .abs()
                            .max()
                            .item()
                        )
                        self.assertLessEqual(
                            diff,
                            STOCK_EQUIV_TOL,
                            f"verify token (req {i}, offset {j}) diverges from "
                            f"stock decode by {diff:.3e}",
                        )


if __name__ == "__main__":
    unittest.main()

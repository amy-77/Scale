"""Sparse prefill contracts: config, ranked-leaf expansion, admission, and (GPU) Top-K vs dense."""
import dataclasses
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.attention.nsa.adaptive_hisa import prefill_select as ps
from sglang.srt.layers.attention.nsa.adaptive_hisa.config import PartitionConfig, PartitionConfigError


class TestSparsePrefillConfig(unittest.TestCase):
    def test_flag_and_dependencies(self):
        cfg = PartitionConfig(mode='adaptive_decode').validate()
        self.assertFalse(cfg.sparse_prefill)
        cfg = PartitionConfig(mode='adaptive_decode', sparse_prefill=True).validate()
        self.assertIn('sparse_prefill=True@2048rows', cfg.describe())
        for overrides in [dict(build_summaries=False), dict(sparse_prefill_rows=0)]:
            with self.assertRaises(PartitionConfigError):
                dataclasses.replace(cfg, **overrides).validate()

    def test_env_parsing(self):
        from sglang.srt.layers.attention.nsa.adaptive_hisa.config import config_from_env

        env = {'SGLANG_NSA_ADAPTIVE_HISA_MODE': 'adaptive_decode',
               'SGLANG_NSA_ADAPTIVE_HISA_SPARSE_PREFILL': '1',
               'SGLANG_NSA_ADAPTIVE_HISA_SPARSE_PREFILL_ROWS': '512'}
        with patch.dict('os.environ', env, clear=False):
            cfg = config_from_env()
        self.assertTrue(cfg.sparse_prefill)
        self.assertEqual(cfg.sparse_prefill_rows, 512)
        self.assertEqual(cfg.prefill_candidate_tokens, cfg.candidate_tokens)  # 0 -> decode budget
        env['SGLANG_NSA_ADAPTIVE_HISA_SPARSE_PREFILL_CANDIDATES'] = '16384'
        with patch.dict('os.environ', env, clear=False):
            cfg = config_from_env()
        self.assertEqual(cfg.prefill_candidate_tokens, 16384)
        self.assertNotEqual(cfg.candidate_tokens, 16384)
        with self.assertRaises(PartitionConfigError):
            dataclasses.replace(cfg, sparse_prefill_candidates=1000).validate()


@unittest.skipUnless(torch.cuda.is_available(), 'expansion is a Triton kernel')
class TestExpandCandidates(unittest.TestCase):
    """Ranked-leaf expansion kernel against a Python reference."""

    def _reference(self, order, starts, lengths, budget, n_complete, local_len):
        out = []
        for leaf in order:
            if len(out) >= budget:
                break
            take = min(lengths[leaf], budget - len(out))
            out.extend(range(starts[leaf], starts[leaf] + take))
        out.extend(range(n_complete, n_complete + local_len))
        return out

    def test_matches_reference_with_clipping_and_padding(self):
        torch.manual_seed(3)
        lengths = [40, 100, 64, 9, 300, 1, 64, 200, 0, 0]  # two padding leaves
        starts = [sum(lengths[:i]) for i in range(len(lengths))]
        n_complete = sum(lengths)
        for budget in (64, 300, 512, 700):
            for local_len in (0, 5):
                scores = torch.randn(3, len(lengths), device='cuda')
                scores[:, 8:] = float('-inf')  # padding rows never rank
                ls = torch.tensor(starts, dtype=torch.int32, device='cuda')
                ll = torch.tensor(lengths, dtype=torch.int32, device='cuda')
                order, inc = ps.rank_leaves(scores, ls, ll)
                cand = ps.expand_candidates(order, inc, ls, ll, slice(0, 3), budget, n_complete, local_len).cpu()
                self.assertEqual(cand.shape, (3, budget + local_len))
                self.assertEqual(cand.dtype, torch.int32)
                for r in range(3):
                    ref = self._reference(order[r].tolist(), starts, lengths, budget, n_complete, local_len)
                    self.assertEqual(cand[r].tolist(), ref, msg=f'budget={budget} row={r}')
                    ids = cand[r, :budget].tolist()
                    self.assertEqual(len(set(ids)), budget)  # no duplicates inside the budget

    def test_row_subset(self):
        lengths = [64] * 16
        starts = [64 * i for i in range(16)]
        scores = torch.arange(16, dtype=torch.float32, device='cuda').repeat(4, 1)
        scores[1] = -scores[1]
        ls = torch.tensor(starts, dtype=torch.int32, device='cuda')
        ll = torch.tensor(lengths, dtype=torch.int32, device='cuda')
        order, inc = ps.rank_leaves(scores, ls, ll)
        cand = ps.expand_candidates(order, inc, ls, ll, slice(1, 3), 128, 1024, 0).cpu()
        # candidates come out in rank order (best leaf first), tokens ascending inside a leaf
        self.assertEqual(cand[0].tolist(), list(range(0, 128)))                              # row 1: leaves 0, 1
        self.assertEqual(cand[1].tolist(), list(range(960, 1024)) + list(range(896, 960)))   # row 2: leaves 15, 14


@unittest.skipUnless(torch.cuda.is_available(), 'batched weighted select is a CUDA op')
class TestWeightedSelect(unittest.TestCase):
    """Batched weighted radix select == (score desc, index asc) prefix with a clipped crossing leaf."""

    def _reference(self, scores_row, starts, lengths, budget):
        order = sorted(range(len(lengths)), key=lambda i: (-scores_row[i], i))
        out, rem = [], budget
        for i in order:
            if rem <= 0 or lengths[i] <= 0 or scores_row[i] == float('-inf'):
                continue
            take = min(rem, lengths[i])
            out.extend(range(starts[i], starts[i] + take))
            rem -= take
        return sorted(out)

    def _case(self, n_q, lengths, budget, local_len, ties=False, seed=0):
        torch.manual_seed(seed)
        n = len(lengths)
        cap = n + 3  # padding columns
        starts = [sum(lengths[:i]) for i in range(n)]
        n_complete = sum(lengths)
        ls = torch.zeros(cap, dtype=torch.int32, device='cuda'); ls[:n] = torch.tensor(starts)
        ll = torch.zeros(cap, dtype=torch.int32, device='cuda'); ll[:n] = torch.tensor(lengths)
        raw_scores = (torch.randint(0, 4, (n_q, cap), device='cuda').float() if ties
                      else torch.randn(n_q, cap, device='cuda'))
        scores = raw_scores.clone()
        scores[:, n:] = float('-inf')
        scores[:, ls[:cap] < 64] = float('inf')  # sink leaves forced first (as coarse_leaf_scores does)
        scores[:, n:] = float('-inf')
        nl = torch.tensor([n], dtype=torch.int32, device='cuda')
        cand, seg, seg_count = ps.select_candidates_weighted(
            scores, ls, ll, nl, slice(0, n_q), budget, n_complete, local_len, intervals=True)
        # Production path leaves the Tensor-Core score matrix untouched and
        # fuses sink promotion / padding rejection into the selector.
        fused = ps.select_candidates_weighted(
            raw_scores, ls, ll, nl, slice(0, n_q), budget, n_complete, local_len, sink=64)
        self.assertTrue(torch.equal(fused, cand))
        self.assertEqual(cand.shape, (n_q, budget + local_len))
        cand = cand.cpu(); seg = [t.cpu() for t in seg]; seg_count = seg_count.cpu()
        for r in range(n_q):
            ref = self._reference(scores[r].tolist(), starts, lengths, budget)
            got = cand[r, :budget].tolist()
            self.assertEqual(sorted(v for v in got if v >= 0), ref, msg=f'row {r}')
            self.assertTrue(all(v == -1 for v in got[len(ref):]))  # only when the leaves hold < budget tokens
            self.assertEqual(cand[r, budget:].tolist(), list(range(n_complete, n_complete + local_len)))
            segs = sorted((int(seg[2][r, j]), int(seg[0][r, j]), int(seg[1][r, j])) for j in range(int(seg_count[r])))
            toks = []
            for off, st, ln in segs:
                self.assertEqual(off, len(toks)); self.assertGreater(ln, 0)
                toks.extend(range(st, st + ln))
            self.assertEqual(toks, got[: len(toks)])
        if not ties and n_complete >= budget:  # legacy argsort path: same token set (distinct scores)
            order, inc = ps.rank_leaves(scores, ls, ll)
            old = ps.expand_candidates(order, inc, ls, ll, slice(0, n_q), budget, n_complete, local_len).cpu()
            self.assertTrue(torch.equal(old[:, :budget].sort(1).values, cand[:, :budget].sort(1).values))

    def test_clipping_padding_and_local_window(self):
        self._case(5, [40, 100, 64, 9, 300, 1, 64, 200], 300, 7)

    def test_ties_are_index_ascending(self):
        self._case(6, [32] * 50 + [64] * 30, 1000, 0, ties=True)

    def test_budget_exceeds_leaves_pads_minus_one(self):
        self._case(3, [64] * 10, 1000, 4)

    def test_large_row(self):
        torch.manual_seed(1)
        lengths = torch.randint(1, 200, (1900,)).tolist()
        self._case(4, lengths, 8192, 100, seed=2)

    def test_query_local_leaf_is_forced(self):
        starts = torch.arange(0, 384, 64, dtype=torch.int32, device='cuda')
        lengths = torch.full((6,), 64, dtype=torch.int32, device='cuda')
        # Local leaves deliberately have the worst scores. The leaf containing
        # position (ctx - 1) must still win the one-leaf token budget.
        scores = torch.tensor(
            [[10., 9., 8., 7., -10., -20.], [10., 9., 8., 7., -10., -20.]],
            device='cuda',
        )
        ctx = torch.tensor([257, 321], dtype=torch.int32, device='cuda')
        out = ps.select_candidates_weighted(
            scores,
            starts,
            lengths,
            torch.tensor([6], dtype=torch.int32, device='cuda'),
            slice(0, 2),
            64,
            256,
            0,
            force_positions=ctx,
        )
        self.assertEqual(out[0].cpu().tolist(), list(range(256, 320)))
        self.assertEqual(out[1].cpu().tolist(), list(range(320, 384)))


class TestAdmission(unittest.TestCase):
    def _batch(self, seq_len=20000, extend=3616, bs=1, final=False):
        return SimpleNamespace(
            batch_size=bs, seq_lens_cpu=torch.tensor([seq_len]), extend_seq_lens_cpu=torch.tensor([extend]),
            prefill_final_cpu=[final],
            req_pool_indices_cpu=[3], req_pool_indices=torch.tensor([3]),
            forward_mode=SimpleNamespace(is_extend_without_speculative=lambda: True,
                                         is_split_prefill=lambda: False, is_dllm_extend=lambda: False),
            spec_algorithm=None, attn_cp_metadata=None,
        )

    def test_admission_rules(self):
        cfg = PartitionConfig(mode='adaptive_decode', sparse_prefill=True).validate()
        entry = SimpleNamespace(capacity=128, epoch=0, n_complete=16384, n_tokens=16384)
        meta = SimpleNamespace(force_unfused_topk=True)
        with patch.object(ps, 'get_config', return_value=cfg), \
                patch.object(ps, '_fused_topk_requested', return_value=False), \
                patch('sglang.srt.layers.attention.nsa.adaptive_hisa.prefill_runtime._admission_skip', return_value=None), \
                patch('sglang.srt.layers.attention.nsa.adaptive_hisa.prefill_runtime.get_summary_entry', return_value=entry), \
                patch('sglang.srt.layers.attention.nsa.adaptive_hisa.prefill_runtime.STATE') as state:
            state.epoch.return_value = 0
            got = ps.sparse_prefill_admission(self._batch(), 5, meta)
            self.assertIsNotNone(got)
            self.assertEqual(got[2:], (20000 - 3616, 20000, cfg.candidate_tokens))
            # final chunk: own budget, or dense when requested
            final_cfg = dataclasses.replace(cfg, sparse_prefill_final_candidates=16384).validate()
            with patch.object(ps, 'get_config', return_value=final_cfg):
                self.assertEqual(ps.sparse_prefill_admission(self._batch(final=True), 5, meta)[4], 16384)
                self.assertEqual(ps.sparse_prefill_admission(self._batch(), 5, meta)[4], cfg.candidate_tokens)
                entry.n_complete = 16000  # final budget no longer fits -> dense for the final chunk only
                self.assertIsNone(ps.sparse_prefill_admission(self._batch(final=True), 5, meta))
                self.assertIsNotNone(ps.sparse_prefill_admission(self._batch(), 5, meta))
                entry.n_complete = 16384
            with patch.object(ps, 'get_config', return_value=dataclasses.replace(cfg, sparse_prefill_dense_final=True)):
                self.assertIsNone(ps.sparse_prefill_admission(self._batch(final=True), 5, meta))
                self.assertIsNotNone(ps.sparse_prefill_admission(self._batch(), 5, meta))
            # sealed prefix shorter than the candidate budget -> dense
            entry.n_complete = 8000
            self.assertIsNone(ps.sparse_prefill_admission(self._batch(), 5, meta))
            entry.n_complete = 16384
            # partition newer than this chunk's start (must never happen) -> dense
            self.assertIsNone(ps.sparse_prefill_admission(self._batch(seq_len=16400, extend=100), 5, meta))
            # stale epoch -> dense
            state.epoch.return_value = 1
            self.assertIsNone(ps.sparse_prefill_admission(self._batch(), 5, meta))
            state.epoch.return_value = 0
            # flag off -> dense
            with patch.object(ps, 'get_config', return_value=dataclasses.replace(cfg, sparse_prefill=False)):
                self.assertIsNone(ps.sparse_prefill_admission(self._batch(), 5, meta))


@unittest.skipUnless(torch.cuda.is_available(), 'requires CUDA, DeepGEMM and the fused Top-K op')
class TestSparsePrefillGPU(unittest.TestCase):
    """Fine stage + Top-K must reproduce the dense Top-2048 restricted to the candidate set."""

    def test_split_topk_equals_fused_topk_on_concatenation(self):
        """Top-K over (leaf logits | window logits) read in place selects the
        same tokens as the fused Top-K over their torch.cat, including rows
        whose valid prefix ends inside the window, a strided window slice and
        rows with fewer than 2048 valid columns."""
        from sglang.srt.layers.attention.nsa.hisa.hisa_topk_fused import (
            hisa_topk_candidates_fused,
            hisa_topk_candidates_split,
        )

        torch.manual_seed(5)
        dev = torch.device('cuda')
        R, budget, width, n_complete = 6, 8192, 8448, 65536
        # Gaussian logits: the values are distinct (no tie at the threshold) and
        # spread over several FP32 high-byte buckets, which HISA's radix kernel
        # needs (a single huge threshold bucket overflows its shared buffer and
        # the fused kernel itself becomes inexact — a pre-existing limit).
        score_a = torch.randn(R, budget, device=dev)
        score_b_full = torch.randn(R, width + 64, device=dev)
        score_b = score_b_full[:, :width]  # strided (row stride width + 64)
        cand_a = torch.randperm(n_complete, device=dev)[: R * budget].view(R, budget).to(torch.int32)
        ke = torch.tensor([width, 4000, 1, 0, 7000, width - 1], dtype=torch.int32, device=dev)
        ctx = n_complete + ke
        lengths = (budget + ke).contiguous()
        # row 3: fewer than 2048 valid columns overall -> identity path with -1 padding
        lengths[3] = 1500
        split = hisa_topk_candidates_split(score_a, score_b, lengths, cand_a, n_complete, ctx)
        local_ids = torch.arange(n_complete, n_complete + width, dtype=torch.int32, device=dev)
        fused = hisa_topk_candidates_fused(
            torch.cat((score_a, score_b), 1).contiguous(),
            torch.cat((cand_a, local_ids.expand(R, -1)), 1).contiguous(),
            lengths, ctx, None,
        )
        # order is not specified (radix + atomics); compare sorted rows
        self.assertTrue(torch.equal(split.sort(1).values, fused.sort(1).values))
        self.assertEqual(int((split[3] == -1).sum()), 2048 - 1500)
        # and both equal the exact Top-K over the valid prefix of the concatenation
        score_cat = torch.cat((score_a, score_b), 1)
        cand_cat = torch.cat((cand_a, local_ids.expand(R, -1)), 1)
        for r in range(R):
            L = int(lengths[r])
            ref = cand_cat[r, torch.topk(score_cat[r, :L], min(2048, L)).indices]
            got = split[r]
            got = got[got >= 0]
            self.assertTrue(torch.equal(got.sort().values, ref.sort().values), msg=f"row {r}")
            self.assertTrue(bool((got[got >= n_complete] < ctx[r]).all()))

    def test_topk_equals_dense_restricted_to_candidates(self):
        import deep_gemm

        torch.manual_seed(11)
        dev = torch.device('cuda')
        cfg = PartitionConfig(mode='adaptive_decode', sparse_prefill=True, sparse_prefill_rows=16).validate()
        n_complete, seq_len, n_q, H = 16384, 16384 + 300, 40, 64  # budget 8192 < prefix: real selection
        k = torch.randn(seq_len, 128, device=dev).to(torch.float8_e4m3fn)
        s = torch.rand(seq_len, device=dev) + 0.5
        q = torch.randn(n_q, H, 128, device=dev).to(torch.float8_e4m3fn)
        w = torch.rand(n_q, H, device=dev)
        pos1 = torch.arange(n_complete + 1, n_complete + 1 + n_q, device=dev, dtype=torch.int32)
        pos1[-1] = seq_len
        # 64-token leaves as the partition; their summaries are the (requantised) means
        m = n_complete // 64
        leaf_start = torch.arange(0, n_complete, 64, device=dev, dtype=torch.int32)
        leaf_len = torch.full((m,), 64, device=dev, dtype=torch.int32)
        means = (k[:n_complete].float() * s[:n_complete, None]).view(m, 64, 128).mean(1)
        keys = means.to(torch.float8_e4m3fn)
        scales = torch.ones(m, device=dev)
        # index-K cache pages: [pages, 64*132] = 64 fp8 rows then 64 fp32 scales
        pages = (seq_len + 63) // 64
        buf = torch.zeros((pages, 64 * 132), dtype=torch.uint8, device=dev)
        kp = torch.zeros((pages * 64, 128), dtype=torch.uint8, device=dev); kp[:seq_len] = k.view(torch.uint8)
        sp = torch.zeros(pages * 64, device=dev); sp[:seq_len] = s
        buf[:, : 64 * 128] = kp.view(pages, -1)
        buf[:, 64 * 128:] = sp.view(pages, 64).view(torch.uint8).view(pages, 256)
        table = torch.arange(pages, device=dev, dtype=torch.int32)[None]

        out = ps.sparse_topk_core(q, w, pos1, buf, table, keys, scales, leaf_start, leaf_len,
                                  torch.tensor(m, device=dev), n_complete, seq_len, cfg)
        self.assertEqual(out.shape, (n_q, 2048))
        out = out.long()
        self.assertTrue(bool((out >= 0).all()) and bool((out < pos1[:, None].long()).all()))
        srt, _ = out.sort(dim=1)
        self.assertFalse(bool((srt[:, 1:] == srt[:, :-1]).any()))

        dense = deep_gemm.fp8_mqa_logits(q, (k, s), w, torch.zeros_like(pos1), pos1, clean_logits=False)[:, :seq_len]
        dense = dense.masked_fill(torch.arange(seq_len, device=dev)[None] >= pos1[:, None], float('-inf'))
        coarse = ps.coarse_leaf_scores(q, w, keys, scales, torch.tensor(m, device=dev), leaf_len,
                                       leaf_start, cfg.sink_tokens)
        # the sink leaf (start 0 < 64) is forced first for every row
        self.assertTrue(bool(torch.isinf(coarse[:, 0]).all()) and bool((coarse[:, 0] > 0).all()))
        self.assertFalse(bool(torch.isposinf(coarse[:, 1:]).any()))
        order, inc = ps.rank_leaves(coarse, leaf_start, leaf_len)
        self.assertTrue(bool((order[:, 0] == 0).all()))
        cand = ps.expand_candidates(order, inc, leaf_start, leaf_len, slice(0, n_q), cfg.candidate_tokens, n_complete, seq_len - n_complete).long()
        allowed = torch.zeros_like(dense, dtype=torch.bool)
        allowed.scatter_(1, cand, cand < pos1[:, None].long())
        ref = torch.topk(dense.masked_fill(~allowed, float('-inf')), 2048, dim=1).indices
        ref_mask = torch.zeros_like(dense, dtype=torch.bool).scatter_(1, ref, True)
        agreement = ref_mask.gather(1, out).float().mean().item()
        self.assertGreater(agreement, 0.995)  # fp8 GEMM tie-breaking noise only

        # all-sparse scoring of the local window and the flat-key shortcut give the same sets
        out_mask = torch.zeros_like(dense, dtype=torch.bool).scatter_(1, out, True)
        for kwargs in (dict(dense_local=False), dict(k_flat=(k, s)), dict(k_flat=(k, s), dense_local=False)):
            alt = ps.sparse_topk_core(q, w, pos1, buf, table, keys, scales, leaf_start, leaf_len,
                                      torch.tensor(m, device=dev), n_complete, seq_len, cfg, **kwargs).long()
            self.assertGreater(out_mask.gather(1, alt).float().mean().item(), 0.995, msg=str(kwargs))

        # Row chunking only bounds temporary workspaces; a single full-row
        # launch must select the same token sets.
        full_rows_cfg = dataclasses.replace(cfg, sparse_prefill_rows=n_q)
        full_rows = ps.sparse_topk_core(
            q, w, pos1, buf, table, keys, scales, leaf_start, leaf_len,
            torch.tensor(m, device=dev), n_complete, seq_len, full_rows_cfg,
            k_flat=(k, s),
        ).long()
        self.assertGreater(out_mask.gather(1, full_rows).float().mean().item(), 0.995)

        # HISA-shaped block-local mode shares the 8192-token budget between
        # adaptive prefix leaves and 64-token local pseudo-leaves. It is an
        # approximation, but its output contract remains causal and unique.
        block = ps.sparse_topk_core(
            q, w, pos1, buf, table, keys, scales, leaf_start, leaf_len,
            torch.tensor(m, device=dev), n_complete, seq_len, cfg,
            k_flat=(k, s), block_local=True,
        ).long()
        self.assertTrue(bool((block >= 0).all()) and bool((block < pos1[:, None].long()).all()))
        block_sorted, _ = block.sort(dim=1)
        self.assertFalse(bool((block_sorted[:, 1:] == block_sorted[:, :-1]).any()))

        # fine scorer: flat persistent K=1 kernel == paged K=1 kernel (same fp8 GEMM, exact match),
        # including -1 padding and tokens past each row's causal end
        cand = cand.to(torch.int32)
        cand[:, -7:] = -1
        cand[3, :5] = pos1[3] + torch.arange(5, device=dev, dtype=torch.int32)  # >= causal end -> -inf
        raw_pages = buf.view(-1, 64, 1, 132)
        paged = ps._fine_scores(q, w, cand, pos1, None, raw_pages, table)
        flat = ps._fine_scores(q, w, cand, pos1, ps._flat_keys((k, s), seq_len), raw_pages, table)
        self.assertEqual(flat.shape, paged.shape)
        self.assertTrue(torch.equal(torch.isneginf(flat), torch.isneginf(paged)))
        self.assertTrue(bool(torch.isneginf(flat[:, -7:]).all()) and bool(torch.isneginf(flat[3, :5]).all()))
        finite = torch.isfinite(paged)
        self.assertTrue(torch.equal(flat[finite], paged[finite]))
        self.assertIsNone(ps._flat_keys((k[:100], s[:100]), seq_len))  # too short -> paged fallback


if __name__ == '__main__':
    unittest.main()

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


class TestAdmission(unittest.TestCase):
    def _batch(self, seq_len=20000, extend=3616, bs=1):
        return SimpleNamespace(
            batch_size=bs, seq_lens_cpu=torch.tensor([seq_len]), extend_seq_lens_cpu=torch.tensor([extend]),
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
            self.assertEqual(got[2:], (20000 - 3616, 20000))
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
        for kwargs in (dict(dense_local=False), dict(k_flat=(k, s))):
            alt = ps.sparse_topk_core(q, w, pos1, buf, table, keys, scales, leaf_start, leaf_len,
                                      torch.tensor(m, device=dev), n_complete, seq_len, cfg, **kwargs).long()
            self.assertGreater(out_mask.gather(1, alt).float().mean().item(), 0.995, msg=str(kwargs))


if __name__ == '__main__':
    unittest.main()

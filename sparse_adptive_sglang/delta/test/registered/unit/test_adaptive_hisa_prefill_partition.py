"""Adaptive-HISA Phase B tests: config, CPU reference, GPU backend, summaries, wiring.

Everything here runs on CPU tensors (the ``gpu`` backend is plain torch and
also runs on the CPU device); CUDA cases add Triton kernels and the
cpu_reference overlap worker. No server is launched.
"""

import os
import sys
import unittest
from dataclasses import replace

import numpy as np
import torch

from sglang.srt.layers.attention.nsa.adaptive_hisa.config import (
    ATOM,
    ROOT,
    PartitionConfigError,
    config_from_env,
    get_config,
    reset_config_cache,
)
from sglang.srt.layers.attention.nsa.adaptive_hisa.legacy.reference import (
    gather_compact,
    pack_index_buffer,
)
from sglang.srt.layers.attention.nsa.adaptive_hisa.partition_gpu import (
    build_partition_gpu,
    build_score_tree,
)
from sglang.srt.layers.attention.nsa.adaptive_hisa.partition_reference import (
    _leaf_totals,
    heap_reference_merge,
    lambda_dp_repair,
    sync_nonoverlap_merge,
)
from sglang.srt.layers.attention.nsa.adaptive_hisa.prefill_partition import (
    build_prefill_partition_from_keys,
    greedy_adjacent_merge,
    key_values,
    n_complete_tokens,
    partition_cpu_phase,
    partition_keys_gpu_phase,
    score_energies,
    validate_leaves,
)
from sglang.srt.layers.attention.nsa.adaptive_hisa.prefill_runtime import (
    STATE,
    finish_prefill_partitions,
    get_gpu_partition,
    get_prefill_partition,
    get_summary_entry,
    partition_request_flags,
    prefill_skip_reason,
    prepare_prefill_partition,
    release_request,
    reset_prefill_state,
    schedule_prefill_partition,
)
from sglang.srt.layers.attention.nsa.adaptive_hisa.summary_kernels import (
    _torch_act_quant,
    leaf_key_means,
    requant_summaries,
)
from sglang.srt.layers.attention.nsa.adaptive_hisa.summary_pool import (
    ROWS_PER_PAGE,
    SummaryPool,
    SummaryPoolExhausted,
    build_summaries,
    get_summary_pool,
    pages_for,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=60, suite="stage-a-test-cpu")

_SIBLING = "/DATA/disk0/qyl/code/sglang_hisa"
_ENV = "SGLANG_NSA_ADAPTIVE_HISA_"
KEY_DIM = 128
_ENV_KEYS = (
    "MODE",
    "PREFILL_PARTITION",
    "MERGE_POLICY",
    "PARTITION_MERGE",
    "SPLIT_BACKEND",
    "PARTITION_OVERLAP",
    "BUILD_SUMMARIES",
    "MAX_MERGE_LEN",
    "MERGE_ROUNDS",
    "PARTITION_VALIDATE",
    "ATOM",
    "LAYERS",
    "REPAIR_TIE_PASSES",
    "GRAPH_BUILD",
    "GPU_STREAM",
    "CALIB_QUERIES",
    "PARTITION_REUSE_LOGITS",
    "PARTITION_METRIC", "MERGE_TARGET_DIVISOR", "MERGE_TARGET_ROUNDS",
)


def _set_env(*bases: dict, **values) -> None:
    """Reset every Adaptive-HISA variable, then apply ``bases`` and overrides."""
    for key in _ENV_KEYS:
        os.environ.pop(_ENV + key, None)
    merged: dict = {}
    for base in bases:
        merged.update(base)
    merged.update(values)
    for key, value in merged.items():
        os.environ[_ENV + key] = str(value)
    reset_config_cache()


GPU = dict(MODE="build_only", SPLIT_BACKEND="gpu", MERGE_POLICY="off", PARTITION_VALIDATE="1")
GPU_MERGE = dict(GPU, MERGE_POLICY="sync_nonoverlap")
# The numba path that was A/B'ed before (heap merge with the 256 cap).
V7_REUSE = dict(
    MODE="build_only",
    SPLIT_BACKEND="cpu_reference",
    MERGE_POLICY="heap_reference",
    MAX_MERGE_LEN="256",
    PARTITION_VALIDATE="1",
)


def _direct_energy(scores: torch.Tensor, start: int, length: int) -> float:
    chunk = scores[:, start : start + length].to(torch.float64)
    mean = chunk.mean(dim=-1, keepdim=True)
    return float(((chunk - mean) ** 2).sum())


def _keys(n: int, dim: int = 128, seed: int = 0):
    gen = torch.Generator().manual_seed(seed)
    raw = torch.randint(0, 255, (n, dim), generator=gen, dtype=torch.int32)
    raw = raw.clamp(0, 126).to(torch.uint8)  # avoid the fp8 NaN encoding
    scale = torch.linspace(0.5, 1.5, n)
    return raw.view(torch.float8_e4m3fn), scale


def _queries(rows: int, heads: int = 64, dim: int = 128, seed: int = 1):
    gen = torch.Generator().manual_seed(seed)
    q = torch.randn(rows, heads, dim, generator=gen).to(torch.float8_e4m3fn)
    w = torch.randn(rows, heads, generator=gen)
    return q, w


def _scores(kind: str, n_tokens: int, seed: int = 0) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    if kind == "random":
        return torch.randn(KEY_DIM, n_tokens, generator=gen).abs()
    if kind == "zero":
        return torch.zeros(KEY_DIM, n_tokens)
    if kind == "piecewise":
        base = torch.randint(0, 5, (KEY_DIM, n_tokens // 32), generator=gen).float()
        return base.repeat_interleave(32, dim=1)
    if kind == "sparse":
        s = torch.zeros(KEY_DIM, n_tokens)
        idx = torch.randint(0, n_tokens, (KEY_DIM, max(1, n_tokens // 200)), generator=gen)
        s.scatter_(1, idx, 50.0)
        return s
    raise ValueError(kind)


def _ref_means(fp8: torch.Tensor, scale: torch.Tensor, start, length) -> torch.Tensor:
    keys = fp8.float() * scale[:, None].float()
    out = torch.zeros(start.numel(), fp8.shape[1], device=fp8.device)
    for i in range(start.numel()):
        s, ln = int(start[i]), int(length[i])
        if ln:
            out[i] = keys[s : s + ln].mean(0)
    return out


class _Req:
    def __init__(self, output_ids, extend_input_len, cached_tokens):
        self.output_ids = output_ids
        self.extend_input_len = extend_input_len
        self.cached_tokens = cached_tokens


class _Mode:
    def is_extend_without_speculative(self):
        return True

    def is_split_prefill(self):
        return False

    def is_dllm_extend(self):
        return False

    def is_extend(self):
        return True


class _FakeKVPool:
    """Only what ``get_summary_pool`` reads: per-layer raw index-K page buffers."""

    def __init__(self, layers: int = 8, pages: int = 64, device="cpu"):
        self.index_k_with_scale_buffer = [
            torch.zeros(pages, ROWS_PER_PAGE * 132, dtype=torch.uint8, device=device)
            for _ in range(layers)
        ]


class _Batch:
    def __init__(self, *, final, new_tokens, n_tokens, req_idx=7, device="cpu", layers=8):
        self.batch_size = 1
        self.forward_mode = _Mode()
        self.spec_algorithm = None
        self.attn_cp_metadata = None
        self.req_pool_indices = torch.tensor([req_idx])
        self.seq_lens_cpu = torch.tensor([n_tokens])
        self.prefill_final_cpu = [final]
        self.prefill_new_tokens_cpu = [new_tokens]
        self.input_ids = torch.zeros(n_tokens, dtype=torch.int32)
        self.token_to_kv_pool = _FakeKVPool(layers=layers, device=device)


class _Base(CustomTestCase):
    ENV = GPU

    def setUp(self):
        super().setUp()
        _set_env(self.ENV)
        reset_prefill_state()

    def tearDown(self):
        reset_prefill_state()
        _set_env()
        super().tearDown()


# --------------------------------------------------------------------------- #
# B0: configuration contract
# --------------------------------------------------------------------------- #


class TestPartitionConfig(_Base):
    def test_defaults_and_mode(self):
        _set_env()
        self.assertFalse(get_config().enabled)
        _set_env(MODE="build_only")
        cfg = get_config()
        self.assertTrue(cfg.enabled)
        self.assertEqual((cfg.split_backend, cfg.merge_policy, cfg.merge_rounds), ("gpu", "sync_nonoverlap", 2))
        self.assertEqual((cfg.atom, cfg.root, cfg.summary_compression), (1, 256, 8))
        self.assertEqual((cfg.partition_metric, cfg.method_name, cfg.energy_rows), ("key_sse", "P-key", 128))
        self.assertEqual(cfg.max_merge_len, 0)
        self.assertTrue(cfg.build_summaries and not cfg.cpu_overlap)
        self.assertTrue(cfg.graph_build)
        self.assertEqual(cfg.gpu_stream, "side")
        self.assertEqual(cfg.levels, 9)
        _set_env(MODE="build_only", GPU_STREAM="main", GRAPH_BUILD="0")
        cfg = get_config()
        self.assertEqual((cfg.gpu_stream, cfg.graph_build), ("main", False))
        with self.assertRaises(PartitionConfigError):
            _set_env(MODE="build_only", GPU_STREAM="left")
            config_from_env()
        for removed in ("CALIB_QUERIES", "PARTITION_REUSE_LOGITS"):
            with self.assertRaises(PartitionConfigError):
                _set_env(MODE="build_only", **{removed: "1"})
                config_from_env()

    def test_legacy_flags_and_migration_errors(self):
        _set_env(PREFILL_PARTITION="1")
        self.assertEqual(get_config().mode, "build_only")
        _set_env(PREFILL_PARTITION="1", PARTITION_MERGE="0")
        self.assertEqual(get_config().merge_policy, "off")
        with self.assertRaises(PartitionConfigError):
            _set_env(PREFILL_PARTITION="1", PARTITION_MERGE="1")
            config_from_env()
        with self.assertRaises(PartitionConfigError):
            _set_env(MODE="off", PREFILL_PARTITION="1")
            config_from_env()
        _set_env(MODE="adaptive_decode")
        self.assertEqual(config_from_env().mode, "adaptive_decode")
        with self.assertRaises(PartitionConfigError):
            _set_env(MODE="build_only", MERGE_POLICY="sync_nonoverlap", PARTITION_MERGE="0")
            config_from_env()

    def test_backend_rules(self):
        _set_env(V7_REUSE)
        cfg = get_config()
        self.assertFalse(cfg.build_summaries)  # summaries are gpu-only
        self.assertEqual(cfg.gpu_stream, "main")  # stream choice is gpu-only too
        self.assertEqual(cfg.max_merge_len, 256)
        with self.assertRaises(PartitionConfigError):
            _set_env(V7_REUSE, GPU_STREAM="side")
            config_from_env()
        _set_env(V7_REUSE, PARTITION_OVERLAP="1")
        self.assertTrue(get_config().cpu_overlap)
        with self.assertRaises(PartitionConfigError):
            _set_env(MODE="build_only", SPLIT_BACKEND="gpu", PARTITION_OVERLAP="1")
            config_from_env()
        with self.assertRaises(PartitionConfigError):
            _set_env(MODE="build_only", SPLIT_BACKEND="cpu_reference", BUILD_SUMMARIES="1")
            config_from_env()
        with self.assertRaises(PartitionConfigError):
            _set_env(MODE="build_only", ATOM="3")
            config_from_env()

    def test_refuses_conflicting_experiments(self):
        os.environ["SGLANG_NSA_PER_HEAD_INDEX"] = "1"
        try:
            with self.assertRaises(PartitionConfigError):
                _set_env(MODE="build_only")
                config_from_env()
            _set_env()
            self.assertFalse(get_config().enabled)  # off never conflicts
        finally:
            os.environ.pop("SGLANG_NSA_PER_HEAD_INDEX", None)
            reset_config_cache()


# --------------------------------------------------------------------------- #
# B0: CPU reference (tree, λ-DP + repair, merge semantics)
# --------------------------------------------------------------------------- #


@unittest.skip("legacy query-score tests removed with P-qfull")
class TestCpuReference(_Base):
    def test_score_moment_merge_matches_direct_variance(self):
        scores = torch.tensor(
            [[0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 3.0, 3.0, 3.0, 0.0, 0.0, 9.0]],
            dtype=torch.float32,
        )
        energies = score_energies(scores, atom=4, root=8)
        self.assertEqual([tuple(e.shape) for e in energies], [(2,), (1,)])
        for level, nodes in enumerate(energies):
            length = 4 * (1 << level)
            for idx, got in enumerate(nodes.tolist()):
                self.assertAlmostEqual(got, _direct_energy(scores, idx * length, length), places=6)

    def test_tree_levels_cover_256_down_to_atom(self):
        energies = score_energies(torch.randn(3, ROOT))
        lengths = [ATOM * (1 << level) for level in range(len(energies))]
        self.assertEqual(lengths, [1, 2, 4, 8, 16, 32, 64, 128, 256])
        self.assertEqual([int(e.numel()) for e in energies], [ROOT // ln for ln in lengths])

    def test_gpu_tree_matches_reference_energies(self):
        scores = _scores("random", 1024)
        flat, layout, prefix = build_score_tree(scores, atom=ATOM, root=ROOT)
        ref = torch.cat([e.to(torch.float64) for e in score_energies(scores)])
        self.assertEqual(flat.numel(), ref.numel())
        self.assertTrue(torch.allclose(flat, ref, rtol=1e-9, atol=1e-6))
        self.assertTrue(torch.allclose(prefix[:, -1], scores.double().sum(-1)))

    def test_global_budget_coverage_and_tail(self):
        _set_env(V7_REUSE, MERGE_POLICY="off")
        q, w = _queries(CALIB_QUERIES)
        k, scale = _keys(300)
        part = build_prefill_partition(k, scale, q, w, 300)
        self.assertEqual((part.n_complete, part.tail_len), (256, 44))
        self.assertEqual(int(part.leaf_start.numel()), 256 // 8)
        validate_leaves(part.leaf_start, part.leaf_len, part.n_complete)
        self.assertEqual(sum(v["leaves"] for v in part.histogram().values()), 32)

    def test_precomputed_scores_match_full_builder(self):
        _set_env(V7_REUSE)
        q, w = _queries(CALIB_QUERIES, seed=12)
        k, scale = _keys(300)
        scores = calibration_scores(q, w, k, scale, 256)
        reference = build_prefill_partition(k, scale, q, w, 300, merge=True)
        got = partition_cpu_phase(partition_scores_gpu_phase(scores, 300), merge=True)
        self.assertTrue(torch.equal(got.leaf_start, reference.leaf_start))
        self.assertTrue(torch.equal(got.leaf_len, reference.leaf_len))
        self.assertEqual(got.method, "P-qfull-heap_reference")
        self.assertLessEqual(int(got.leaf_len.max()), 256)

    def test_empty_and_short_prefix_have_no_leaves(self):
        _set_env(V7_REUSE)
        q, w = _queries(CALIB_QUERIES)
        k, scale = _keys(100)
        part = build_prefill_partition(k, scale, q, w, 100)
        self.assertEqual((part.n_complete, part.tail_len, part.leaf_start.numel()), (0, 100, 0))
        self.assertEqual(n_complete_tokens(255), 0)
        self.assertEqual(n_complete_tokens(256), 256)

    def test_tie_keeps_parent_and_repairs_smaller_start_first(self):
        energies = score_energies(torch.zeros(2, 512))
        start, length, meta = lambda_dp_repair(energies, 64)
        validate_leaves(start, length, 512)
        self.assertEqual(meta["dp_leaves"], 2)
        self.assertGreater(meta["repairs"], 0)
        self.assertEqual((int((start < 256).sum()), int((start >= 256).sum())), (63, 1))
        self.assertEqual(int(length[start >= 256][0]), 256)
        again_s, again_l, _ = lambda_dp_repair(energies, 64)
        self.assertTrue(torch.equal(start, again_s) and torch.equal(length, again_l))

    def test_global_budget_moves_to_the_high_variance_root(self):
        scores = torch.zeros(2, 512)
        for atom in range(ROOT // ATOM):
            scores[:, ROOT + atom * ATOM : ROOT + (atom + 1) * ATOM] = float(atom + 1)
        start, length, _ = lambda_dp_repair(score_energies(scores), 64)
        flat, hot = start < ROOT, start >= ROOT
        self.assertEqual((int(flat.sum()), int(length[flat][0]), int(hot.sum())), (1, ROOT, 63))
        self.assertTrue(bool((length[hot] <= 8).all()))
        self.assertEqual(int(start[hot][0] + 0), ROOT)
        self.assertEqual(int(start[hot][-1] + length[hot][-1]), 2 * ROOT)

    def test_page_permutation_does_not_change_logical_partition(self):
        _set_env(V7_REUSE, MERGE_POLICY="off")
        n = ROOT
        k, scale = _keys(n)
        q, w = _queries(CALIB_QUERIES, seed=2)
        logical = build_prefill_partition(k, scale, q, w, n)
        buf = pack_index_buffer(k, scale.float())
        order = torch.tensor([2, 0, 3, 1])
        page_table = torch.empty_like(order)
        page_table[order] = torch.arange(order.numel())
        got_k, got_s = gather_compact(buf[order], page_table, torch.arange(n))
        self.assertTrue(torch.equal(got_k, k.view(torch.uint8)))
        permuted = build_prefill_partition(got_k.view(torch.float8_e4m3fn), got_s, q, w, n)
        self.assertTrue(torch.equal(logical.leaf_start, permuted.leaf_start))
        self.assertTrue(torch.equal(logical.leaf_len, permuted.leaf_len))

    def test_heap_merge_full_collapse_and_strict_threshold(self):
        start, length = torch.arange(8), torch.ones(8, dtype=torch.long)
        merged_start, merged_len, meta = greedy_adjacent_merge(start, length, torch.zeros(2, 8), threshold=1.0)
        self.assertEqual((merged_start.tolist(), merged_len.tolist(), meta["merge_count"]), ([0], [8], 7))
        validate_leaves(merged_start, merged_len, 8, allow_arbitrary=True)
        # SSE increase for [0] + [1] is exactly 0.5. Equality does not merge.
        s, ln, m = greedy_adjacent_merge(
            torch.tensor([0, 1]), torch.ones(2, dtype=torch.long), torch.tensor([[0.0, 1.0]]), threshold=0.5
        )
        self.assertEqual((s.tolist(), ln.tolist(), m["merge_count"]), ([0, 1], [1, 1], 0))

    def test_heap_merge_respects_cap(self):
        start = torch.arange(0, 1024, 128)
        length = torch.full((8,), 128, dtype=torch.long)
        merged_start, merged_len, meta = greedy_adjacent_merge(
            start, length, torch.zeros(2, 1024), threshold=1.0, max_leaf_len=256
        )
        self.assertEqual(merged_len.tolist(), [256] * 4)
        self.assertEqual(meta["merge_count"], 4)
        uncapped_s, uncapped_l, _ = greedy_adjacent_merge(
            start, length, torch.zeros(2, 1024), threshold=1.0, max_leaf_len=0
        )
        self.assertEqual((uncapped_s.tolist(), uncapped_l.tolist()), ([0], [1024]))

    def test_sync_nonoverlap_differs_from_heap_on_review_counterexample(self):
        """GPT review §4.5 counterexample (merge_semantics_checks.json)."""
        leaves = [(0, 128, 1.0), (128, 64, 4.0), (192, 64, 0.0), (256, 64, 3.0), (320, 64, 100.0), (384, 128, 200.0)]
        starts = np.array([s for s, _, _ in leaves], dtype=np.int64)
        lengths = np.array([c for _, c, _ in leaves], dtype=np.int64)
        totals = np.array([[c * m] for _, c, m in leaves], dtype=np.float64)
        heap_s, heap_l, _ = heap_reference_merge(starts, lengths, totals, 500.0, max_merge_len=256)
        self.assertEqual(list(zip(heap_s.tolist(), heap_l.tolist())), [(0, 128), (128, 192), (320, 64), (384, 128)])
        one_s, one_l, info = sync_nonoverlap_merge(starts, lengths, totals, 500.0, rounds=1, max_merge_len=256)
        self.assertEqual(list(zip(one_s.tolist(), one_l.tolist())), [(0, 192), (192, 128), (320, 64), (384, 128)])
        self.assertEqual(info["round_merge_counts"], [2])
        two_s, two_l, info = sync_nonoverlap_merge(starts, lengths, totals, 500.0, rounds=2, max_merge_len=0)
        self.assertEqual(list(zip(two_s.tolist(), two_l.tolist())), [(0, 320), (320, 64), (384, 128)])
        self.assertEqual(info["round_merge_counts"], [2, 1])
        # Rounds stop early when a round selects nothing.
        _, _, info = sync_nonoverlap_merge(starts, lengths, totals, 500.0, rounds=5, max_merge_len=0)
        self.assertEqual(info["merge_rounds_run"], 2)

    def test_sibling_offline_parity(self):
        if not os.path.isdir(_SIBLING):
            self.skipTest("sibling offline package is not present")
        os.environ["ADAPTIVE_HISA_ATOM"] = str(ATOM)
        if _SIBLING not in sys.path:
            sys.path.insert(0, _SIBLING)
        try:
            from experiments.adaptive_hisa.partition import (
                greedy_adjacent_merge as sibling_merge,
                lambda_dp_repair as sibling_dp,
            )
            from experiments.adaptive_hisa.tree_stats import build_tree, score_tree_energy
        except Exception as exc:
            self.skipTest(f"sibling import failed: {exc}")
        gen = torch.Generator().manual_seed(11)
        for n_tokens, budget in ((1024, 128), (2048, 256), (4096, 512)):
            random_scores = torch.randn(CALIB_QUERIES, n_tokens, generator=gen).cumsum(-1)
            ours = score_energies(random_scores)
            tree = build_tree(torch.zeros(n_tokens, 1), atom=ATOM, root=ROOT)
            theirs = score_tree_energy(tree, random_scores)
            start, length, meta = lambda_dp_repair(ours, budget)
            ref = sibling_dp(tree, budget, energies=theirs, method="P-qfull")
            self.assertTrue(torch.equal(start, ref.leaf_start), (n_tokens, budget))
            self.assertTrue(torch.equal(length, ref.leaf_len), (n_tokens, budget))
            self.assertAlmostEqual(meta["lambda"], float(ref.meta["lambda"]), delta=1e-5 * max(1.0, meta["lambda"]))
            merged_start, merged_len, _ = greedy_adjacent_merge(
                start, length, random_scores, float(ref.meta["lambda"]), max_leaf_len=0
            )
            merged_ref = sibling_merge(ref, random_scores)
            self.assertTrue(torch.equal(merged_start, merged_ref.leaf_start))
            self.assertTrue(torch.equal(merged_len, merged_ref.leaf_len))


# --------------------------------------------------------------------------- #
# B1/B2: GPU backend parity with the CPU reference (runs on the CPU device too)
# --------------------------------------------------------------------------- #


class TestGpuBackend(_Base):
    DEVICE = "cpu"

    def _check_split(self, scores, n_tokens, cfg):
        part = build_partition_gpu(scores.to(self.DEVICE), n_tokens, cfg)
        host = part.to_host()
        done = host.n_complete
        budget = done // cfg.summary_compression
        ref_s, ref_l, meta = lambda_dp_repair(score_energies(scores[:, :done], atom=cfg.atom, root=cfg.root), budget, atom=cfg.atom)
        self.assertEqual(part.capacity, budget)
        self.assertTrue(bool(part.status_ok))
        self.assertEqual(int(part.num_leaves), budget)
        self.assertTrue(torch.equal(host.leaf_start.long(), ref_s))
        self.assertTrue(torch.equal(host.leaf_len.long(), ref_l))
        self.assertEqual(host.meta["dp_leaves"], meta["dp_leaves"])
        self.assertEqual(host.meta["repairs"], meta["repairs"])
        self.assertAlmostEqual(host.meta["lambda"], meta["lambda"], delta=1e-9 * max(1.0, abs(meta["lambda"])))
        # padding contract of the fixed-capacity buffers
        self.assertEqual(part.leaf_start.numel(), budget)
        return part, host, ref_s, ref_l, meta

    # Parity with the CPU reference checks λ to 1e-9, which needs the search
    # run to ``lambda_rel_tol`` (10 rounds); the production default caps the
    # rounds (``lambda_max_rounds``) and is a speed knob, not the algorithm.
    def test_split_parity_all_distributions(self):
        cfg = replace(get_config(), lambda_max_rounds=0).validate()
        for kind in ("random", "zero", "piecewise", "sparse"):
            for n_tokens in (256, 1024 + 37, 4096):
                self._check_split(_scores(kind, n_tokens, seed=3), n_tokens, cfg)

    def test_split_parity_atom4_and_compression(self):
        cfg = replace(get_config(), atom=4, summary_compression=16, lambda_max_rounds=0).validate()
        _, host, *_ = self._check_split(_scores("random", 2048), 2048, cfg)
        self.assertTrue(bool((host.leaf_len % 4 == 0).all()))
        self.assertEqual(host.leaf_start.numel(), 2048 // 16)

    def test_short_tail_has_no_leaves(self):
        part = build_partition_gpu(_scores("random", 200).to(self.DEVICE), 200, get_config())
        self.assertEqual((part.n_complete, part.capacity, part.leaf_start.numel()), (0, 0, 0))
        self.assertEqual(part.to_host().leaf_start.numel(), 0)

    def test_merge_parity_two_rounds(self):
        _set_env(GPU_MERGE)
        cfg = get_config()
        for kind, n_tokens, cap in (("random", 4096, 0), ("piecewise", 2048, 0), ("random", 4096, 256), ("zero", 1024, 0)):
            cfg_i = replace(cfg, max_merge_len=cap)
            scores = _scores(kind, n_tokens, seed=5)
            part = build_partition_gpu(scores.to(self.DEVICE), n_tokens, cfg_i)
            host = part.to_host()
            done = host.n_complete
            budget = done // cfg.summary_compression
            ref_s, ref_l, meta = lambda_dp_repair(score_energies(scores[:, :done]), budget)
            totals = _leaf_totals(scores[:, :done].numpy().astype(np.float32), ref_s.numpy(), ref_l.numpy())
            m_s, m_l, info = sync_nonoverlap_merge(
                ref_s.numpy(), ref_l.numpy(), totals, meta["lambda"] * cfg.merge_alpha, rounds=2, max_merge_len=cap
            )
            self.assertTrue(torch.equal(host.leaf_start.long(), torch.from_numpy(m_s)), (kind, cap))
            self.assertTrue(torch.equal(host.leaf_len.long(), torch.from_numpy(m_l)), (kind, cap))
            self.assertEqual(host.meta["round_merge_counts"], info["round_merge_counts"])
            self.assertEqual(int(part.num_leaves), m_s.shape[0])
            self.assertEqual(part.capacity, budget)  # capacity stays M0
            validate_leaves(host.leaf_start, host.leaf_len, done, allow_arbitrary=True, max_len=cap)
            # rows past num_leaves are padding
            self.assertTrue(bool((part.leaf_len[int(part.num_leaves) :] == 0).all()))
            if cap:
                self.assertLessEqual(int(host.leaf_len.max()), cap)

    def test_heap_reference_needs_cpu_backend(self):
        cfg = replace(get_config(), merge_policy="heap_reference")
        with self.assertRaises(NotImplementedError):
            build_partition_gpu(_scores("random", 512).to(self.DEVICE), 512, cfg)



# --------------------------------------------------------------------------- #
# B2: FP8 summaries and the paged pool
# --------------------------------------------------------------------------- #


class TestSummaries(_Base):
    ENV = GPU_MERGE
    DEVICE = "cpu"

    def test_leaf_means_padding_and_paged_roundtrip(self):
        cfg = get_config()
        n_tokens = 4096 + 100
        part = build_partition_gpu(_scores("random", n_tokens).to(self.DEVICE), n_tokens, cfg)
        fp8, scale = _keys(n_tokens)
        fp8, scale = fp8.to(self.DEVICE), scale.to(self.DEVICE)
        pool = SummaryPool(num_layers=2, pages_per_layer=pages_for(n_tokens // 64 + 1, cfg.summary_compression), device=self.DEVICE)
        entry = pool.allocate(1, 5, 0, part)
        build_summaries(pool, entry, fp8, scale, scale_fmt=None)
        num = int(part.num_leaves)
        self.assertEqual(entry.capacity, part.capacity)
        self.assertEqual(len(entry.pages), (part.capacity + ROWS_PER_PAGE - 1) // ROWS_PER_PAGE)
        means = leaf_key_means(fp8, scale, part.leaf_start, part.leaf_len)
        ref = _ref_means(fp8, scale, part.leaf_start, part.leaf_len)
        self.assertLess(float((means[:num] - ref[:num]).abs().max()), 1e-4)
        self.assertTrue(bool((means[num:] == 0).all()))
        k_out, s_out = pool.read(entry)
        ref_fp8, ref_scale = _torch_act_quant(ref.to(torch.bfloat16), False)
        self.assertTrue(torch.equal(k_out[:num].view(torch.uint8), ref_fp8[:num].view(torch.uint8)))
        self.assertTrue(torch.allclose(s_out[:num], ref_scale[:num, 0], rtol=1e-3, atol=0))
        # the raw keys are read-only inputs
        self.assertTrue(torch.equal(fp8.view(torch.uint8), _keys(n_tokens)[0].view(torch.uint8).to(self.DEVICE)))

    def test_allocation_reuse_release_and_exhaustion(self):
        cfg = get_config()
        part = build_partition_gpu(_scores("random", 2048).to(self.DEVICE), 2048, cfg)
        pool = SummaryPool(num_layers=2, pages_per_layer=8, device=self.DEVICE)
        e1 = pool.allocate(0, 3, 0, part)
        used = pool.stats()["pages_used"]
        self.assertEqual(used, 4)  # M0 = 256 rows = 4 pages
        e2 = pool.allocate(0, 3, 1, part)  # same slot: old pages freed first
        self.assertEqual(pool.stats()["pages_used"], used)
        self.assertEqual(sorted(e1.pages), sorted(e2.pages))
        pool.allocate(1, 3, 1, part)
        self.assertEqual(pool.stats()["entries"], 2)
        self.assertEqual(pool.release(3), 2)
        self.assertEqual(pool.stats()["pages_used"], 0)
        pool.allocate(0, 4, 0, part)
        pool.allocate(0, 5, 0, part)
        with self.assertRaises(SummaryPoolExhausted):
            pool.allocate(0, 6, 0, part)
        self.assertEqual(pool.get(0, 6), None)

    def test_short_tail_allocates_nothing(self):
        part = build_partition_gpu(_scores("random", 100).to(self.DEVICE), 100, get_config())
        pool = SummaryPool(num_layers=1, pages_per_layer=2, device=self.DEVICE)
        entry = pool.allocate(0, 1, 0, part)
        build_summaries(pool, entry, *_keys(100), scale_fmt=None)
        self.assertEqual((entry.capacity, len(entry.pages), pool.stats()["pages_used"]), (0, 0, 0))

    def test_pool_sizing_from_kv_pool(self):
        kv = _FakeKVPool(layers=3, pages=1000)
        pool = get_summary_pool(kv, 8)
        self.assertEqual((pool.num_layers, pool.pages_per_layer), (3, 1000 // 8 + 16))
        self.assertIs(get_summary_pool(kv, 8), pool)


# --------------------------------------------------------------------------- #
# B3: build_only wiring through prefill_runtime
# --------------------------------------------------------------------------- #


@unittest.skip("legacy calibration wiring removed with P-qfull")
class TestBuildOnlyWiring(_Base):
    DEVICE = "cpu"

    def _final_batch(self, n_tokens, req_idx=7, new_tokens=None):
        return _Batch(final=True, new_tokens=n_tokens if new_tokens is None else new_tokens, n_tokens=n_tokens, req_idx=req_idx, device=self.DEVICE)

    def _to(self, *tensors):
        return [t.to(self.DEVICE) for t in tensors]

    def test_scheduler_flags(self):
        still, done, decode, cached = _Req([], 32, 0), _Req([], 16, 0), _Req([1], 1, 0), _Req([], 20, 80)
        final, new_tokens = partition_request_flags([still, done, decode, cached], still, is_extend=True, seq_lens=[32, 48, 49, 100])
        self.assertEqual(final, [False, True, False, True])
        self.assertEqual(new_tokens, [32, 48, 0, 20])

    def test_chunked_prefill_rolling_tail_then_build(self):
        batch = _Batch(final=False, new_tokens=24, n_tokens=24, device=self.DEVICE)
        first_q, first_w = self._to(*_queries(16, seed=3))
        note_prefill_queries(batch, 5, first_q, first_w)
        last_q, last_w = self._to(*_queries(8, seed=4))
        batch.prefill_final_cpu = [True]
        note_prefill_queries(batch, 5, last_q, last_w)
        tail = STATE.tail(5, 7)
        self.assertEqual(tuple(tail.q.shape), (CALIB_QUERIES, 64, 128))
        self.assertTrue(torch.equal(tail.q[:16].view(torch.uint8), first_q.view(torch.uint8)))
        self.assertTrue(torch.equal(tail.q[16:].view(torch.uint8), last_q.view(torch.uint8)))

        topk = torch.arange(2048)
        before = topk.clone()
        k, scale = self._to(*_keys(512))
        k_before = k.view(torch.uint8).clone()
        batch.seq_lens_cpu = torch.tensor([512])
        batch.prefill_new_tokens_cpu = [512]
        schedule_prefill_partition(batch, 5, k_fp8=k, k_scale=scale, scale_fmt=None)
        finish_prefill_partitions()
        self.assertTrue(torch.equal(topk, before))
        self.assertTrue(torch.equal(k.view(torch.uint8), k_before))  # raw K untouched
        part = get_prefill_partition(5, 7)
        self.assertIsNotNone(part)
        self.assertEqual(part.leaf_start.numel(), 512 // 8)
        self.assertEqual(part.meta["backend"], "gpu")
        self.assertIsNotNone(get_gpu_partition(5, 7))
        entry = get_summary_entry(5, 7)
        self.assertIsNotNone(entry)
        self.assertEqual((entry.capacity, len(entry.pages), entry.n_complete), (64, 1, 512))
        # summaries equal an independent recomputation from the same leaves
        pool = get_summary_pool(batch.token_to_kv_pool, 8)
        k_out, s_out = pool.read(entry)
        ref_fp8, ref_scale = requant_summaries(leaf_key_means(k, scale, entry.leaf_start, entry.leaf_len), None)
        self.assertTrue(torch.equal(k_out.view(torch.uint8), ref_fp8.view(torch.uint8)))
        self.assertTrue(torch.allclose(s_out, ref_scale.reshape(-1)))

    def test_reuse_logits_path_matches_fallback_scorer(self):
        q, w = self._to(*_queries(CALIB_QUERIES, seed=8))
        k, scale = self._to(*_keys(1024, seed=2))
        a = self._final_batch(1024, req_idx=1)
        note_prefill_queries(a, 2, q, w)
        schedule_prefill_partition(a, 2, k_fp8=k, k_scale=scale, scale_fmt=None)
        b = self._final_batch(1024, req_idx=2)
        note_prefill_queries(b, 2, q, w)
        prepared = prepare_prefill_partition(b, 2, max_tokens=1024)
        self.assertEqual((prepared.n_tokens, prepared.n_complete, prepared.req_idx), (1024, 1024, 2))
        # the "piggybacked" rows: same unmasked scorer output
        scores = calibration_scores(prepared.q, prepared.w, k, scale, 1024)
        schedule_prefill_partition_scores(prepared, scores, BuildContext(k, scale, None, b.token_to_kv_pool))
        finish_prefill_partitions()
        pa, pb = get_prefill_partition(2, 1), get_prefill_partition(2, 2)
        self.assertTrue(torch.equal(pa.leaf_start, pb.leaf_start) and torch.equal(pa.leaf_len, pb.leaf_len))
        ea, eb = get_summary_entry(2, 1), get_summary_entry(2, 2)
        self.assertNotEqual(sorted(ea.pages), sorted(eb.pages))
        pool = get_summary_pool(b.token_to_kv_pool, 8)
        self.assertTrue(torch.equal(pool.read(ea)[0].view(torch.uint8), pool.read(eb)[0].view(torch.uint8)))

    def test_short_tail_and_prefix_hit_fallbacks(self):
        q, w = self._to(*_queries(CALIB_QUERIES, seed=9))
        k, scale = self._to(*_keys(300))
        # 100 tokens: nothing sealed -> partition with no leaves, no summary pages
        short = self._final_batch(100, req_idx=3)
        note_prefill_queries(short, 1, q, w)
        schedule_prefill_partition(short, 1, k_fp8=k[:100], k_scale=scale[:100], scale_fmt=None)
        finish_prefill_partitions()
        part = get_prefill_partition(1, 3)
        self.assertIsNotNone(part)
        self.assertEqual((part.n_complete, part.leaf_start.numel(), part.tail_len), (0, 0, 100))
        self.assertIsNone(get_summary_entry(1, 3))
        # prefix hit leaves < 24 new tokens: refuse, keep the reason
        hit = self._final_batch(300, req_idx=4, new_tokens=10)
        note_prefill_queries(hit, 1, q, w)
        schedule_prefill_partition(hit, 1, k_fp8=k, k_scale=scale, scale_fmt=None)
        self.assertEqual(prefill_skip_reason(1, 4), "prefix_hit_new_tokens_lt_24")
        self.assertIsNone(get_prefill_partition(1, 4))
        # a request that never retained a tail
        cold = self._final_batch(300, req_idx=5)
        schedule_prefill_partition(cold, 1, k_fp8=k, k_scale=scale, scale_fmt=None)
        self.assertEqual(prefill_skip_reason(1, 5), "calibration_tail_lt_24")
        # not the final chunk: nothing happens
        mid = _Batch(final=False, new_tokens=300, n_tokens=300, req_idx=6, device=self.DEVICE)
        note_prefill_queries(mid, 1, q, w)
        schedule_prefill_partition(mid, 1, k_fp8=k, k_scale=scale, scale_fmt=None)
        self.assertIsNone(get_prefill_partition(1, 6))
        self.assertIsNone(prefill_skip_reason(1, 6))

    def test_request_release_and_slot_reuse(self):
        q, w = self._to(*_queries(CALIB_QUERIES, seed=10))
        k, scale = self._to(*_keys(768))
        batch = self._final_batch(768, req_idx=7)
        for layer in (0, 1):
            note_prefill_queries(batch, layer, q, w)
            schedule_prefill_partition(batch, layer, k_fp8=k, k_scale=scale, scale_fmt=None)
        finish_prefill_partitions()
        pool = get_summary_pool(batch.token_to_kv_pool, 8)
        pages = {layer: sorted(get_summary_entry(layer, 7).pages) for layer in (0, 1)}
        self.assertEqual(pool.stats()["entries"], 2)
        epoch = STATE.epoch(7)
        release_request(7)
        self.assertEqual(STATE.epoch(7), epoch + 1)
        self.assertIsNone(STATE.tail(0, 7))
        self.assertIsNone(get_prefill_partition(0, 7))
        self.assertIsNone(get_summary_entry(0, 7))
        self.assertEqual(pool.stats()["pages_used"], 0)
        # a new request in the same slot rebuilds and reuses the freed pages
        again = self._final_batch(768, req_idx=7)
        again.token_to_kv_pool = batch.token_to_kv_pool
        for layer in (0, 1):
            note_prefill_queries(again, layer, q, w)
            schedule_prefill_partition(again, layer, k_fp8=k, k_scale=scale, scale_fmt=None)
        finish_prefill_partitions()
        self.assertEqual({layer: sorted(get_summary_entry(layer, 7).pages) for layer in (0, 1)}, pages)
        self.assertEqual(get_summary_entry(0, 7).epoch, epoch + 1)

    def test_merge_enabled_runtime_keeps_capacity_and_marks_padding(self):
        _set_env(GPU_MERGE)
        q, w = self._to(*_queries(CALIB_QUERIES, seed=14))
        k, scale = self._to(*_keys(4096))
        batch = self._final_batch(4096, req_idx=8)
        note_prefill_queries(batch, 3, q, w)
        schedule_prefill_partition(batch, 3, k_fp8=k, k_scale=scale, scale_fmt=None)
        finish_prefill_partitions()
        part = get_prefill_partition(3, 8)
        gpu = get_gpu_partition(3, 8)
        entry = get_summary_entry(3, 8)
        self.assertEqual(part.method, "P-qfull-sync_nonoverlap")
        self.assertEqual(gpu.capacity, 512)
        self.assertEqual(entry.capacity, 512)
        self.assertEqual(part.leaf_start.numel(), int(gpu.num_leaves))
        self.assertLessEqual(len(part.meta["round_merge_counts"]), 2)
        self.assertTrue(bool((gpu.leaf_len[int(gpu.num_leaves) :] == 0).all()))
        validate_leaves(part.leaf_start, part.leaf_len, 4096, allow_arbitrary=True, max_len=0)

    def test_cpu_reference_backend_still_wires(self):
        """The previously A/B'ed numba path (v7_reuse) is unchanged: no summaries."""
        _set_env(V7_REUSE)
        q, w = self._to(*_queries(CALIB_QUERIES, seed=15))
        k, scale = self._to(*_keys(2048))
        batch = self._final_batch(2048, req_idx=9)
        note_prefill_queries(batch, 4, q, w)
        schedule_prefill_partition(batch, 4, k_fp8=k, k_scale=scale, scale_fmt=None)
        finish_prefill_partitions()
        part = get_prefill_partition(4, 9)
        self.assertEqual(part.method, "P-qfull-heap_reference")
        self.assertEqual(part.meta["backend"], "cpu_reference")
        self.assertLessEqual(int(part.leaf_len.max()), 256)
        self.assertIsNone(get_gpu_partition(4, 9))
        self.assertIsNone(get_summary_entry(4, 9))

    def test_backends_agree_through_the_runtime(self):
        q, w = self._to(*_queries(CALIB_QUERIES, seed=16))
        k, scale = self._to(*_keys(1536, seed=4))
        _set_env(GPU)
        a = self._final_batch(1536, req_idx=11)
        note_prefill_queries(a, 0, q, w)
        schedule_prefill_partition(a, 0, k_fp8=k, k_scale=scale, scale_fmt=None)
        finish_prefill_partitions()
        gpu_part = get_prefill_partition(0, 11)
        _set_env(V7_REUSE, MERGE_POLICY="off")
        b = self._final_batch(1536, req_idx=12)
        note_prefill_queries(b, 0, q, w)
        schedule_prefill_partition(b, 0, k_fp8=k, k_scale=scale, scale_fmt=None)
        finish_prefill_partitions()
        cpu_part = get_prefill_partition(0, 12)
        self.assertTrue(torch.equal(gpu_part.leaf_start, cpu_part.leaf_start))
        self.assertTrue(torch.equal(gpu_part.leaf_len, cpu_part.leaf_len))


# --------------------------------------------------------------------------- #
# CUDA: real kernels
# --------------------------------------------------------------------------- #


@unittest.skip("legacy query-score CUDA tests removed with P-qfull")
@unittest.skipUnless(torch.cuda.is_available(), "CUDA")
class TestGpuBackendCuda(TestGpuBackend):
    DEVICE = "cuda"

    def test_triton_kernels_match_torch_fallback(self):
        """Fused per-root kernels (tree, λ rounds, mask, gain, staircase, merge) == torch ops."""
        from sglang.srt.layers.attention.nsa.adaptive_hisa import partition_gpu as pg
        from sglang.srt.layers.attention.nsa.adaptive_hisa import partition_kernels as pk

        _set_env(GPU_MERGE)
        cfg = get_config()
        for kind, n in (("random", 4096), ("zero", 2048), ("piecewise", 8192), ("sparse", 4096), ("random", 32768)):
            scores = _scores(kind, n, seed=21).cuda()
            with_triton = build_partition_gpu(scores, n, cfg)
            flat_t, layout, _ = pg.build_score_tree(scores, atom=cfg.atom, root=cfg.root)
            pk.HAS_TRITON = False
            try:
                flat_r, _, _ = pg.build_score_tree(scores, atom=cfg.atom, root=cfg.root)
                without = build_partition_gpu(scores, n, cfg)
            finally:
                pk.HAS_TRITON = True
            self.assertTrue(torch.allclose(flat_t, flat_r, rtol=1e-12, atol=1e-9), (kind, n))
            self.assertEqual(int(with_triton.num_leaves), int(without.num_leaves), (kind, n))
            self.assertTrue(torch.equal(with_triton.leaf_start, without.leaf_start), (kind, n))
            self.assertTrue(torch.equal(with_triton.leaf_len, without.leaf_len), (kind, n))
            self.assertEqual(int(with_triton.repairs), int(without.repairs))
            self.assertEqual(with_triton.round_merges.tolist(), without.round_merges.tolist())
            self.assertLessEqual(
                abs(float(with_triton.lam) - float(without.lam)), 1e-12 * max(1.0, abs(float(without.lam)))
            )

    def test_graph_replay_matches_eager(self):
        """CUDA-graph replay of the build == eager, across sizes, strided input, cache reuse."""
        from sglang.srt.layers.attention.nsa.adaptive_hisa import partition_gpu as pg

        _set_env(GPU_MERGE)
        cfg = get_config()
        pg.reset_graph_cache()
        try:
            for kind, n in (("random", 256), ("piecewise", 4096), ("sparse", 4096 + 100), ("random", 32768)):
                for seed in (1, 2):
                    full = _scores(kind, n + 512, seed=seed).cuda()
                    scores = full[:, :n]  # strided view, like a slice of the logits
                    eager = build_partition_gpu(scores, n, cfg).to_host()
                    graph = pg.graphed_builder(scores.shape[0], n, cfg, scores.device)
                    self.assertIsNotNone(graph)
                    replay = graph.build(scores, n).to_host()
                    self.assertTrue(torch.equal(eager.leaf_start, replay.leaf_start), (kind, n, seed))
                    self.assertTrue(torch.equal(eager.leaf_len, replay.leaf_len), (kind, n, seed))
                    for key in ("lambda", "dp_leaves", "repairs", "round_merge_counts", "M_after_merge", "status_ok"):
                        self.assertEqual(eager.meta[key], replay.meta[key], (kind, n, seed, key))
                    self.assertEqual(replay.n_tokens, n)
                    self.assertTrue(replay.meta["graph_replay"])
                # same N and C -> same cached graph (one capture per request length)
                again = pg.graphed_builder(scores.shape[0], n, cfg, scores.device)
                self.assertIs(again, graph)
                self.assertGreaterEqual(graph.replays, 2)
            self.assertLessEqual(len(pg._GRAPH_CACHE), pg._GRAPH_CACHE_MAX)
            # tail-only difference keeps the same graph (same sealed prefix)
            self.assertTrue(graph.matches(CALIB_QUERIES, 32768 + 17, cfg))
            self.assertFalse(graph.matches(CALIB_QUERIES, 32768 + 256, cfg))
            # a short tail (nothing sealed) must not disable the graph path
            self.assertIsNone(pg.graphed_builder(CALIB_QUERIES, ROOT - 1, cfg, torch.device("cuda")))
            self.assertFalse(pg._GRAPH_DISABLED)
            self.assertIsNotNone(pg.graphed_builder(CALIB_QUERIES, ROOT, cfg, torch.device("cuda")))
        finally:
            pg.reset_graph_cache()

    def test_graph_capture_oom_is_retried_then_cooled_down_not_disabled(self):
        """An OOM inside capture: evict + empty_cache + one retry; a second OOM
        skips the graph path for a cooldown window only. Eviction happens before
        the new capture so at most MAX-1 old pools coexist with the new one."""
        from unittest import mock

        from sglang.srt.layers.attention.nsa.adaptive_hisa import partition_gpu as pg

        _set_env(GPU_MERGE)
        cfg = get_config()
        dev = torch.device("cuda")
        pg.reset_graph_cache()
        real_init = pg.GraphedBuilder.__init__
        try:
            # fill the cache and note eviction order
            first = pg.graphed_builder(CALIB_QUERIES, ROOT * 4, cfg, dev)
            second = pg.graphed_builder(CALIB_QUERIES, ROOT * 5, cfg, dev)
            self.assertEqual(len(pg._GRAPH_CACHE), pg._GRAPH_CACHE_MAX)
            seen = []

            def counting_init(self_, *a, **k):
                seen.append(len(pg._GRAPH_CACHE))  # cache size while capturing
                return real_init(self_, *a, **k)

            with mock.patch.object(pg.GraphedBuilder, "__init__", counting_init):
                third = pg.graphed_builder(CALIB_QUERIES, ROOT * 6, cfg, dev)
            self.assertIsNotNone(third)
            self.assertEqual(seen, [pg._GRAPH_CACHE_MAX - 1])  # trimmed before the capture
            self.assertNotIn(first, pg._GRAPH_CACHE.values())
            self.assertIn(second, pg._GRAPH_CACHE.values())

            # OOM once -> retried after releasing memory, graph still produced
            calls = {"n": 0}

            def oom_once(self_, *a, **k):
                calls["n"] += 1
                if calls["n"] == 1:
                    raise torch.OutOfMemoryError("CUDA out of memory. injected")
                return real_init(self_, *a, **k)

            with mock.patch.object(pg.GraphedBuilder, "__init__", oom_once), \
                    mock.patch.object(torch.cuda, "empty_cache", wraps=torch.cuda.empty_cache) as ec:
                g = pg.graphed_builder(CALIB_QUERIES, ROOT * 7, cfg, dev)
            self.assertIsNotNone(g)
            self.assertEqual(calls["n"], 2)
            self.assertEqual(ec.call_count, 1)
            self.assertFalse(pg._GRAPH_DISABLED)
            self.assertEqual(pg._GRAPH_SKIP, 0)
            scores = _scores("random", ROOT * 7, seed=3).cuda()
            replay = g.build(scores, ROOT * 7).to_host()
            eager = build_partition_gpu(scores, ROOT * 7, cfg).to_host()
            self.assertTrue(torch.equal(eager.leaf_start, replay.leaf_start))

            # OOM twice -> eager for the cooldown window, cached graphs still served, then retry
            def oom_always(self_, *a, **k):
                raise torch.OutOfMemoryError("CUDA out of memory. injected")

            with mock.patch.object(pg.GraphedBuilder, "__init__", oom_always), self.assertLogs(pg.__name__, level="WARNING"):
                self.assertIsNone(pg.graphed_builder(CALIB_QUERIES, ROOT * 8, cfg, dev))
            self.assertFalse(pg._GRAPH_DISABLED)
            self.assertEqual(pg._GRAPH_SKIP, pg._GRAPH_OOM_COOLDOWN)
            self.assertEqual(len(pg._GRAPH_CACHE), 0)  # everything was released for the retry
            cooled = pg.graphed_builder(CALIB_QUERIES, ROOT * 7, cfg, dev)  # miss during cooldown
            self.assertIsNone(cooled)
            self.assertEqual(pg._GRAPH_SKIP, pg._GRAPH_OOM_COOLDOWN - 1)
            pg._GRAPH_SKIP = 1
            self.assertIsNone(pg.graphed_builder(CALIB_QUERIES, ROOT * 8, cfg, dev))
            self.assertIsNotNone(pg.graphed_builder(CALIB_QUERIES, ROOT * 8, cfg, dev))  # cooldown over
            self.assertEqual(pg._GRAPH_STATS["oom"], 3)
            self.assertEqual(pg._GRAPH_STATS["oom_retry_ok"], 1)

            # a non-memory capture error still disables the graph path
            def boom(self_, *a, **k):
                raise RuntimeError("capture failed")

            with mock.patch.object(pg.GraphedBuilder, "__init__", boom), self.assertLogs(pg.__name__, level="ERROR"):
                self.assertIsNone(pg.graphed_builder(CALIB_QUERIES, ROOT * 9, cfg, dev))
            self.assertTrue(pg._GRAPH_DISABLED)
        finally:
            pg.reset_graph_cache()


@unittest.skipUnless(torch.cuda.is_available(), "CUDA")
class TestSummariesCuda(TestSummaries):
    DEVICE = "cuda"

    def test_ue8m0_matches_act_quant(self):
        cfg = get_config()
        part = build_partition_gpu(_scores("random", 8192).cuda(), 8192, cfg)
        fp8, scale = _keys(8192)
        fp8, scale = fp8.cuda(), scale.cuda()
        means = leaf_key_means(fp8, scale, part.leaf_start, part.leaf_len)
        q8, qs = requant_summaries(means, "ue8m0")
        r8, rs = _torch_act_quant(means.to(torch.bfloat16), True)
        self.assertTrue(torch.equal(qs.reshape(-1), rs.reshape(-1)))
        self.assertTrue(torch.equal(q8.view(torch.uint8), r8.view(torch.uint8)))


@unittest.skipUnless(torch.cuda.is_available(), "CUDA")
class TestBuildOnlyWiringCuda(TestBuildOnlyWiring):
    DEVICE = "cuda"

    def test_ready_event_and_profile_meta(self):
        os.environ[_ENV + "PARTITION_LOG_LAYERS"] = "1"
        try:
            _set_env(GPU_MERGE)
            q, w = self._to(*_queries(CALIB_QUERIES, seed=17))
            k, scale = self._to(*_keys(4096))
            batch = self._final_batch(4096, req_idx=13)
            note_prefill_queries(batch, 0, q, w)
            schedule_prefill_partition(batch, 0, k_fp8=k, k_scale=scale, scale_fmt="ue8m0")
            finish_prefill_partitions()
            entry = get_summary_entry(0, 13)
            self.assertIsNotNone(entry.ready_event)
            entry.wait()
            part = get_prefill_partition(0, 13)
            # Graph replay only times the whole build; eager has per-stage events.
            keys = ("score_s", "summary_s", "total_s")
            if not part.meta.get("graph_replay"):
                keys += ("tree_s", "dp_s", "repair_s", "merge_s")
            for key in keys:
                self.assertIsInstance(part.meta[key], float)
                self.assertGreaterEqual(part.meta[key], 0.0)
        finally:
            os.environ.pop(_ENV + "PARTITION_LOG_LAYERS", None)

    def test_side_stream_matches_main_stream(self):
        """GPU_STREAM=side: same leaves and summaries as the in-stream build, joined at finish."""
        from sglang.srt.layers.attention.nsa.adaptive_hisa import prefill_runtime as rt

        q, w = self._to(*_queries(CALIB_QUERIES, seed=23))
        k, scale = self._to(*_keys(ROOT * 24 + 40))
        results = {}
        for stream, req in (("main", 31), ("side", 32)):
            _set_env(GPU_MERGE, GPU_STREAM=stream)
            batch = self._final_batch(ROOT * 24 + 40, req_idx=req)
            note_prefill_queries(batch, 3, q, w)
            schedule_prefill_partition(batch, 3, k_fp8=k, k_scale=scale, scale_fmt="ue8m0")
            if stream == "side":
                self.assertTrue(rt._STATS.get("side_used"))
            finish_prefill_partitions()
            self.assertFalse(rt._STATS.get("side_used"))
            part = get_prefill_partition(3, req)
            entry = get_summary_entry(3, req)
            entry.wait()
            from sglang.srt.layers.attention.nsa.adaptive_hisa import summary_pool as _sp

            fp8, sc = _sp._POOL.read(entry)
            results[stream] = (part, fp8.clone(), sc.clone())
            self.assertEqual(part.meta["stream"], stream)
        main, side = results["main"], results["side"]
        self.assertTrue(torch.equal(main[0].leaf_start, side[0].leaf_start))
        self.assertTrue(torch.equal(main[0].leaf_len, side[0].leaf_len))
        self.assertTrue(torch.equal(main[1].view(torch.uint8), side[1].view(torch.uint8)))
        self.assertTrue(torch.equal(main[2], side[2]))

    def test_cpu_reference_overlap_matches_sync(self):
        _set_env(V7_REUSE)
        q, w = self._to(*_queries(CALIB_QUERIES, seed=6))
        k, scale = self._to(*_keys(ROOT * 4))
        sync = build_prefill_partition(k, scale, q, w, ROOT * 4, merge=True)
        _set_env(V7_REUSE, PARTITION_OVERLAP="1")
        batch = self._final_batch(ROOT * 4, req_idx=9)
        note_prefill_queries(batch, 2, q, w)
        topk = torch.full((1, 2048), -1, device="cuda")
        held = topk.clone()
        schedule_prefill_partition(batch, 2, k_fp8=k, k_scale=scale)
        finish_prefill_partitions()
        self.assertTrue(torch.equal(topk, held))
        got = get_prefill_partition(2, 9)
        self.assertTrue(torch.equal(got.leaf_start, sync.leaf_start))
        self.assertTrue(torch.equal(got.leaf_len, sync.leaf_len))
        self.assertEqual(got.method, "P-qfull-heap_reference")
        self.assertEqual(got.meta["premerge_M"], ROOT * 4 // 8)

    def test_appended_fp8_scorer_preserves_official_rows_and_partition(self):
        import deep_gemm

        _set_env(GPU_MERGE)
        cfg = get_config()
        rows = 32
        q, w = self._to(*_queries(rows, seed=13))
        k, scale = self._to(*_keys(ROOT * 4))
        n = ROOT * 4
        ks = torch.zeros(rows, dtype=torch.int32, device="cuda")
        ke = torch.full((rows,), n, dtype=torch.int32, device="cuda")
        official = deep_gemm.fp8_mqa_logits(q, (k, scale), w, ks, ke, clean_logits=False)
        tail_q, tail_w = q[-CALIB_QUERIES:], w[-CALIB_QUERIES:]
        combined = deep_gemm.fp8_mqa_logits(
            torch.cat((q, tail_q)),
            (k, scale),
            torch.cat((w, tail_w)),
            torch.cat((ks, torch.zeros(CALIB_QUERIES, dtype=torch.int32, device="cuda"))),
            torch.cat((ke, torch.full((CALIB_QUERIES,), n, dtype=torch.int32, device="cuda"))),
            clean_logits=False,
        )
        self.assertTrue(torch.equal(official, combined[:rows]))  # official rows untouched
        canonical = calibration_scores(tail_q, tail_w, k, scale, n)
        piggyback = combined[rows:]
        self.assertLess(float((canonical - piggyback).abs().max() / canonical.abs().max()), 5e-4)
        # gpu backend on the piggybacked rows == cpu reference on the same rows
        gpu_part = build_partition_gpu(piggyback, n, cfg).to_host()
        cpu_cfg = replace(cfg, split_backend="cpu_reference", build_summaries=False)
        cpu_part = partition_cpu_phase(partition_scores_gpu_phase(piggyback, n), cfg=cpu_cfg)
        self.assertTrue(torch.equal(gpu_part.leaf_start, cpu_part.leaf_start))
        self.assertTrue(torch.equal(gpu_part.leaf_len, cpu_part.leaf_len))


# --------------------------------------------------------------------------- #
# P-key (partition_metric=key_sse): query-independent control
# --------------------------------------------------------------------------- #


class TestKeyMetric(_Base):
    ENV = dict(GPU_MERGE, PARTITION_METRIC="key_sse")
    DEVICE = "cpu"

    def _to(self, *tensors):
        return [t.to(self.DEVICE) for t in tensors]

    def test_config_contract(self):
        cfg = get_config()
        self.assertEqual((cfg.partition_metric, cfg.method_name, cfg.energy_rows), ("key_sse", "P-key", 128))
        _set_env(self.ENV, MODE="adaptive_decode", BUILD_SUMMARIES="1")
        self.assertEqual(get_config().mode, "adaptive_decode")
        _set_env(self.ENV, PARTITION_METRIC="radius")
        with self.assertRaises(PartitionConfigError):
            get_config()

    def test_key_energies_equal_direct_key_sse(self):
        k, scale = self._to(*_keys(1024, seed=5))
        values = key_values(k, scale, 1024)
        self.assertEqual(tuple(values.shape), (128, 1024))
        keys = k.float() * scale[:, None]
        energies = score_energies(values, atom=ATOM, root=ROOT)
        for level, e in enumerate(energies):
            size = ATOM << level
            for b in range(min(e.numel(), 6)):
                chunk = keys[b * size : (b + 1) * size].double()
                direct = float(((chunk - chunk.mean(0)) ** 2).sum())
                self.assertAlmostEqual(float(e[b]), direct, delta=1e-6 * max(1.0, direct))

    def test_gpu_backend_matches_cpu_reference_and_ignores_queries(self):
        cfg = get_config()
        n = ROOT * 9 + 40
        k, scale = self._to(*_keys(n, seed=6))
        ref = partition_cpu_phase(partition_keys_gpu_phase(k, scale, n, atom=ATOM, root=ROOT), cfg=cfg)
        gpu = build_partition_gpu(key_values(k, scale, ref.n_complete), n, cfg).to_host()
        self.assertEqual(ref.method, "P-key-sync_nonoverlap")
        self.assertEqual(gpu.method, "P-key-sync_nonoverlap")
        self.assertEqual(gpu.meta["metric"], "key_sse")
        self.assertTrue(torch.equal(ref.leaf_start.cpu(), gpu.leaf_start.cpu()))
        self.assertTrue(torch.equal(ref.leaf_len.cpu(), gpu.leaf_len.cpu()))
        validate_leaves(
            gpu.leaf_start.cpu(), gpu.leaf_len.cpu(), ref.n_complete,
            allow_arbitrary=True, max_len=cfg.max_merge_len,
        )
    def test_target_merge_reaches_l_over_64_and_matches_reference(self):
        from sglang.srt.layers.attention.nsa.adaptive_hisa.partition_gpu import target_merge_threshold

        _set_env(self.ENV, MERGE_TARGET_DIVISOR="64")
        cfg = get_config()
        self.assertEqual((cfg.merge_target_divisor, cfg.merge_target_rounds), (64, 8))
        self.assertIn("merge_target=L/64@8", cfg.describe())
        n = ROOT * 17 + 40
        k, scale = self._to(*_keys(n, seed=6))
        ref = partition_cpu_phase(partition_keys_gpu_phase(k, scale, n, atom=ATOM, root=ROOT), cfg=cfg)
        gpu = build_partition_gpu(key_values(k, scale, ref.n_complete), n, cfg).to_host()
        target = ref.n_complete // 64
        self.assertEqual(cfg.merge_target(ref.n_complete), target)
        self.assertEqual(ref.method, "P-key-sync_nonoverlap-target64")
        self.assertEqual(gpu.method, ref.method)
        self.assertEqual(gpu.meta["merge_target"], target)
        # M0 = L/8 before merging, exactly L/64 after (random keys have no cost ties)
        self.assertEqual(ref.meta["premerge_M"], ref.n_complete // 8)
        self.assertEqual(int(ref.leaf_len.numel()), target)
        self.assertEqual(int(gpu.leaf_len.numel()), target)
        self.assertLess(len(ref.meta["round_merge_counts"]), cfg.merge_target_rounds)
        self.assertTrue(torch.equal(ref.leaf_start.cpu(), gpu.leaf_start.cpu()))
        self.assertTrue(torch.equal(ref.leaf_len.cpu(), gpu.leaf_len.cpu()))
        validate_leaves(gpu.leaf_start.cpu(), gpu.leaf_len.cpu(), ref.n_complete, allow_arbitrary=True, max_len=0)
        # the threshold admits exactly the (count - target) cheapest edges, ties included
        cost = torch.tensor([3.0, 1.0, float("inf"), 1.0, 2.0], dtype=torch.float64, device=self.DEVICE)
        count = torch.tensor(6, device=self.DEVICE)
        thr = target_merge_threshold(cost, count, torch.tensor(4, device=self.DEVICE))
        self.assertEqual(int((cost < thr).sum()), 2)
        thr = target_merge_threshold(cost, count, torch.tensor(5, device=self.DEVICE))
        self.assertEqual(int((cost < thr).sum()), 2)  # 1.0 tie at the boundary
        thr = target_merge_threshold(cost, count, torch.tensor(6, device=self.DEVICE))
        self.assertEqual(int((cost < thr).sum()), 0)
        # divisor below the split compression is rejected
        _set_env(self.ENV, MERGE_TARGET_DIVISOR="4")
        with self.assertRaises(PartitionConfigError):
            get_config()

    def test_wiring_builds_without_any_query_tail(self):
        n = 640
        batch = _Batch(final=True, new_tokens=n, n_tokens=n, req_idx=3, device=self.DEVICE)
        k, scale = self._to(*_keys(n, seed=1))
        schedule_prefill_partition(batch, 1, k_fp8=k, k_scale=scale, scale_fmt=None)
        finish_prefill_partitions()
        self.assertIsNone(prefill_skip_reason(1, 3))
        part = get_prefill_partition(1, 3)
        self.assertIsNotNone(part)
        self.assertEqual(part.method, "P-key-sync_nonoverlap")
        self.assertEqual(part.n_complete, 512)
        self.assertLessEqual(part.leaf_start.numel(), 512 // 8)
        entry = get_summary_entry(1, 3)
        self.assertIsNotNone(entry)
        self.assertEqual(entry.n_complete, 512)
        pool = get_summary_pool(batch.token_to_kv_pool, 8)
        k_out, _ = pool.read(entry)
        ref_fp8, _ = requant_summaries(leaf_key_means(k, scale, entry.leaf_start, entry.leaf_len), None)
        self.assertTrue(torch.equal(k_out.view(torch.uint8), ref_fp8.view(torch.uint8)))

    def test_cpu_reference_backend_builds_from_keys(self):
        _set_env(self.ENV, SPLIT_BACKEND="cpu_reference", MERGE_POLICY="sync_nonoverlap")
        n = ROOT * 5
        batch = _Batch(final=True, new_tokens=n, n_tokens=n, req_idx=4, device=self.DEVICE)
        k, scale = self._to(*_keys(n, seed=3))
        schedule_prefill_partition(batch, 0, k_fp8=k, k_scale=scale, scale_fmt=None)
        finish_prefill_partitions()
        part = get_prefill_partition(0, 4)
        self.assertIsNotNone(part)
        self.assertEqual(part.method, "P-key-sync_nonoverlap")
        self.assertEqual(part.meta["backend"], "cpu_reference")
        direct = build_prefill_partition_from_keys(k, scale, n, cfg=get_config())
        self.assertTrue(torch.equal(part.leaf_start, direct.leaf_start))
        self.assertTrue(torch.equal(part.leaf_len, direct.leaf_len))

    def test_lambda_round_cap_and_leaf_deficit_early_stop(self):
        """``lambda_max_rounds`` caps the search; ``lambda_deficit_tol`` stops on leaf count.

        Early stop must (a) be a no-op when disabled, (b) only trigger once the
        feasible end already has ``budget - count <= tol`` DP leaves, and (c)
        still yield exactly ``budget`` leaves after the exact repair.
        """
        from sglang.srt.layers.attention.nsa.adaptive_hisa import partition_gpu as pg

        cfg = get_config()
        for kind, n in ((11, ROOT * 16), (12, ROOT * 32 + 7), (13, ROOT * 8)):
            done = n_complete_tokens(n, ROOT)
            scores = key_values(*self._to(*_keys(n, seed=kind)), done)
            budget = done // cfg.summary_compression
            flat, layout, _ = pg.build_score_tree(scores[:, :done], atom=cfg.atom, root=cfg.root)
            k, tol = cfg.lambda_candidates, cfg.lambda_rel_tol
            lam0, _, r0, used0 = pg.search_lambda(flat, layout, budget, candidates=k, rel_tol=tol)
            lam1, _, r1, used1 = pg.search_lambda(
                flat, layout, budget, candidates=k, rel_tol=tol, max_rounds=0, deficit_tol=-1.0
            )
            self.assertEqual((r0, int(used0)), (r1, int(used1)))
            self.assertEqual(r0, int(used0))
            self.assertEqual(float(lam0), float(lam1))
            # exact-count early stop
            lam2, _, r2, used2 = pg.search_lambda(
                flat, layout, budget, candidates=k, rel_tol=tol, deficit_tol=0.0
            )
            self.assertEqual(r2, r0)
            self.assertLessEqual(int(used2), r2)
            count2 = int(pg.dp_counts(flat, layout, lam2.reshape(1))[0])
            self.assertLessEqual(count2, budget)
            if int(used2) < r2:
                self.assertEqual(count2, budget, (kind, n))
            # ratio tolerance: stop as soon as the deficit is within 1% of budget
            lam3, _, r3, used3 = pg.search_lambda(
                flat, layout, budget, candidates=k, rel_tol=tol, max_rounds=5, deficit_tol=0.01
            )
            self.assertEqual(r3, 5)
            self.assertLessEqual(int(used3), 5)
            count3 = int(pg.dp_counts(flat, layout, lam3.reshape(1))[0])
            self.assertLessEqual(count3, budget)
            if int(used3) < 5:
                self.assertLessEqual(budget - count3, int(0.01 * budget))
            # end to end: repair still fills to the exact budget
            for cfg_i in (
                replace(cfg, lambda_max_rounds=4),
                replace(cfg, lambda_max_rounds=5, lambda_deficit_tol=0.001),
                replace(cfg, lambda_deficit_tol=0.0),
            ):
                part = build_partition_gpu(scores, n, cfg_i.validate())
                self.assertTrue(bool(part.status_ok), (kind, n, cfg_i.lambda_max_rounds))
                self.assertEqual(int(part.dp_leaves) + int(part.repairs), budget)  # pre-merge M0
                self.assertLessEqual(int(part.meta["lambda_rounds_used"]), int(part.meta["lambda_rounds"]))
                host = part.to_host()
                validate_leaves(host.leaf_start, host.leaf_len, done, allow_arbitrary=True)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestKeyMetricCuda(TestKeyMetric):
    DEVICE = "cuda"

    def test_lambda_early_stop_triton_matches_torch(self):
        """Device-side ``done`` flag path == torch fallback: same λ, same rounds used."""
        from sglang.srt.layers.attention.nsa.adaptive_hisa import partition_gpu as pg
        from sglang.srt.layers.attention.nsa.adaptive_hisa import partition_kernels as pk

        cfg = get_config()
        for kind, n in ((23, ROOT * 16), (24, ROOT * 64), (25, ROOT * 128)):
            scores = key_values(*self._to(*_keys(n, seed=kind)), n)
            flat, layout, _ = pg.build_score_tree(scores, atom=cfg.atom, root=cfg.root)
            budget = n // cfg.summary_compression
            for max_rounds, dtol in ((0, 0.0), (5, 0.001), (4, 0.01), (10, -1.0)):
                kw = dict(candidates=cfg.lambda_candidates, rel_tol=cfg.lambda_rel_tol,
                          max_rounds=max_rounds, deficit_tol=dtol)
                lam_t, _, r_t, used_t = pg.search_lambda(flat, layout, budget, **kw)
                pk.HAS_TRITON = False
                try:
                    lam_r, _, r_r, used_r = pg.search_lambda(flat, layout, budget, **kw)
                finally:
                    pk.HAS_TRITON = True
                self.assertEqual((r_t, int(used_t)), (r_r, int(used_r)), (kind, n, max_rounds, dtol))
                self.assertLessEqual(abs(float(lam_t) - float(lam_r)), 1e-12 * max(1.0, abs(float(lam_r))))

    def test_graph_replay_matches_eager(self):
        from sglang.srt.layers.attention.nsa.adaptive_hisa.partition_gpu import (
            graphed_builder,
            reset_graph_cache,
        )

        cfg = get_config()
        n = ROOT * 33 + 5
        k, scale = self._to(*_keys(n, seed=11))
        values = key_values(k, scale, n_complete_tokens(n, ROOT))
        eager = build_partition_gpu(values, n, cfg).to_host()
        reset_graph_cache()
        graph = graphed_builder(values.shape[0], n, cfg, values.device)
        self.assertIsNotNone(graph)
        self.assertEqual(graph.n_queries, 128)
        replay = graph.build(values, n).to_host()
        self.assertTrue(torch.equal(eager.leaf_start.cpu(), replay.leaf_start.cpu()))
        self.assertTrue(torch.equal(eager.leaf_len.cpu(), replay.leaf_len.cpu()))
        reset_graph_cache()

    def test_target_merge_graph_replay_matches_eager(self):
        from sglang.srt.layers.attention.nsa.adaptive_hisa.partition_gpu import (
            graphed_builder,
            reset_graph_cache,
        )

        _set_env(self.ENV, MERGE_TARGET_DIVISOR="64")
        cfg = get_config()
        n = ROOT * 33 + 5
        k, scale = self._to(*_keys(n, seed=11))
        done = n_complete_tokens(n, ROOT)
        values = key_values(k, scale, done)
        eager = build_partition_gpu(values, n, cfg).to_host()
        self.assertEqual(int(eager.leaf_len.numel()), done // 64)
        reset_graph_cache()
        graph = graphed_builder(values.shape[0], n, cfg, values.device)
        self.assertIsNotNone(graph)
        replay = graph.build(values, n).to_host()
        self.assertTrue(torch.equal(eager.leaf_start.cpu(), replay.leaf_start.cpu()))
        self.assertTrue(torch.equal(eager.leaf_len.cpu(), replay.leaf_len.cpu()))
        # a second request through the same graph converges to its own target
        k2, scale2 = self._to(*_keys(n, seed=12))
        replay2 = graph.build(key_values(k2, scale2, done), n).to_host()
        self.assertEqual(int(replay2.leaf_len.numel()), done // 64)
        reset_graph_cache()


class TestRawFp8RegisterPath(unittest.TestCase):
    """raw-FP8 fused kernels stay bit-compatible with the torch reference."""

    def test_tree_totals_and_summary_match_reference(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")
        from sglang.srt.layers.attention.nsa.adaptive_hisa.partition_gpu import TreeLayout
        from sglang.srt.layers.attention.nsa.adaptive_hisa.pkey_fp8_builder import (
            _build_key_tree_torch,
            build_key_tree_from_fp8,
            build_partition_from_fp8,
            leaf_totals_from_fp8,
            summaries_from_totals,
        )
        from sglang.srt.layers.attention.nsa.adaptive_hisa.summary_kernels import requant_summaries

        device = torch.device("cuda")
        _set_env(
            dict(
                MODE="build_only",
                PARTITION_METRIC="key_sse",
                SPLIT_BACKEND="gpu",
                MERGE_POLICY="sync_nonoverlap",
                MERGE_TARGET_DIVISOR="64",
            )
        )
        cfg = get_config()
        for n in (ROOT, ROOT * 4, 4096):
            k, scale = _keys(n, seed=n)
            k, scale = k.to(device), scale.to(device)
            fused, layout = build_key_tree_from_fp8(k, scale, n, atom=ATOM, root=ROOT)
            reference = _build_key_tree_torch(k, scale, TreeLayout(ATOM, ROOT, n))
            rel = (fused - reference).abs() / reference.abs().clamp_min(1.0)
            self.assertLess(float(rel.max()), 1e-12, f"tree mismatch at N={n}")
            start = torch.arange(0, n, ATOM, device=device, dtype=torch.int64)
            length = torch.full((n // ATOM,), ATOM, device=device, dtype=torch.int64)
            # Force the dense prefix reference by calling the torch tail through a CPU tensor.
            totals = leaf_totals_from_fp8(k, scale, start, length, n)
            cpu_totals = leaf_totals_from_fp8(k.cpu(), scale.cpu(), start.cpu(), length.cpu(), n)
            rel_totals = (totals.cpu() - cpu_totals).abs() / cpu_totals.abs().clamp_min(1.0)
            self.assertLess(float(rel_totals.max()), 1e-6)
            part = build_partition_from_fp8(k, scale, n, cfg)
            self.assertEqual(int(part.num_leaves.item()), max(1, n // 64))
            finals = part.meta["_merge_totals"]
            fp8, qscale = summaries_from_totals(finals, part.leaf_len, None)
            denom = part.leaf_len.to(finals.dtype).clamp_min(1).unsqueeze(-1)
            means = torch.where(part.leaf_len.unsqueeze(-1) > 0, (finals / denom).float(), 0)
            ref_fp8, ref_scale = requant_summaries(means, None)
            self.assertTrue(torch.equal(fp8.view(torch.uint8), ref_fp8.view(torch.uint8)))
            self.assertTrue(torch.equal(qscale, ref_scale))

    def test_chunked_tree_matches_single_program_kernel(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")
        from sglang.srt.layers.attention.nsa.adaptive_hisa import partition_kernels as pk
        from sglang.srt.layers.attention.nsa.adaptive_hisa.partition_gpu import TreeLayout

        device = torch.device("cuda")
        for n in (ROOT, ROOT * 3, 4096):
            k, scale = _keys(n, seed=n + 1)
            k, scale = k.to(device), scale.to(device)
            layout = TreeLayout(ATOM, ROOT, n)
            chunked = pk.raw_fp8_tree_triton(k, scale, layout)
            single = pk.raw_fp8_tree_triton(k, scale, layout, single_program=True)
            # same pairwise moment tree; only the D-axis reduction order differs
            rel = (chunked - single).abs() / single.abs().clamp_min(1.0)
            self.assertLess(float(rel.max()), 1e-12, f"tree mismatch at N={n}")

    def test_incremental_tree_is_bitwise_identical(self):
        """Extending the previous chunk's tree == rebuilding it from scratch."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")
        from sglang.srt.layers.attention.nsa.adaptive_hisa import partition_kernels as pk
        from sglang.srt.layers.attention.nsa.adaptive_hisa.partition_gpu import TreeLayout
        from sglang.srt.layers.attention.nsa.adaptive_hisa.pkey_fp8_builder import (
            build_key_tree_from_fp8,
        )

        device = torch.device("cuda")
        n_max = ROOT * 24
        k, scale = _keys(n_max, seed=7)
        k, scale = k.to(device), scale.to(device)
        for atom, root in ((ATOM, ROOT), (1, ROOT), (ATOM * 4, ROOT)):
            cached, n_old = None, 0
            # chunked prefill: the sealed prefix grows by a few roots per chunk
            for n in (root, root * 3, root * 4, root * 9, n_max):
                layout = TreeLayout(atom, root, n)
                full = pk.raw_fp8_tree_triton(k[:n], scale[:n], layout)
                if cached is not None:
                    inc = pk.raw_fp8_tree_triton_incremental(k[:n], scale[:n], layout, cached, n_old)
                    self.assertTrue(torch.equal(inc, full), f"atom={atom} root={root} N={n}")
                    via_builder, _ = build_key_tree_from_fp8(
                        k[:n], scale[:n], n, atom=atom, root=root, tree_cache=(cached, n_old)
                    )
                    self.assertTrue(torch.equal(via_builder, full))
                cached, n_old = full, n
            # not a strict prefix / misaligned cache -> silently rebuilds in full
            layout = TreeLayout(atom, root, n_max)
            self.assertTrue(torch.equal(
                pk.raw_fp8_tree_triton_incremental(k, scale, layout, cached, n_max), cached))
            half = pk.raw_fp8_tree_triton(k[: n_max // 2], scale[: n_max // 2], TreeLayout(atom, root, n_max // 2))
            self.assertTrue(torch.equal(
                pk.raw_fp8_tree_triton_incremental(k, scale, layout, half, n_max // 2 - 1), cached))

    def test_tree_cache_epoch_and_final(self):
        """The per-request tree cache only serves the same epoch and strict prefixes."""
        from sglang.srt.layers.attention.nsa.adaptive_hisa.prefill_runtime import _TreeCache

        cache = _TreeCache()
        flat = torch.zeros(3)
        cache.put(4, 1, epoch=2, n_complete=512, flat=flat)
        self.assertIsNone(cache.get(4, 1, 3, 1024))  # request slot reused
        self.assertIsNone(cache.get(4, 1, 2, 512))  # same prefix: nothing to extend
        self.assertIsNone(cache.get(5, 1, 2, 1024))  # other layer
        hit = cache.get(4, 1, 2, 1024)
        self.assertIsNotNone(hit)
        self.assertIs(hit[0], flat)
        self.assertEqual(hit[1], 512)
        cache.drop(4, 1)
        self.assertIsNone(cache.get(4, 1, 2, 1024))
        cache.put(4, 1, epoch=2, n_complete=512, flat=flat)
        cache.release(1)
        self.assertIsNone(cache.get(4, 1, 2, 1024))

    def test_radix_selection_matches_sort_path(self):
        """CUDA k-th selection (merge thresholds, repair pops) is bit-identical to sorting."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")
        from sglang.srt.layers.attention.nsa.adaptive_hisa import partition_select as sel
        from sglang.srt.layers.attention.nsa.adaptive_hisa.partition_gpu import kth_cost_threshold
        from sglang.srt.layers.attention.nsa.adaptive_hisa.pkey_fp8_builder import build_partition_from_fp8

        if not sel.available(torch.empty(1, device="cuda")):
            self.skipTest(f"radix_select op unavailable: {sel._STATE.get('error')}")
        device = torch.device("cuda")
        gen = torch.Generator(device="cpu").manual_seed(3)
        for n_cost, k in ((5, 1), (4095, 700), (40000, 12345), (40000, 0), (40000, 50000)):
            cost = torch.rand(n_cost, generator=gen, dtype=torch.float64).to(device) * 10
            cost[torch.rand(n_cost, generator=gen).to(device) < 0.3] = float("inf")
            cost = cost.round(decimals=1)  # duplicates at the boundary
            kt = torch.tensor(k, dtype=torch.int64, device=device)
            got = sel.kth_threshold_f64(cost, kt).reshape(())
            sel._STATE["loaded"] = False
            try:
                ref = kth_cost_threshold(cost, kt)
            finally:
                sel._STATE["loaded"] = True
            self.assertTrue(torch.equal(got, ref), (n_cost, k, got.item(), ref.item()))
        _set_env(
            dict(
                MODE="build_only",
                PARTITION_METRIC="key_sse",
                SPLIT_BACKEND="gpu",
                MERGE_POLICY="sync_nonoverlap",
                MERGE_TARGET_DIVISOR="64",
            )
        )
        cfg = get_config()
        for n in (ROOT * 4, 4096, 20000):
            k, scale = _keys(n, seed=n + 7)
            k, scale = k.to(device), scale.to(device)
            fast = build_partition_from_fp8(k, scale, n, cfg)
            sel._STATE["loaded"] = False
            try:
                slow = build_partition_from_fp8(k, scale, n, cfg)
            finally:
                sel._STATE["loaded"] = True
            self.assertEqual(int(fast.num_leaves.item()), int(slow.num_leaves.item()))
            self.assertTrue(torch.equal(fast.leaf_start, slow.leaf_start), f"start mismatch at N={n}")
            self.assertTrue(torch.equal(fast.leaf_len, slow.leaf_len), f"len mismatch at N={n}")
            self.assertEqual([int(r) for r in fast.round_merges], [int(r) for r in slow.round_merges])


@unittest.skip("calibration scorer removed with P-qfull")
class TestCalibrationContract(CustomTestCase):
    def test_weights_may_include_a_trailing_one(self):
        q, w = _queries(2, heads=4, dim=8, seed=7)
        k, scale = _keys(8, dim=8)
        plain = calibration_scores(q, w, k, scale, 8)
        squeezed = calibration_scores(q, w[:, :, None], k, scale, 8)
        self.assertTrue(torch.allclose(plain, squeezed))


if __name__ == "__main__":
    unittest.main()

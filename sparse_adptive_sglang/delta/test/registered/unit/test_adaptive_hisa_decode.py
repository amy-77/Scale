"""Decode contracts: candidate coverage, paged gathers, exact reranking, graph rebinding."""
import dataclasses
import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.attention.nsa.adaptive_hisa.config import PartitionConfig, PartitionConfigError


class TestDecodeConfig(unittest.TestCase):
    def test_opt_in_and_dependencies(self):
        self.assertEqual(PartitionConfig().validate().mode, 'off')
        cfg = PartitionConfig(mode='adaptive_decode').validate()
        self.assertEqual((cfg.fallback_layers, cfg.sink_tokens, cfg.tail_tokens, cfg.decode_chunk),
                         (0, 64, 256, 64))
        for overrides in [dict(build_summaries=False), dict(fallback_layers=-1),
                          dict(candidate_tokens=2047), dict(index_topk=1024),
                          dict(tail_tokens=8192 - 2048), dict(decode_chunk=12), dict(sink_tokens=-1)]:
            with self.assertRaises(PartitionConfigError):
                dataclasses.replace(PartitionConfig(mode='adaptive_decode'), **overrides).validate()


@unittest.skipUnless(torch.cuda.is_available(), 'requires CUDA and DeepGEMM')
class TestDecodeGPU(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(712)

    def test_expansion_whole_leaves_tail_and_padding(self):
        from sglang.srt.layers.attention.nsa.adaptive_hisa.decode_kernels import (
            expand_candidates,
            expand_ranked_candidates,
        )
        lengths = [1, 7, 1024, 17, 4096, 66, 4096, 10]
        starts = [sum(lengths[:i]) for i in range(len(lengths))]
        complete = sum(lengths)
        order = [4, 0, 2, 6, 1, 3, 7, 5] + list(range(8,64))
        gpu_starts = torch.tensor(starts + [complete]*56, device='cuda', dtype=torch.int32)
        gpu_lengths = torch.tensor(
            lengths + [0] * 56, device='cuda', dtype=torch.int32
        )
        gpu_order = torch.tensor(order, device='cuda')
        for tail in [0, 1, 127, 2048, 8191, 8192]:
            expected=[]; cumulative=0; selected=[]
            for rank in order:
                size = lengths[rank] if rank < 8 else 0
                cumulative += size
                take = size if cumulative <= 8192-tail else 0
                selected.append(take)
                if take: expected.extend(range(starts[rank], starts[rank]+size))
            expected.extend(range(complete,complete+tail))
            prefix=torch.tensor(selected,device='cuda',dtype=torch.int32).cumsum(0,dtype=torch.int32)
            candidates,count=expand_candidates(gpu_order,prefix,gpu_starts,
                torch.tensor([complete],device='cuda',dtype=torch.int32),
                torch.tensor([complete+tail],device='cuda',dtype=torch.int32),8192)
            ranked_prefix = gpu_lengths[gpu_order].cumsum(0, dtype=torch.int32)
            fused_candidates, fused_count = expand_ranked_candidates(
                gpu_order,
                ranked_prefix,
                gpu_starts,
                torch.tensor([complete],device='cuda',dtype=torch.int32),
                torch.tensor([complete+tail],device='cuda',dtype=torch.int32),
                8192,
            )
            actual=candidates[0,:count.item()].tolist()
            self.assertEqual(actual,expected)
            self.assertTrue(torch.equal(fused_candidates, candidates))
            self.assertTrue(torch.equal(fused_count, count))
            self.assertEqual(len(actual),len(set(actual)))
            self.assertTrue(bool((candidates[0,len(actual):]==-1).all()))

    def test_weighted_selector_matches_stable_prefix_set(self):
        from sglang.srt.layers.attention.nsa.adaptive_hisa.decode_select import (
            expand_selected_candidates,
            selector_profile_snapshot,
            weighted_select_candidates,
            weighted_select_candidates_legacy,
            weighted_prefix_profile,
        )

        def reference(scores, starts, lengths, seq, budget, sink, tail):
            # h20-1 accuracy_runtime.select_candidates, expressed on the host
            # (guards, clipping, exact fill with a partial crossing leaf),
            # re-emitted in logical order.
            n = len(lengths)
            sink_end = min(sink, seq)
            tail_start = max(sink_end, seq - tail)
            room = budget - sink_end - (seq - tail_start)
            order = sorted(range(n), key=lambda i: (-scores[i], i))
            taken = [0] * n
            for leaf in order:
                first = min(max(starts[leaf], sink_end), tail_start)
                last = min(max(starts[leaf] + lengths[leaf], sink_end), tail_start)
                clipped = max(last - first, 0)
                if room <= 0 or clipped <= 0:
                    if room <= 0:
                        break
                    continue
                take = min(clipped, room)
                taken[leaf] = (first, take)
                room -= take
            expected = list(range(sink_end))
            for leaf in range(n):
                if taken[leaf]:
                    first, take = taken[leaf]
                    expected.extend(range(first, first + take))
            expected.extend(range(tail_start, seq))
            return expected

        def check(scores, lengths, unsealed=0, budget=8192, sink=64, tail=256):
            n = len(lengths)
            capacity = math.ceil(n / 64) * 64
            starts = [sum(lengths[:i]) for i in range(n)]
            complete = sum(lengths)
            seq = complete + unsealed
            expected = reference(scores, starts, lengths, seq, budget, sink, tail)
            self.assertEqual(len(expected), min(budget, seq))
            self.assertEqual(len(set(expected)), len(expected))

            gpu_scores = torch.tensor(
                scores + [float('-inf')] * (capacity - n),
                dtype=torch.float32,
                device='cuda',
            )
            gpu_lengths = torch.tensor(
                lengths + [0] * (capacity - n),
                dtype=torch.int32,
                device='cuda',
            )
            gpu_starts = torch.tensor(
                starts + [complete] * (capacity - n),
                dtype=torch.int32,
                device='cuda',
            )
            candidates, count = weighted_select_candidates(
                gpu_scores,
                gpu_lengths,
                gpu_starts,
                torch.tensor([[n]], dtype=torch.int32, device='cuda'),
                torch.tensor([seq], dtype=torch.int32, device='cuda'),
                torch.empty(capacity, dtype=torch.int32, device='cuda'),
                budget,
                sink,
                tail,
            )
            actual_count = int(count.item())
            self.assertEqual(actual_count, len(expected))
            self.assertEqual(
                candidates[0, :actual_count].tolist(), expected
            )
            self.assertTrue(
                bool((candidates[0, actual_count:] == -1).all())
            )
            if capacity <= 4096:
                legacy_candidates, legacy_count = weighted_select_candidates_legacy(
                    gpu_scores,
                    gpu_lengths,
                    gpu_starts,
                    torch.tensor([[n]], dtype=torch.int32, device='cuda'),
                    torch.tensor([seq], dtype=torch.int32, device='cuda'),
                    torch.empty(capacity, dtype=torch.int32, device='cuda'),
                    budget,
                    sink,
                    tail,
                )
                self.assertEqual(int(legacy_count.item()), actual_count)
                self.assertTrue(torch.equal(
                    legacy_candidates[:, :actual_count],
                    candidates[:, :actual_count],
                ))
                split_prefix = torch.empty(
                    capacity, dtype=torch.int32, device='cuda'
                )
                weighted_prefix_profile(
                    gpu_scores,
                    gpu_lengths,
                    gpu_starts,
                    torch.tensor([[n]], dtype=torch.int32, device='cuda'),
                    torch.tensor([seq], dtype=torch.int32, device='cuda'),
                    split_prefix,
                    budget,
                    sink,
                    tail,
                )
                split_candidates = torch.empty_like(candidates)
                split_count = torch.empty_like(count)
                expand_selected_candidates(
                    split_prefix,
                    gpu_starts,
                    gpu_lengths,
                    torch.tensor([[n]], dtype=torch.int32, device='cuda'),
                    torch.tensor([seq], dtype=torch.int32, device='cuda'),
                    budget,
                    sink,
                    tail,
                    candidates_out=split_candidates,
                    count_out=split_count,
                )
                self.assertEqual(int(split_count.item()), actual_count)
                self.assertTrue(torch.equal(
                    split_candidates,
                    candidates,
                ))

        # Equal values, infinities, and signed zero exercise stable index ties;
        # small guards keep every leaf partially visible.
        check(
            [
                float('inf'),
                float('inf'),
                1.0,
                0.0,
                -0.0,
                -1.0,
                float('-inf'),
            ],
            [7, 11, 17, 19, 23, 29, 31],
            unsealed=5,
            budget=128,
            sink=4,
            tail=9,
        )
        # The crossing leaf is taken partially (exact fill); lower-ranked
        # leaves are rejected even if they would fit individually.
        check([4.0, 3.0, 2.0, 1.0], [5000, 4000, 1, 1])
        # The highest-ranked leaf alone exceeds the room after the guards.
        check([3.0, 2.0, 1.0], [9000, 7, 9], unsealed=3)
        # Leaves overlapping the sink / tail are clipped, not dropped.
        check([1.0, 5.0, 2.0, 4.0], [100, 4000, 4000, 300], unsealed=100)
        # Short contexts: everything is selected (guards may cover it all).
        check([1.0, 2.0], [100, 100], unsealed=50)
        check([1.0], [8000], unsealed=0)
        # All-length-one needs thousands of selected leaves and disproves a
        # fixed Top-2048/4096 shortcut.
        check([float(i) for i in range(128)], [64] * 128, unsealed=1)
        # Worst-case fast-path pack: every production-capacity leaf lands in
        # the same high-byte bucket and ties are cut by logical leaf index.
        check([1.0] * 4096, [3] * 4096)
        # Above the shared-memory cap dispatches to the exact legacy fallback.
        check([float(i % 13) for i in range(9000)], [1] * 9000, unsealed=7)
        for seed in range(5):
            generator = torch.Generator().manual_seed(1900 + seed)
            lengths = torch.randint(
                1, 97, (257,), generator=generator
            ).tolist()
            # Quantization deliberately creates many score ties.
            scores = (
                torch.randint(-8, 9, (257,), generator=generator).float()
                / 4
            ).tolist()
            check(scores, lengths, unsealed=(seed * 37) % 8)
        profile = selector_profile_snapshot(reset=True)
        self.assertGreater(profile["fast_calls"], 0)
        self.assertIsNotNone(profile["p50_bucket_ratio_pct"])

    def test_raw_gather_noncontiguous_physical_pages(self):
        from sglang.srt.layers.attention.nsa.adaptive_hisa.decode_kernels import gather_candidate_keys
        n=16384
        bits=torch.randint(0,256,(n,128),device='cuda',dtype=torch.uint8)
        scales=torch.rand(n,device='cuda')
        logical=torch.cat((bits.reshape(-1,8192),scales.reshape(-1,64).view(torch.uint8)),1)
        permutation=torch.randperm(n//64,device='cuda')
        raw=torch.empty_like(logical); raw[permutation]=logical
        ids=torch.randperm(n,device='cuda')[:8192].int().reshape(1,-1); ids[0,-3:]=-1
        keys,got_scale=gather_candidate_keys(raw,permutation.int().reshape(1,-1),ids,
            torch.tensor([n],device='cuda',dtype=torch.int32))
        self.assertTrue(torch.equal(keys[:-3].view(torch.uint8),bits[ids[0,:-3].long()]))
        self.assertTrue(torch.equal(got_scale[:-3],scales[ids[0,:-3].long()]))
        self.assertTrue(bool((keys[-3:].view(torch.uint8)==0).all()))
        self.assertTrue(bool((got_scale[-3:]==0).all()))

    def fixture(self, n=8192, epoch=0):
        from sglang.srt.layers.attention.nsa.adaptive_hisa.decode_runtime import DecodeWorkspace
        from sglang.srt.layers.attention.nsa.adaptive_hisa.summary_pool import SummaryPool, build_summaries
        from sglang.srt.layers.attention.nsa.triton_kernel import act_quant
        lengths=[]; remain=n
        while remain:
            size=min(remain,int(torch.randint(1,129,()).item()))
            lengths.append(size); remain-=size
        count=len(lengths); cap=math.ceil(count/64)*64
        lens=torch.tensor(lengths+[0]*(cap-count),device='cuda',dtype=torch.int32)
        starts=lens.cumsum(0,dtype=torch.int32)-lens
        part=SimpleNamespace(capacity=cap,n_tokens=n,n_complete=n,leaf_start=starts,leaf_len=lens,
            num_leaves=torch.tensor(count,device='cuda'),status_ok=torch.tensor(True,device='cuda'))
        pool=SummaryPool(1,math.ceil(cap/64)+2,'cuda'); entry=pool.allocate(0,1,epoch,part)
        keys,scales=act_quant(torch.randn(n,128,device='cuda',dtype=torch.bfloat16),128,'ue8m0')
        scales=scales.flatten()
        build_summaries(pool,entry,keys,scales,scale_fmt='ue8m0')
        raw=torch.cat((keys.view(torch.uint8).reshape(-1,8192),scales.reshape(-1,64).view(torch.uint8)),1)
        pages=torch.arange(n//64,device='cuda',dtype=torch.int32).reshape(1,-1)
        work=DecodeWorkspace(2048,'cuda'); work.bind(entry)
        q=torch.randn(1,64,128,device='cuda').to(torch.float8_e4m3fn)
        weights=torch.rand(1,64,device='cuda')
        seq=torch.tensor([n],device='cuda',dtype=torch.int32)
        return work,pool,entry,raw,pages,q,weights,seq,keys,scales

    def test_full_coverage_exact_scores_and_topk(self):
        import deep_gemm
        from sglang.srt.layers.attention.nsa.adaptive_hisa.decode_runtime import select_topk
        work,pool,entry,raw,pages,q,w,seq,k,s=self.fixture()
        cfg=PartitionConfig(mode='adaptive_decode').validate()
        out,candidates,count,_,scores=select_topk(work,pool.bufs[0],raw,pages,seq,q,w,cfg,return_debug=True)
        self.assertEqual(count.item(),8192)
        self.assertEqual(len(set(candidates.flatten().tolist())),8192)
        reference=deep_gemm.fp8_mqa_logits(q,(k,s),w,work.zero,seq,clean_logits=False)
        torch.testing.assert_close(scores[:,:8192],reference.gather(1,candidates.long()),rtol=1e-5,atol=1e-4)
        # Selection is exact for the HISA sparse scorer; compare its actual
        # scores to avoid treating cross-kernel FP32 ties as membership errors.
        mapped=torch.empty_like(reference).scatter_(1,candidates.long(),scores[:,:8192])
        threshold=mapped.topk(2048,dim=1).values[:,-1:]
        self.assertTrue(bool((mapped.gather(1,out.long())>=threshold).all()))
        self.assertEqual(out.dtype,torch.int32)
        self.assertEqual(len(set(out.flatten().tolist())),2048)

    def test_graph_replay_after_workspace_rebind_and_new_tail(self):
        from sglang.srt.layers.attention.nsa.adaptive_hisa.decode_runtime import select_topk
        work,pool,entry,raw,pages,q,w,seq,k,s=self.fixture(16384)
        cfg=PartitionConfig(mode='adaptive_decode').validate()
        for _ in range(3): select_topk(work,pool.bufs[0],raw,pages,seq,q,w,cfg)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual=select_topk(work,pool.bufs[0],raw,pages,seq,q,w,cfg,return_debug=True)
        graph.replay()
        expected=select_topk(work,pool.bufs[0],raw,pages,seq,q,w,cfg,return_debug=True)
        self.assert_same_ranking(actual,expected)
        # Freeze only the shorter prefix; the remaining tokens become mandatory
        # raw tail. Bind a new epoch to the SAME workspace addresses.
        entry.epoch+=1
        entry.n_tokens=entry.n_complete=16128
        part=entry.partition
        lengths=part.leaf_len.clone()
        lengths=torch.minimum(lengths,(16128-part.leaf_start).clamp_min(0))
        part.leaf_len=lengths
        part.num_leaves=(lengths>0).sum()
        work.bind(entry)
        graph.replay()
        expected=select_topk(work,pool.bufs[0],raw,pages,seq,q,w,cfg,return_debug=True)
        self.assert_same_ranking(actual,expected)
        _,candidates,count,_,_=expected
        chosen=candidates[0,:count.item()].tolist()
        self.assertEqual(count.item(),8192)
        self.assertEqual(len(set(chosen)),8192)
        # Sink and the most recent 256 tokens are always raw candidates; the
        # 256 generated tokens were sealed into 4 fixed-64 leaves (clipped
        # away by the tail guard until they age out of it).
        self.assertTrue(set(range(64)).issubset(chosen))
        self.assertTrue(set(range(16128,16384)).issubset(chosen))
        self.assertEqual(work.previous_blocks.item(),4)
        self.assertEqual(work.num_leaves.item(),work.n_base_leaves+4)
        self.assertTrue(work.eligible(16384,cfg))
        self.assertTrue(work.eligible(16128+8192,cfg))
        self.assertFalse(work.eligible(100,cfg))
        self.assertFalse(work.eligible(16128+cfg.decode_chunk*work.capacity,cfg))

    def assert_same_ranking(self, actual, expected):
        # fast_topk is unordered and equal-score boundary membership can vary.
        # Compare the entire candidate/scoring pipeline exactly, then verify
        # each returned set satisfies the same Top-2048 score threshold.
        a,ac,an,aco,als=actual; e,ec,en,eco,els=expected
        self.assertTrue(torch.equal(ac,ec), 'graph candidate table differs')
        self.assertTrue(torch.equal(an,en))
        self.assertTrue(torch.equal(aco,eco), 'graph coarse scores differ')
        n=int(en.item())
        self.assertTrue(torch.equal(als[:,:n],els[:,:n]), 'graph fine scores differ')
        token_scores=dict(zip(ec[0,:n].tolist(),els[0,:n].tolist()))
        threshold=float(els[0,:n].topk(2048).values[-1])
        for picks in (a,e):
            values=picks.flatten().tolist()
            self.assertEqual(len(set(values)),2048)
            self.assertTrue(all(token_scores[x]>=threshold for x in values))

    def test_graph_admission_epoch_release_tail_budget_and_batch_fallback(self):
        from sglang.srt.layers.attention.nsa.adaptive_hisa import decode_runtime as runtime
        work,pool,entry,*_=self.fixture(16384)
        cfg=PartitionConfig(mode='adaptive_decode',fallback_layers=0).validate()
        batch=SimpleNamespace(batch_size=1,seq_lens_cpu=torch.tensor([16385]),
            req_pool_indices=torch.tensor([1],device='cuda'),
            req_pool_indices_cpu=[1],
            token_to_kv_pool=SimpleNamespace(index_k_with_scale_buffer=[None]))
        entries={0:entry}
        runtime._WORKSPACES['unit_fixture']=work
        try:
            with patch.object(runtime,'get_config',return_value=cfg), \
                 patch.object(runtime,'_supported_batch',side_effect=lambda b:b.batch_size==1), \
                 patch.object(runtime,'_workspace',return_value=(pool,work)), \
                 patch('sglang.srt.layers.attention.nsa.adaptive_hisa.prefill_runtime.get_summary_entry',
                       side_effect=lambda layer,req:entries.get(layer)):
                # Uncaptured workspaces cannot authorize a different model graph.
                self.assertFalse(runtime.allow_cuda_graph(batch))
                work.graph_ready=True
                self.assertTrue(runtime.allow_cuda_graph(batch))
                # Reusing the ForwardBatch at a later step rechecks the tail.
                batch.seq_lens_cpu.fill_(16384+8192)
                self.assertTrue(runtime.allow_cuda_graph(batch))
                batch.seq_lens_cpu.fill_(16384+64*work.capacity)
                self.assertFalse(runtime.allow_cuda_graph(batch))
                batch.seq_lens_cpu.fill_(16385)
                self.assertTrue(runtime.allow_cuda_graph(batch))
                entries.clear()
                runtime.release_decode_request(1)
                self.assertFalse(work.valid)
                self.assertFalse(runtime.allow_cuda_graph(batch))
                # A reused request slot with a new epoch binds fresh metadata.
                entry.epoch+=1
                entries[0]=entry
                runtime.release_decode_request(1)
                self.assertTrue(runtime.allow_cuda_graph(batch))
                self.assertEqual(work.binding[1],entry.epoch)
                batch.batch_size=2
                self.assertFalse(runtime.allow_cuda_graph(batch))
        finally:
            runtime._WORKSPACES.pop('unit_fixture',None)

    def test_hisa_sparse_token_ids_match_original_fixed_blocks(self):
        from sglang.srt.layers.attention.nsa.hisa.triton_kernels import sparse_paged_mqa_triton
        _,_,_,raw,pages,q,w,seq,*_=self.fixture(16384)
        blocks=torch.randperm(256,device='cuda')[:128].int().reshape(1,1,-1)
        tokens=(blocks[:,:,:,None]*64+torch.arange(64,device='cuda')).reshape(1,1,-1).int()
        args=(q.unsqueeze(1),raw.view(-1,64,1,132))
        fixed=sparse_paged_mqa_triton(*args,blocks,64,w,seq,pages)
        adaptive=sparse_paged_mqa_triton(*args,tokens,1,w,seq,pages)
        self.assertTrue(torch.equal(fixed,adaptive))

    def test_hisa_fused_topk_logical_physical_and_padding(self):
        from sglang.srt.layers.attention.nsa.hisa.hisa_topk_fused import hisa_topk_candidates_fused
        scores=torch.randn(1,8192,device='cuda')
        candidates=torch.randperm(16384,device='cuda')[:8192].int().reshape(1,-1)
        pages=torch.randperm(16384,device='cuda').int().reshape(1,-1)
        seq=torch.tensor([16384],device='cuda',dtype=torch.int32)
        for n in [0,37,2048,8173]:
            count=torch.tensor([n],device='cuda',dtype=torch.int32)
            logical=hisa_topk_candidates_fused(scores,candidates,count,seq)
            physical=hisa_topk_candidates_fused(scores,candidates,count,seq,pages)
            expected=candidates[0,scores[0,:n].topk(min(n,2048)).indices]
            self.assertEqual(set(logical[logical>=0].tolist()),set(expected.tolist()))
            self.assertEqual(set(physical[physical>=0].tolist()),set(pages[0,expected.long()].tolist()))
            self.assertEqual(int((logical==-1).sum()),2048-min(n,2048))

    def test_incremental_summary_matches_hisa_completed_blocks_and_catches_up(self):
        from sglang.srt.layers.attention.nsa.adaptive_hisa.incremental import update_completed_blocks
        from sglang.srt.layers.attention.nsa.hisa.tilelang_kernels import fp8_native_paged_mean_pooling_completed_blocks_interface
        work,pool,entry,raw,pages,q,w,seq,*_=self.fixture(16384)
        # This test exercises the generated region after a frozen 8192-token prefix.
        work.base_complete.fill_(8192);work.complete.fill_(8192)
        work.base_rows.fill_(513);work.num_leaves.fill_(513)
        work.summary_pages.fill_(19)
        prefix=work.summary_pages[:8].clone()
        original=torch.zeros((4,8448),device='cuda',dtype=torch.uint8)
        request_tokens=torch.arange(16384,device='cuda',dtype=torch.int32).reshape(1,-1)
        request_ids=torch.zeros(1,device='cuda',dtype=torch.int64)
        pool_pages=torch.arange(4,device='cuda',dtype=torch.int32).reshape(1,-1)
        previous=torch.tensor([8192],device='cuda',dtype=torch.int32)
        # chunk=64 reproduces the HISA completed-block kernel bit for bit.
        for n in [8193,8255,8256,8267,16384]:
            seq.fill_(n)
            update_completed_blocks(work,raw,pages,seq,64)
            completed=(n-8192)//64
            self.assertEqual(work.previous_blocks.item(),completed)
            self.assertEqual(work.num_leaves.item(),513+completed)
            self.assertEqual(work.complete.item(),8192+64*completed)
            self.assertTrue(torch.equal(prefix,work.summary_pages[:8]))
        fp8_native_paged_mean_pooling_completed_blocks_interface(
            raw,request_tokens,pool_pages,request_ids,previous,seq,original,
            64,64,64,128)
        for i in range(128):
            row=513+i;p,r=divmod(row,64);op,orr=divmod(128+i,64)
            self.assertTrue(torch.equal(work.summary_pages[p,r*128:(r+1)*128],
                                        original[op,orr*128:(orr+1)*128]))
            torch.testing.assert_close(work.summary_pages[p].view(torch.float32)[2048+r],
                                       original[op].view(torch.float32)[2048+orr],rtol=1e-6,atol=1e-9)
            self.assertEqual(work.starts[row].item(),8192+64*i)
            self.assertEqual(work.lengths[row].item(),64)

    def test_incremental_fixed8_chunks_match_dequantized_mean(self):
        from sglang.srt.layers.attention.nsa.adaptive_hisa.incremental import update_completed_blocks
        work,pool,entry,raw,pages,q,w,seq,keys,scales=self.fixture(16384)
        work.base_complete.fill_(8192);work.complete.fill_(8192)
        work.base_rows.fill_(513);work.num_leaves.fill_(513)
        work.summary_pages.fill_(19)
        prefix=work.summary_pages[:8].clone()
        for n in [8193,8199,8200,8207,8264,12345,16384]:
            seq.fill_(n)
            update_completed_blocks(work,raw,pages,seq,8)
            completed=(n-8192)//8
            self.assertEqual(work.previous_blocks.item(),completed)
            self.assertEqual(work.num_leaves.item(),513+completed)
            self.assertEqual(work.complete.item(),8192+8*completed)
            self.assertTrue(torch.equal(prefix,work.summary_pages[:8]))
        dequant=keys.float()*scales[:,None]
        means=dequant[8192:].reshape(1024,8,128).mean(1)
        rows=torch.arange(513,513+1024,device='cuda')
        page,slot=rows//64,rows%64
        got_scale=work.summary_pages.view(torch.float32)[page,2048+slot]
        got=work.summary_pages[page[:,None],slot[:,None]*128+torch.arange(128,device='cuda')].view(torch.float8_e4m3fn).float()*got_scale[:,None]
        ref_scale=(means.abs().amax(1)/448.0).clamp_min(1e-10)
        torch.testing.assert_close(got_scale,ref_scale,rtol=1e-5,atol=0)
        # FP8 e4m3 rounding: half-ulp relative, subnormal floor 2^-9 * scale.
        torch.testing.assert_close(got,means,rtol=2**-4+1e-6,atol=float(ref_scale.max())*2**-9)
        self.assertTrue(torch.equal(work.starts[513:513+1024],
                                    torch.arange(8192,16384,8,device='cuda',dtype=torch.int32)))
        self.assertTrue(bool((work.lengths[513:513+1024]==8).all()))


if __name__=='__main__':
    unittest.main()

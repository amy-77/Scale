"""GPU reference checks for K=4 and regressions at K=8/16 before evaluation."""
import json
import sys
from pathlib import Path

import argparse
import importlib.metadata

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--hisa-source', type=Path, default=Path(__file__).resolve().parents[2] / 'HISA_current')
parser.add_argument('--output', type=Path, default=Path('results/hisa_fine_kernel_validation.json'))
args = parser.parse_args()
import torch

sys.path.insert(0, str(args.hisa_source.resolve() / 'python'))
from sglang.srt.layers.attention.nsa.hisa.triton_kernels import (
    block_mean_pooling_triton, block_sparse_mqa_triton, sparse_paged_mqa_triton,
)
from sglang.srt.layers.attention.nsa.hisa.tilelang_kernels import fp8_native_block_mean_pooling_grouped_interface
from sglang.srt.layers.attention.nsa.hisa.fast_topk_runtime import fast_topk_runtime

torch.manual_seed(20261107)
device = 'cuda'
checks = []
for block in [4, 8, 16]:
    for n in [7, 63, 64, 65, 257]:
        keys = torch.randn(n, 128, device=device).to(torch.float8_e4m3fn)
        scales = torch.rand(n, device=device) * 0.1 + 0.01
        raw = keys.float() * scales[:, None]
        means = torch.stack([raw[i:i+block].mean(0) for i in range(0, n, block)])
        expected_scale = (means.abs().amax(1) / 448.0).clamp_min(1e-10)
        expected_q = (means / expected_scale[:, None]).to(torch.float8_e4m3fn)
        for name, function in [('triton', block_mean_pooling_triton), ('tilelang', fp8_native_block_mean_pooling_grouped_interface)]:
            pooled, pooled_scale = function(keys, scales, block)
            torch.testing.assert_close(pooled_scale, expected_scale, rtol=2e-5, atol=1e-8)
            # Quantization at a rounding boundary may differ by one FP8 ULP.
            error = ((pooled.float() - expected_q.float()).abs() * expected_scale[:, None]).amax().item()
            assert error <= 0.13 * means.abs().amax().item() + 1e-6, (block, n, name, error)
            checks.append({'check': 'pool', 'backend': name, 'block': block, 'n': n, 'max_quantized_error': error})

    n, heads, dim, count = 257, 64, 128, 19
    keys = torch.randn(n, dim, device=device).to(torch.float8_e4m3fn)
    scales = torch.rand(n, device=device) * 0.1 + 0.01
    queries = torch.randn(1, heads, dim, device=device).to(torch.float8_e4m3fn)
    weights = torch.rand(1, heads, device=device) / heads
    selected = torch.randint(0, (n + block - 1) // block, (1, count), device=device, dtype=torch.int32)
    selected[0, -1] = n // block
    ks = torch.zeros(1, device=device, dtype=torch.int32)
    ke = torch.tensor([n], device=device, dtype=torch.int32)
    positions = (selected[:, :, None] * block + torch.arange(block, device=device)).reshape(1, -1)
    raw = keys.float() * scales[:, None]
    reference = (torch.einsum('mhd,mtd->mht', queries.float(), raw[positions.clamp_max(n-1)])
                 .relu() * weights[:, :, None]).sum(1)
    reference[positions >= n] = float('-inf')
    ragged = block_sparse_mqa_triton(queries, keys, scales, selected, block, weights, ks, ke)
    torch.testing.assert_close(ragged, reference, rtol=0.015, atol=0.002)

    pages = (n + 63) // 64
    mapping = torch.randperm(pages, device=device, dtype=torch.int32)
    packed = torch.zeros(pages, 64 * (dim + 4), dtype=torch.uint8, device=device)
    for logical in range(pages):
        physical = int(mapping[logical])
        start, end = logical * 64, min((logical+1) * 64, n)
        length = end - start
        packed[physical, :length*dim] = keys[start:end].view(torch.uint8).reshape(-1)
        packed[physical, 64*dim:64*dim+length*4] = scales[start:end].view(torch.uint8)
    paged = sparse_paged_mqa_triton(
        queries.unsqueeze(1), packed.view(pages, 64, 1, dim+4), selected.unsqueeze(1),
        block, weights.unsqueeze(1), ke, mapping.unsqueeze(0),
    ).squeeze(1)
    torch.testing.assert_close(paged, reference, rtol=0.015, atol=0.002)
    checks.append({'check': 'sparse_logits_against_torch', 'block': block, 'ragged_and_permuted_pages': True,
                   'partial_block': True, 'nonmultiple_topk': count})

scores = torch.randn(4, 32768, device=device)
for k in [512, 1024, 2048]:
    indices = fast_topk_runtime(scores, k).long()
    got = scores.gather(1, indices).sort(1).values
    expected = scores.topk(k, dim=1).values.sort(1).values
    torch.testing.assert_close(got, expected, rtol=0, atol=0)
    checks.append({'check': 'topk_exact_against_torch', 'k': k})
torch.cuda.synchronize()
output = {'passed': True, 'gpu': torch.cuda.get_device_name(), 'checks': checks}
output['environment'] = {name: importlib.metadata.version(name) for name in ['torch', 'tilelang', 'apache-tvm-ffi', 'triton']}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(output, indent=2)+'\n')
print(json.dumps({'passed': True, 'checks': len(checks)}), flush=True)

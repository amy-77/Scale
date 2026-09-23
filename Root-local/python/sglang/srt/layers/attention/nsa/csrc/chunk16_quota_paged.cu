// Chunk16 配额 Top-K：面向 SGLang NSA Indexer cache 的分页版本。
//
// 选择算法与 headwise_minmax/csrc/chunk16_quota.cu 相同，但 K 不通过连续
// [nk, 128] buffer 读取，而是经 request 的 block table 寻址。一个 chunk
// （16 tokens）不会跨越 page（64 tokens）边界，因此 stage 2 每个 chunk
// 只需一次 block-table 查找。
//
// 流程（每个 (query, head) 行独立；位置均为 request-relative）：
//   stage 0（调用方、cuBLAS）：
//           coarse_scores = q_bf16 @ chunk_sum_bf16.T，覆盖该 request 的
//           逻辑 chunks。若走 mapped / MISA prune 路径，则只对
//           chunk_ids 给出的逻辑 chunks 打分（见 selected_chunk_scores_paged）。
//   stage 1（本 kernel）：
//           对 SMEM 中的 BF16 粗分做精确两级（高 8 位 + 低 8 位）radix
//           select，选出 Top-SEL(128) 可见 chunks；强制保留 sink chunk(0)
//           与尾部（当前 token）chunk。
//   stage 2（本 kernel）：
//           仅对选中 chunks 做精确 FP8 QK。NDENSE(52) 个稠密 chunks 各保留
//           QDENSE(8) tokens，其余 76 个稀疏 chunks 各保留 QSPARSE(4)：
//           52*8 + 76*4 = 720 indices，不足填 -1。sink 与当前 token 用
//           +inf 强制保留。chunk 内 Top-K 用 16-lane bitonic sort。
//           kernel 只输出索引。
//
// 每个 CTA（4 warps）处理一行 (query, head)。
// NTHREAD 必须 >= SEL（并行排名计数时每个线程最多对应一个候选项）。
//
// 额外接口：
//   chunk16_quota_paged         —— 对全部逻辑 chunks 做上述流程；
//   chunk16_quota_paged_mapped  —— scores/keys 的列索引经 chunk_ids 映射到
//                                  逻辑 chunk id（-1 表示屏蔽）；用于 MISA
//                                  粗块剪枝后只扫描保留区域；
//   selected_chunk_scores_paged —— 只对 mapped 的逻辑 chunks 计算
//                                  q · chunk_sum 粗分（替代全量 cuBLAS）。
//
// 例外：chunk16_quota_paged_mapped_mean 是 mean-only 实验契约。其 coarse
// region 已由调用方纯 mean 排名，因此 stage 1/2 均不额外 pin sink/tail。
// 普通 mapped/fraction 路径仍使用默认 pin_boundaries=true。

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#define CHUNK 16
#define SEL 128
#define NDENSE 52
#define QDENSE 8
#define QSPARSE 4
#define OUTK 720
#define NWARP 4
#define NTHREAD (NWARP * 32)
#define DIM 128
#define PAGE 64
// NSA indexer cache page layout: 64 * (128 fp8 + 4 scale bytes)
#define PAGE_BYTES (PAGE * (DIM + 4))

// order-preserving u16 keys reserved by the kernel: 0 = masked sentinel,
// 0xFFFF / 0xFFFE = pinned sink / tail chunk (both decode from NaN bit
// patterns, which GEMM scores never produce)
#define KEY_SINK 0xFFFFu
#define KEY_TAIL 0xFFFEu

__device__ __forceinline__ unsigned key_of(unsigned short bits) {
    return (bits & 0x8000u) ? (unsigned short)~bits : (bits | 0x8000u);
}

__global__ void __launch_bounds__(NTHREAD)
chunk16_quota_paged_kernel(
    const __nv_bfloat16* __restrict__ scores,  // [M, C] coarse chunk scores
    const __nv_bfloat16* __restrict__ q,       // [M, 128]
    const unsigned char* __restrict__ kv,      // [P, PAGE_BYTES] paged cache
    const int* __restrict__ bt,                // [B, bt_stride] block tables
    const int* __restrict__ batch,             // [M] request index per row
    const int* __restrict__ ke_row,            // [M] visible token count
    const int* __restrict__ chunk_ids,          // optional [M, C] logical chunks
    int* __restrict__ out,                     // [M, OUTK] pre-filled -1
    const int C,
    const int bt_stride,
    const bool pin_boundaries
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp = tid / 32;
    const int lane = tid % 32;

    extern __shared__ unsigned short keys[];   // [C] score keys
    __shared__ float q_s[DIM];
    __shared__ int hist[NWARP][256];
    __shared__ int fi[SEL];                    // selected chunk ids, desc rank
    __shared__ unsigned short fk[SEL];
    __shared__ int n_top, n_eq, sel_n;
    __shared__ int bin_hi, cnt_gt_hi, thr_key, need_eq;

    const int ke_r = ke_row[row];
    const int c_hi = (ke_r + CHUNK - 1) / CHUNK;
    const int* bt_row = bt + (int64_t)batch[row] * bt_stride;

    if (tid < DIM)
        q_s[tid] = __bfloat162float(q[(int64_t)row * DIM + tid]);
    for (int b = tid; b < 256; b += NTHREAD) {
        #pragma unroll
        for (int w = 0; w < NWARP; ++w)
            hist[w][b] = 0;
    }
    if (tid == 0) { n_top = 0; n_eq = 0; }
    __syncthreads();

    // ---- load pass: HBM -> SMEM keys + per-warp high-byte histogram ------
    const __nv_bfloat16* srow = scores + (int64_t)row * C;
    for (int base = 0; base < C; base += NTHREAD * 8) {
        const int c0 = base + tid * 8;
        if (c0 >= C) continue;  // C % 8 == 0, so c0 < C implies c0 + 8 <= C
        const uint4 v = *reinterpret_cast<const uint4*>(srow + c0);
        const unsigned short* b16 = reinterpret_cast<const unsigned short*>(&v);
        #pragma unroll
        for (int u = 0; u < 8; ++u) {
            const int slot = c0 + u;
            const int c = chunk_ids == nullptr
                ? slot
                : chunk_ids[(int64_t)row * C + slot];
            unsigned kk = 0;
            if (c >= 0 && c < c_hi) {
                kk = key_of(b16[u]);
                if (pin_boundaries && c == 0) kk = KEY_SINK;         // pin sink chunk
                if (pin_boundaries && c == c_hi - 1) kk = KEY_TAIL;  // pin current-token chunk
                atomicAdd(&hist[warp][kk >> 8], 1);
            }
            keys[slot] = (unsigned short)kk;
        }
    }
    __syncthreads();

    // ---- radix pass 1: high byte bin holding rank SEL ---------------------
    for (int b = tid; b < 256; b += NTHREAD) {
        int acc = hist[0][b];
        #pragma unroll
        for (int w = 1; w < NWARP; ++w)
            acc += hist[w][b];
        hist[0][b] = acc;
    }
    __syncthreads();
    if (tid == 0) {
        int acc = 0, b = 255;
        for (; b >= 0; --b) {
            const int h = hist[0][b];
            if (acc + h >= SEL || (b == 0)) break;
            acc += h;
        }
        bin_hi = b;
        cnt_gt_hi = acc;  // count of keys in bins > b
    }
    for (int b = tid; b < 256; b += NTHREAD)
        hist[1][b] = 0;
    __syncthreads();

    // ---- radix pass 2: low-byte histogram inside bin_hi (SMEM only) ------
    const int bh = bin_hi;
    for (int slot = tid; slot < C; slot += NTHREAD) {
        const unsigned kk = keys[slot];
        if ((int)(kk >> 8) == bh)
            atomicAdd(&hist[1][kk & 0xFF], 1);
    }
    __syncthreads();
    if (tid == 0) {
        int acc = cnt_gt_hi, b = 255;
        for (; b >= 0; --b) {
            const int h = hist[1][b];
            if (acc + h >= SEL || b == 0) break;
            acc += h;
        }
        thr_key = (bh << 8) | b;
        need_eq = SEL - acc;      // how many ==thr to accept (>= 1 unless short row)
        cnt_gt_hi = acc;          // now: count of keys > thr_key
    }
    __syncthreads();

    // ---- collect: keys > thr definitely in; ties fill the remainder ------
    const unsigned thr = (unsigned)thr_key;
    const int n_gt = cnt_gt_hi;
    for (int slot_id = tid; slot_id < C; slot_id += NTHREAD) {
        const unsigned kk = keys[slot_id];
        if (kk == 0) continue;
        if (kk > thr) {
            const int slot = atomicAdd(&n_top, 1);
            fk[slot] = (unsigned short)kk;
            fi[slot] = slot_id;  // temp: unordered candidate slot
        } else if (kk == thr) {
            const int e = atomicAdd(&n_eq, 1);
            if (e < need_eq) {
                fk[n_gt + e] = (unsigned short)kk;
                fi[n_gt + e] = slot_id;
            }
        }
    }
    __syncthreads();
    if (tid == 0)
        sel_n = n_top + min(n_eq, max(need_eq, 0));
    __syncthreads();
    const int n = sel_n;

    // ---- order the <=SEL winners descending by parallel rank-count --------
    int my_id = -1, my_rank = 0;
    if (tid < n) {
        const unsigned kk = fk[tid];
        my_id = fi[tid];
        for (int j = 0; j < n; ++j) {
            const unsigned kj = fk[j];
            if (kj > kk || (kj == kk && fi[j] < my_id)) ++my_rank;
        }
    }
    __syncthreads();
    if (my_id >= 0)
        fi[my_rank] = my_id;
    __syncthreads();

    // ---- stage 2: exact fp8 QK on selected chunks + quota pick -----------
    // fi is descending: rank j (0 = best coarse score) -> fi[j];
    // warps take ranks round-robin.
    for (int j = warp; j < n; j += NWARP) {
        const int candidate_slot = fi[j];
        const int c = chunk_ids == nullptr
            ? candidate_slot
            : chunk_ids[(int64_t)row * C + candidate_slot];
        const int quota = (j < NDENSE) ? QDENSE : QSPARSE;
        const int obase = (j < NDENSE) ? j * QDENSE
                                       : NDENSE * QDENSE + (j - NDENSE) * QSPARSE;
        const int t_loc = lane % 16;          // token within chunk
        const int half = lane / 16;           // 64-dim half of the dot
        const int tok = c * CHUNK + t_loc;    // request-relative position

        // chunk c lives entirely in page bt_row[c / 4], slots (c % 4)*16 ..
        const int page = bt_row[c >> 2];
        const unsigned char* pbase = kv + (int64_t)page * PAGE_BYTES;
        const int slot = (c & 3) * CHUNK + t_loc;

        // rows past ke read garbage from the (allocated) page; they are
        // masked before use
        const unsigned char* krow = pbase + slot * DIM + half * 64;
        const float4* qh4 = reinterpret_cast<const float4*>(q_s + half * 64);
        uint4 v[4];
        #pragma unroll
        for (int i = 0; i < 4; ++i)
            v[i] = reinterpret_cast<const uint4*>(krow)[i];
        const __nv_fp8x2_storage_t* p2 =
            reinterpret_cast<const __nv_fp8x2_storage_t*>(v);
        float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            const float4 qv = qh4[i];
            __half2_raw h01 = __nv_cvt_fp8x2_to_halfraw2(p2[2 * i], __NV_E4M3);
            __half2_raw h23 = __nv_cvt_fp8x2_to_halfraw2(p2[2 * i + 1], __NV_E4M3);
            const float2 f01 = __half22float2(*reinterpret_cast<__half2*>(&h01));
            const float2 f23 = __half22float2(*reinterpret_cast<__half2*>(&h23));
            a0 += qv.x * f01.x;
            a1 += qv.y * f01.y;
            a2 += qv.z * f23.x;
            a3 += qv.w * f23.y;
        }
        float part = (a0 + a1) + (a2 + a3);
        part += __shfl_down_sync(0xffffffffu, part, 16);

        float s = -1e30f;
        if (lane < 16 && tok < ke_r) {
            const float scale =
                *reinterpret_cast<const float*>(pbase + PAGE * DIM + slot * 4);
            s = part * scale;
            if (pin_boundaries && (tok == 0 || tok == ke_r - 1))
                s = 1e30f;  // pin sink + current token
        }

        // bitonic sort-16 across lanes (descending, ties -> lower token id):
        // 10 compare-exchange steps replace quota rounds of serial argmax.
        // width-16 shuffles keep lanes 16..31 in their own (garbage) group.
        int si = lane & 15;
        #pragma unroll
        for (int kk = 2; kk <= 16; kk <<= 1) {
            #pragma unroll
            for (int jj = kk >> 1; jj > 0; jj >>= 1) {
                const float os = __shfl_xor_sync(0xffffffffu, s, jj, 16);
                const int oi = __shfl_xor_sync(0xffffffffu, si, jj, 16);
                const bool keep_max = (((lane & kk) == 0) == ((lane & jj) == 0));
                const bool other_wins = (os > s) || (os == s && oi < si);
                if (keep_max == other_wins) { s = os; si = oi; }
            }
        }
        if (lane < quota && s > -1e30f)
            out[(int64_t)row * OUTK + obase + lane] = c * CHUNK + si;
    }
}

// One warp computes one selected q . chunk_sum score.  Unlike the dense
// cuBLAS path, this follows per-head logical chunk ids and never scores a
// pruned coarse block.
__global__ void selected_chunk_scores_paged_kernel(
    __nv_bfloat16* __restrict__ scores,        // [M, C]
    const __nv_bfloat16* __restrict__ q,       // [M, DIM]
    const float* __restrict__ chunk_sum,       // [P*4, DIM]
    const int* __restrict__ bt,                // [B, bt_stride]
    const int* __restrict__ batch,             // [M]
    const int* __restrict__ ke_row,            // [M]
    const int* __restrict__ chunk_ids,          // [M, C] logical chunks
    const int M,
    const int C,
    const int bt_stride,
    const bool use_mean
) {
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int flat = blockIdx.x * NWARP + warp;
    if (flat >= M * C) return;
    const int row = flat / C;
    const int slot = flat - row * C;
    const int c = chunk_ids[(int64_t)row * C + slot];
    const int c_hi = (ke_row[row] + CHUNK - 1) / CHUNK;
    if (c < 0 || c >= c_hi) {
        if (lane == 0)
            scores[flat] = __float2bfloat16(0.f);
        return;
    }

    const int* bt_row = bt + (int64_t)batch[row] * bt_stride;
    const int page = bt_row[c >> 2];
    const int physical_chunk = page * (PAGE / CHUNK) + (c & 3);
    const float* stat = chunk_sum + (int64_t)physical_chunk * DIM;
    const __nv_bfloat16* qrow = q + (int64_t)row * DIM;
    float value = 0.f;
    #pragma unroll
    for (int d = lane; d < DIM; d += 32)
        value += __bfloat162float(qrow[d]) * stat[d];
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        value += __shfl_down_sync(0xffffffffu, value, offset);
    if (lane == 0)
        scores[flat] = __float2bfloat16(
            use_mean ? value / min(CHUNK, ke_row[row] - c * CHUNK) : value);
}

static void launch_chunk16_quota_paged(
    torch::Tensor scores,  // [M, C] bf16, C % 8 == 0
    torch::Tensor q,       // [M, 128] bf16
    torch::Tensor kv,      // [P, PAGE_BYTES] uint8 NSA indexer cache
    torch::Tensor bt,      // [B, max_pages] i32 block tables
    torch::Tensor batch,   // [M] i32 request index per row
    torch::Tensor ke_row,  // [M] i32 visible token count per row
    const int* chunk_ids,
    torch::Tensor out,     // [M, OUTK] i32 pre-filled -1
    bool pin_boundaries = true
) {
    TORCH_CHECK(scores.is_cuda() && scores.dtype() == torch::kBFloat16 && scores.is_contiguous());
    TORCH_CHECK(scores.size(1) % 8 == 0, "pad C to a multiple of 8");
    TORCH_CHECK(q.dtype() == torch::kBFloat16 && q.is_contiguous() && q.size(1) == DIM);
    TORCH_CHECK(kv.dtype() == torch::kUInt8 && kv.is_contiguous() && kv.size(1) == PAGE_BYTES);
    TORCH_CHECK(bt.dtype() == torch::kInt32 && bt.is_contiguous());
    TORCH_CHECK(batch.dtype() == torch::kInt32 && ke_row.dtype() == torch::kInt32);
    TORCH_CHECK(out.dtype() == torch::kInt32 && out.is_contiguous() && out.size(1) == OUTK);
    const int m = scores.size(0);
    const int c = scores.size(1);
    TORCH_CHECK(q.size(0) == m && out.size(0) == m && batch.numel() == m && ke_row.numel() == m);
    if (m == 0) return;

    const int smem = c * sizeof(unsigned short);
    static int max_smem = 0;
    if (smem > max_smem) {  // opt in to >48KB dynamic SMEM for long contexts
        cudaFuncSetAttribute(
            chunk16_quota_paged_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem);
        max_smem = smem;
    }
    chunk16_quota_paged_kernel<<<m, NTHREAD, smem, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(scores.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()),
        kv.data_ptr<unsigned char>(),
        bt.data_ptr<int>(),
        batch.data_ptr<int>(),
        ke_row.data_ptr<int>(),
        chunk_ids,
        out.data_ptr<int>(),
        c,
        (int)bt.size(1), pin_boundaries);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void chunk16_quota_paged(
    torch::Tensor scores,
    torch::Tensor q,
    torch::Tensor kv,
    torch::Tensor bt,
    torch::Tensor batch,
    torch::Tensor ke_row,
    torch::Tensor out
) {
    launch_chunk16_quota_paged(scores, q, kv, bt, batch, ke_row, nullptr, out);
}

void chunk16_quota_paged_mapped(
    torch::Tensor scores,
    torch::Tensor q,
    torch::Tensor kv,
    torch::Tensor bt,
    torch::Tensor batch,
    torch::Tensor ke_row,
    torch::Tensor chunk_ids,
    torch::Tensor out
) {
    TORCH_CHECK(chunk_ids.is_cuda() && chunk_ids.dtype() == torch::kInt32);
    TORCH_CHECK(chunk_ids.is_contiguous() && chunk_ids.sizes() == scores.sizes());
    launch_chunk16_quota_paged(
        scores, q, kv, bt, batch, ke_row, chunk_ids.data_ptr<int>(), out);
}

void selected_chunk_scores_paged(
    torch::Tensor scores,
    torch::Tensor q,
    torch::Tensor chunk_sum,
    torch::Tensor bt,
    torch::Tensor batch,
    torch::Tensor ke_row,
    torch::Tensor chunk_ids,
    bool use_mean = false
) {
    TORCH_CHECK(scores.is_cuda() && scores.dtype() == torch::kBFloat16 && scores.is_contiguous());
    TORCH_CHECK(q.is_cuda() && q.dtype() == torch::kBFloat16 && q.is_contiguous() && q.size(1) == DIM);
    TORCH_CHECK(chunk_sum.is_cuda() && chunk_sum.dtype() == torch::kFloat32);
    TORCH_CHECK(chunk_sum.is_contiguous() && chunk_sum.size(1) == DIM);
    TORCH_CHECK(bt.is_cuda() && bt.dtype() == torch::kInt32 && bt.is_contiguous());
    TORCH_CHECK(batch.dtype() == torch::kInt32 && ke_row.dtype() == torch::kInt32);
    TORCH_CHECK(chunk_ids.dtype() == torch::kInt32 && chunk_ids.is_contiguous());
    TORCH_CHECK(scores.sizes() == chunk_ids.sizes());
    const int m = scores.size(0);
    const int c = scores.size(1);
    TORCH_CHECK(q.size(0) == m && batch.numel() == m && ke_row.numel() == m);
    if (m == 0 || c == 0) return;
    const int candidates_per_cta = NWARP;
    const int blocks = (m * c + candidates_per_cta - 1) / candidates_per_cta;
    selected_chunk_scores_paged_kernel<<<
        blocks, NTHREAD, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<__nv_bfloat16*>(scores.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()),
        chunk_sum.data_ptr<float>(),
        bt.data_ptr<int>(),
        batch.data_ptr<int>(),
        ke_row.data_ptr<int>(),
        chunk_ids.data_ptr<int>(),
        m,
        c,
        (int)bt.size(1), use_mean);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void chunk16_quota_paged_mapped_mean(
    torch::Tensor scores, torch::Tensor q, torch::Tensor kv, torch::Tensor bt,
    torch::Tensor batch, torch::Tensor ke_row, torch::Tensor chunk_ids, torch::Tensor out
) {
    TORCH_CHECK(chunk_ids.is_cuda() && chunk_ids.dtype() == torch::kInt32);
    TORCH_CHECK(chunk_ids.is_contiguous() && chunk_ids.sizes() == scores.sizes());
    launch_chunk16_quota_paged(
        scores, q, kv, bt, batch, ke_row, chunk_ids.data_ptr<int>(), out, false);
}

void selected_chunk_mean_scores_paged(
    torch::Tensor scores, torch::Tensor q, torch::Tensor sums, torch::Tensor bt,
    torch::Tensor batch, torch::Tensor lengths, torch::Tensor ids
) {
    selected_chunk_scores_paged(scores, q, sums, bt, batch, lengths, ids, true);
}

void selected_chunk_sum_scores_paged(
    torch::Tensor scores, torch::Tensor q, torch::Tensor sums, torch::Tensor bt,
    torch::Tensor batch, torch::Tensor lengths, torch::Tensor ids
) {
    selected_chunk_scores_paged(scores, q, sums, bt, batch, lengths, ids, false);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("chunk16_quota_paged_mapped_mean", &chunk16_quota_paged_mapped_mean);
    m.def("selected_chunk_mean_scores_paged", &selected_chunk_mean_scores_paged);
    m.def("chunk16_quota_paged", &chunk16_quota_paged, "fused chunk16 quota top-k (paged)");
    m.def(
        "chunk16_quota_paged_mapped",
        &chunk16_quota_paged_mapped,
        "fused chunk16 quota top-k over mapped logical chunks");
    m.def(
        "selected_chunk_scores_paged",
        &selected_chunk_sum_scores_paged,
        "qk scores over selected logical chunks");
}

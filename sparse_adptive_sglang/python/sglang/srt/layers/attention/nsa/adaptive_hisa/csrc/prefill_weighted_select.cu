/**
 * Batched token-budgeted leaf selector for Adaptive-HISA sparse prefill.
 *
 * One CTA per query row. For row r with coarse scores ``scores[r, :]`` over
 * ``n = num_leaves`` leaves (``leaf_len[i]`` tokens each), select leaves in
 * (score descending, leaf index ascending) order until ``budget`` tokens are
 * covered; the crossing leaf is taken partially so the budget is filled
 * exactly. This is the same contract as the previous
 * ``argsort -> cumsum -> per-slot binary search`` path in prefill_select.py
 * (ties there followed torch.argsort's unspecified order; here they are
 * index-ascending), and the same algorithm as the decode selector in
 * weighted_select.cu, batched over rows and without sink/tail guards (prefill
 * forces sink leaves to +inf in the coarse scores and scores the causal local
 * window densely).
 *
 * Per row (n <= kMaxLeaves, everything in shared memory):
 *   1. cache ordered FP32 keys and lengths;
 *   2. four 8-bit radix passes: token-weighted 256-bin histogram of the leaves
 *      matching the resolved prefix, then a block-wide suffix scan finds the
 *      bin holding the crossing token (no serial bin loop);
 *   3. a logical-order pass (one packed 64-bit block scan per 256 leaves)
 *      turns the exact threshold into per-leaf ``taken`` lengths and output
 *      offsets (stable index-ascending ties) and compacts the selected leaves;
 *   4. one warp writes each selected interval's token ids contiguously;
 *      optionally the packed ``(start, length, output_offset)`` segments.
 *
 * Measured on H20 (8192 rows x 1920 leaves, budget 8192): 0.50 ms vs 2.03 ms
 * for argsort + cumsum + per-slot binary search; the kernel is issue-bound
 * (~27k warp-instructions per row), not bandwidth-bound.
 *
 * Output layout matches ``expand_candidates``: slots ``[0, budget)`` are leaf
 * tokens in logical order (``-1`` padded if the leaves hold fewer tokens than
 * the budget), slots ``[budget, total)`` are the local window
 * ``n_complete + (s - budget)``.
 */
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/library.h>

#include <cstdint>
#include <optional>

namespace {

constexpr int kThreads = 256;
constexpr int kRadix = 256;
constexpr int kWarps = kThreads / 32;
constexpr int kMaxLeaves = 4096;
// Packed scan fields: above tokens (25 bits), tied tokens (26 bits), above
// leaf count (13 bits, holds 4096). The above sum never exceeds ``budget``
// and the tied sum never exceeds the covered prefix ``n_complete``; both are
// range-checked on the host.
constexpr uint64_t kMask25 = (1ull << 25) - 1;
constexpr uint64_t kMask26 = (1ull << 26) - 1;

static_assert(kThreads == kRadix, "one thread per histogram bin");
static_assert(kMaxLeaves < (1 << 13), "above count must fit 13 bits");
static_assert(kMaxLeaves <= 65536, "s_sel stores uint16 leaf indices");

__device__ __forceinline__ uint32_t ordered_float_key(float value) {
  uint32_t bits = __float_as_uint(value);
  if ((bits & 0x7fffffffu) == 0) {
    bits = 0; // -0.0 == +0.0
  }
  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

// Lean block scan (one item per thread): shuffle warp scan + kWarps partials.
// ``scratch`` needs 2 * kWarps entries of T in shared memory; callers must
// separate consecutive scans by a __syncthreads().
template <typename T>
__device__ __forceinline__ T warp_inclusive_scan(T v, int lane) {
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const T t = __shfl_up_sync(0xffffffffu, v, o);
    if (lane >= o) {
      v += t;
    }
  }
  return v;
}

template <typename T>
__device__ __forceinline__ T block_exclusive_scan(T item, int tid, T *scratch,
                                                  T &aggregate) {
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const T incl = warp_inclusive_scan(item, lane);
  if (lane == 31) {
    scratch[warp] = incl;
  }
  __syncthreads();
  if (tid < 32) {
    T v = lane < kWarps ? scratch[lane] : T(0);
    v = warp_inclusive_scan(v, lane);
    if (lane < kWarps) {
      scratch[kWarps + lane] = v;
    }
  }
  __syncthreads();
  const T warp_prefix = warp > 0 ? scratch[kWarps + warp - 1] : T(0);
  aggregate = scratch[kWarps + kWarps - 1];
  return warp_prefix + incl - item;
}

// Weighted histogram increment. Logits cluster in one or two exponent bins,
// so same-address shared atomics serialise; measured on H20 the native
// shared atomics are still ~1.7x faster than __match_any_sync aggregation
// (-DPWS_MATCH_ATOMICS keeps that variant for other architectures).
__device__ __forceinline__ void histogram_add(uint32_t *histogram, int bin,
                                              uint32_t weight, int lane) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800 && defined(PWS_MATCH_ATOMICS)
  const unsigned peers = __match_any_sync(0xffffffffu, bin);
  const uint32_t sum = __reduce_add_sync(peers, weight);
  if (lane == __ffs(peers) - 1) {
    atomicAdd(&histogram[bin], sum);
  }
#else
  atomicAdd(&histogram[bin], weight);
#endif
}

template <int MAX_LEAVES>
__global__ __launch_bounds__(kThreads) void batched_weighted_select_kernel(
    const float *__restrict__ scores, int64_t score_stride, int row0,
    const int32_t *__restrict__ lengths, const int32_t *__restrict__ starts,
    const int32_t *__restrict__ num_leaves, int32_t *__restrict__ candidates,
    int64_t cand_stride, int32_t *__restrict__ seg_starts,
    int32_t *__restrict__ seg_lengths, int32_t *__restrict__ seg_offsets,
    int32_t *__restrict__ seg_count, int64_t seg_stride, int capacity,
    int budget, int n_complete, int total) {
  // s_key: ordered key during selection, output offset after the final scan.
  // s_len: clipped length during selection, ``taken`` after the final scan.
  __shared__ uint32_t s_key[MAX_LEAVES];
  __shared__ int32_t s_len[MAX_LEAVES];
  // Compacted selected leaf indices: above-threshold leaves from the bottom,
  // selected tied leaves from the top (never overlap: together <= n).
  __shared__ uint16_t s_sel[MAX_LEAVES];
  __shared__ uint32_t histogram[kRadix + 32]; // + per-lane dummy bins
  __shared__ uint32_t threshold_key;
  __shared__ uint32_t prefix_mask;
  __shared__ int32_t remaining;
  __shared__ int32_t select_all;
  __shared__ int32_t leaf_total;
  __shared__ int32_t above_count;
  __shared__ int32_t tied_count;
  __shared__ int32_t seg_total;
  __shared__ uint64_t scan_scratch[2 * kWarps];

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int r = blockIdx.x;
  const int n = min(max(*num_leaves, 0), capacity);
  const float *row_scores = scores + static_cast<int64_t>(row0 + r) * score_stride;
  int32_t *row_out = candidates + static_cast<int64_t>(r) * cand_stride;

  for (int i = tid; i < n; i += kThreads) {
    const int32_t len = lengths[i];
    s_len[i] = max(len, 0);
    s_key[i] = len > 0 ? ordered_float_key(row_scores[i]) : 0u;
  }
  if (tid == 0) {
    threshold_key = 0;
    prefix_mask = 0;
    remaining = budget;
    select_all = 0;
    tied_count = 0;
  }
  __syncthreads();

  // Four radix passes over the 32-bit ordered key, high byte first.
#pragma unroll 1
  for (int shift = 24; shift >= 0; shift -= 8) {
    histogram[tid] = 0;
    __syncthreads();
    const uint32_t pm = prefix_mask;
    const uint32_t tk = threshold_key;
    for (int base = 0; base < n; base += kThreads) { // uniform trip count
      const int i = base + tid;
      // Non-matching leaves add 0 to a private per-lane dummy bin: no branch,
      // no reconvergence, no same-address serialisation.
      int bin = kRadix + lane;
      uint32_t weight = 0;
      if (i < n) {
        const int32_t len = s_len[i];
        const uint32_t key = s_key[i];
        if (len > 0 && (key & pm) == tk) {
          bin = static_cast<int>((key >> shift) & 0xffu);
          weight = static_cast<uint32_t>(len);
        }
      }
      histogram_add(histogram, bin, weight, lane);
    }
    __syncthreads();
    // Thread t owns bin (255 - t): an exclusive prefix over that order is the
    // token weight of all strictly higher bins.
    const int bin = kRadix - 1 - tid;
    const uint32_t weight = histogram[bin];
    const uint32_t rem = static_cast<uint32_t>(remaining);
    uint32_t total_weight = 0;
    const uint32_t above = block_exclusive_scan<uint32_t>(
        weight, tid, reinterpret_cast<uint32_t *>(scan_scratch), total_weight);
    if (above <= rem && rem < above + weight) {
      // Exactly one bin satisfies this when the matching weight exceeds rem.
      const uint32_t byte_mask = 0xffu << shift;
      threshold_key = (tk & ~byte_mask) | (static_cast<uint32_t>(bin) << shift);
      prefix_mask = pm | byte_mask;
      remaining = static_cast<int32_t>(rem - above);
    }
    if (tid == 0 && total_weight <= rem) {
      select_all = 1; // every matching leaf fits: whole prefix selected
    }
    __syncthreads();
    if (select_all) {
      break;
    }
  }

  // Logical-order pass in stripes of kThreads leaves with one packed block
  // scan per stripe. Each item packs three fields:
  //   above_len  [0, 25)   leaves with key > threshold, taken whole
  //   equal_len  [25, 51)  leaves with key == threshold, taken index-ascending
  //                        until ``remaining`` is used up (crossing leaf partial)
  //   above_cnt  [51, 64)  1 for ``above`` leaves
  // The exclusive prefix gives everything at once: output offset =
  // above_excl + min(rem, equal_excl); the above count compacts those leaves
  // into s_sel[0, above_cnt). Selected tied leaves (normally just the one
  // crossing leaf) are appended from the top of s_sel, so expansion touches
  // only selected leaves.
  const uint32_t tk = threshold_key;
  const int32_t rem = remaining;
  const int sa = select_all;
  uint64_t carry = 0; // block-uniform running totals
  for (int base = 0; base < n; base += kThreads) {
    const int i = base + tid;
    int32_t len = 0;
    uint32_t key = 0;
    if (i < n) {
      len = s_len[i];
      key = s_key[i];
    }
    const bool above = len > 0 && (sa || key > tk);
    const bool equal = len > 0 && !sa && key == tk;
    const uint64_t item = (above ? static_cast<uint64_t>(len) | (1ull << 51) : 0ull) |
                          (equal ? static_cast<uint64_t>(len) << 25 : 0ull);
    uint64_t aggregate = 0;
    uint64_t excl = block_exclusive_scan<uint64_t>(item, tid, scan_scratch, aggregate);
    excl += carry;
    carry += aggregate;
    const int32_t above_excl = static_cast<int32_t>(excl & kMask25);
    const int32_t equal_excl = static_cast<int32_t>((excl >> 25) & kMask26);
    const int32_t sel_excl = static_cast<int32_t>(excl >> 51);
    int32_t taken = 0;
    if (above) {
      taken = len;
    } else if (equal) {
      taken = min(max(rem - equal_excl, 0), len);
    }
    const int32_t offset = above_excl + min(rem, equal_excl);
    __syncthreads(); // stripe reads done before rewriting s_key/s_len
    if (i < n) {
      s_len[i] = taken;
      s_key[i] = static_cast<uint32_t>(offset);
      if (above) {
        s_sel[sel_excl] = static_cast<uint16_t>(i);
      } else if (taken > 0) {
        s_sel[MAX_LEAVES - 1 - atomicAdd(&tied_count, 1)] = static_cast<uint16_t>(i);
      }
    }
    __syncthreads(); // scan storage reuse in the next stripe
  }
  if (tid == 0) {
    const int32_t above_total = static_cast<int32_t>(carry & kMask25);
    const int32_t equal_total = static_cast<int32_t>((carry >> 25) & kMask26);
    leaf_total = above_total + min(rem, equal_total); // == budget unless select_all
    above_count = static_cast<int32_t>(carry >> 51);
    seg_total = above_count + tied_count;
  }
  __syncthreads();

  // Interval-forward expansion over the compacted selected leaves only: one
  // warp per selected leaf writes its contiguous token range (no per-slot
  // search, no scan over unselected leaves).
  const int warp = tid >> 5;
  const int n_sel = seg_total;
  const int n_above = above_count;
  for (int k = warp; k < n_sel; k += kWarps) {
    const int i = k < n_above ? s_sel[k] : s_sel[MAX_LEAVES - 1 - (k - n_above)];
    const int32_t taken = s_len[i];
    const int32_t out = static_cast<int32_t>(s_key[i]);
    const int32_t start = starts[i];
    if (seg_starts != nullptr && lane == 0) {
      const int64_t slot = static_cast<int64_t>(r) * seg_stride + k;
      seg_starts[slot] = start;
      seg_lengths[slot] = taken;
      seg_offsets[slot] = out;
    }
    int32_t *dst = row_out + out + lane;
    int32_t value = start + lane;
#pragma unroll 1
    for (int j = lane; j < taken; j += 32, dst += 32, value += 32) {
      *dst = value;
    }
  }
  {
    const int32_t filled = leaf_total; // <= budget
    int32_t *dst = row_out + filled + tid;
#pragma unroll 1
    for (int s = filled + tid; s < budget; s += kThreads, dst += kThreads) {
      *dst = -1;
    }
    dst = row_out + budget + tid;
    int32_t value = n_complete + tid;
#pragma unroll 1
    for (int s = budget + tid; s < total; s += kThreads, dst += kThreads, value += kThreads) {
      *dst = value;
    }
  }
  if (tid == 0 && seg_count != nullptr) {
    seg_count[r] = seg_total;
  }
}

} // namespace

void batched_weighted_select_interface(
    const at::Tensor &scores, int64_t row0, int64_t rows,
    const at::Tensor &lengths, const at::Tensor &starts,
    const at::Tensor &num_leaves, at::Tensor &candidates,
    std::optional<at::Tensor> seg_starts, std::optional<at::Tensor> seg_lengths,
    std::optional<at::Tensor> seg_offsets, std::optional<at::Tensor> seg_count,
    int64_t budget, int64_t n_complete, int64_t total) {
  TORCH_CHECK(scores.is_cuda() && lengths.is_cuda() && starts.is_cuda() &&
                  num_leaves.is_cuda() && candidates.is_cuda(),
              "prefill selector tensors must be CUDA");
  TORCH_CHECK(scores.dim() == 2 && scores.scalar_type() == at::kFloat &&
                  scores.stride(1) == 1,
              "scores must be a row-major f32 [rows, capacity] matrix");
  const int capacity = static_cast<int>(scores.size(1));
  TORCH_CHECK(capacity > 0 && capacity <= kMaxLeaves,
              "prefill selector supports 1..4096 leaves");
  TORCH_CHECK(row0 >= 0 && rows > 0 && row0 + rows <= scores.size(0),
              "row window out of range");
  TORCH_CHECK(lengths.scalar_type() == at::kInt && starts.scalar_type() == at::kInt &&
                  lengths.is_contiguous() && starts.is_contiguous() &&
                  lengths.numel() >= capacity && starts.numel() >= capacity,
              "leaf metadata must be contiguous i32 with >= capacity entries");
  TORCH_CHECK(num_leaves.scalar_type() == at::kInt && num_leaves.numel() == 1,
              "num_leaves must be one i32 device scalar");
  TORCH_CHECK(candidates.scalar_type() == at::kInt && candidates.dim() == 2 &&
                  candidates.is_contiguous() && candidates.size(0) == rows &&
                  candidates.size(1) == total,
              "candidates must be contiguous i32 [rows, total]");
  TORCH_CHECK(budget > 0 && budget <= total, "budget must be in (0, total]");
  TORCH_CHECK(budget < (1ll << 25) && n_complete >= 0 && n_complete < (1ll << 26),
              "budget / n_complete exceed the packed-scan field widths");
  const bool want_seg = seg_starts.has_value();
  int32_t *seg_s = nullptr, *seg_l = nullptr, *seg_o = nullptr, *seg_c = nullptr;
  int64_t seg_stride = 0;
  if (want_seg) {
    TORCH_CHECK(seg_lengths.has_value() && seg_offsets.has_value() && seg_count.has_value(),
                "all segment outputs must be given together");
    const at::Tensor &ss = *seg_starts;
    TORCH_CHECK(ss.is_cuda() && ss.scalar_type() == at::kInt && ss.dim() == 2 &&
                    ss.is_contiguous() && ss.size(0) == rows && ss.size(1) >= capacity,
                "seg_starts must be contiguous i32 [rows, >= capacity]");
    TORCH_CHECK(seg_lengths->sizes() == ss.sizes() && seg_offsets->sizes() == ss.sizes() &&
                    seg_lengths->is_contiguous() && seg_offsets->is_contiguous() &&
                    seg_lengths->scalar_type() == at::kInt &&
                    seg_offsets->scalar_type() == at::kInt,
                "segment buffers must match seg_starts");
    TORCH_CHECK(seg_count->scalar_type() == at::kInt && seg_count->numel() == rows,
                "seg_count must be i32 [rows]");
    seg_s = ss.data_ptr<int32_t>();
    seg_l = seg_lengths->data_ptr<int32_t>();
    seg_o = seg_offsets->data_ptr<int32_t>();
    seg_c = seg_count->data_ptr<int32_t>();
    seg_stride = ss.stride(0);
  }
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  auto launch = [&](auto kernel) {
    kernel<<<static_cast<unsigned>(rows), kThreads, 0, stream>>>(
        scores.data_ptr<float>(), scores.stride(0), static_cast<int>(row0),
        lengths.data_ptr<int32_t>(), starts.data_ptr<int32_t>(),
        num_leaves.data_ptr<int32_t>(), candidates.data_ptr<int32_t>(),
        candidates.stride(0), seg_s, seg_l, seg_o, seg_c, seg_stride, capacity,
        static_cast<int>(budget), static_cast<int>(n_complete),
        static_cast<int>(total));
  };
  if (capacity <= 2048) {
    launch(batched_weighted_select_kernel<2048>);
  } else {
    launch(batched_weighted_select_kernel<kMaxLeaves>);
  }
  TORCH_CHECK(cudaGetLastError() == cudaSuccess,
              "batched_weighted_select_kernel launch failed");
}

TORCH_LIBRARY(adaptive_hisa_prefill_select, m) {
  m.def("select(Tensor scores, int row0, int rows, Tensor lengths, Tensor starts, "
        "Tensor num_leaves, Tensor(a!) candidates, Tensor(b!)? seg_starts, "
        "Tensor(c!)? seg_lengths, Tensor(d!)? seg_offsets, Tensor(e!)? seg_count, "
        "int budget, int n_complete, int total) -> ()");
}

TORCH_LIBRARY_IMPL(adaptive_hisa_prefill_select, CUDA, m) {
  m.impl("select", batched_weighted_select_interface);
}

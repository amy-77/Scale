/**
 * Exact token-budgeted selector for Adaptive-HISA decode.
 *
 * Contract (mirrors the h20-1 accuracy reference ``select_candidates``):
 *   guard  = [0, sink_end) U [tail_start, seq_len), always selected raw,
 *            sink_end = min(sink, seq_len), tail_start = max(sink_end, seq_len - tail)
 *   leaves are clipped to [sink_end, tail_start); the clipped lengths fill
 *            ``budget - |guard|`` in (score descending, leaf index ascending)
 *            order. The crossing leaf is taken partially so the budget is
 *            filled exactly whenever enough tokens exist.
 *
 * A full stable sort of the graph-sized workspace dominated decode. The
 * production fast path generalises HISA's threshold-bucket narrowing:
 * one weighted FP32 high-byte histogram over all leaves, one full scan that
 * packs only the threshold bucket, then three radix passes over that packed
 * set. A final logical scan implements stable leaf-index ties without putting
 * the index into the radix key. Capacities above 4096 retain the original
 * exact six-pass 48-bit selector as a general fallback.
 *
 * Candidate order is logical-token order rather than score order. Expansion
 * is interval-forward (one warp writes each selected range), avoiding the old
 * per-token binary search. Fine scoring and Top-K are permutation invariant;
 * deterministic logical order also makes CUDA graph replay bit-stable.
 */
#include <ATen/cuda/CUDAContext.h>
#include <ATen/core/TensorBody.h>
#include <c10/cuda/CUDAStream.h>
#include <cub/block/block_scan.cuh>
#include <torch/library.h>

#include <cstdint>

namespace {

constexpr int kThreads = 1024;
constexpr int kRadix = 256;
constexpr int kKeyPasses = 6; // f32 ordered key (32 bits) + leaf tie (16 bits)
constexpr int kFastMaxLeaves = 4096;
constexpr int kRatioBins = 101; // integer percentages [0, 100]
constexpr int kStatCalls = 101;
constexpr int kStatSumN = 102;
constexpr int kStatSumM = 103;
constexpr int kStatMaxM = 104;
constexpr int kStatFallback = 105;
constexpr int kStatsSize = 106;

__device__ __forceinline__ uint32_t ordered_float_key(float value) {
  uint32_t bits = __float_as_uint(value);
  // torch stable sort considers -0.0 and +0.0 equal.
  if ((bits & 0x7fffffffu) == 0) {
    bits = 0;
  }
  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

__device__ __forceinline__ uint64_t stable_leaf_key(float score, int index) {
  // capacity is validated <= 65535 by the host interface.
  return (static_cast<uint64_t>(ordered_float_key(score)) << 16) |
         static_cast<uint64_t>(0xffffu - static_cast<uint32_t>(index));
}

// Warp-level helpers for the fast path (Phase A / B / C). They are selected
// at compile time by the bitmask ADAPTIVE_HISA_WS_WARP_OPT (set from
// decode_select.py; 1 = Phase A warp-aggregated histogram, 2 = Phase B ballot
// compaction, 4 = Phase C warp-aggregated histogram) so the per-leaf atomic
// baseline stays available for bit-exact A/B and per-GPU tuning. On H20 the
// per-leaf shared atomics are already cheaper than __match_any_sync, so the
// default is 0; see runner/bench_weighted_select.py. Every helper is called
// by all 32 lanes of a warp with a full mask (the callers iterate in
// whole-block stripes and fold `index < n` into the predicate), so no lane of
// the sync mask can be missing.
#ifndef ADAPTIVE_HISA_WS_WARP_OPT
#define ADAPTIVE_HISA_WS_WARP_OPT 0
#endif

constexpr unsigned kFullWarp = 0xffffffffu;

__device__ __forceinline__ uint32_t warp_group_sum_u32(unsigned peer_mask,
                                                       uint32_t value) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  return __reduce_add_sync(peer_mask, value);
#else
  // All peer lanes execute the same shuffle sequence.
  uint32_t sum = 0;
  unsigned lanes = peer_mask;
  while (lanes != 0) {
    const int src = __ffs(lanes) - 1;
    sum += __shfl_sync(peer_mask, value, src);
    lanes &= lanes - 1;
  }
  return sum;
#endif
}

// histogram[bin] += weight for every valid lane, with one shared atomic per
// (warp, bin) group instead of one per lane. Integer addition, so the bin
// totals are identical to the per-lane version.
__device__ __forceinline__ void warp_aggregated_hist_add(uint32_t *histogram,
                                                         bool valid, int bin,
                                                         uint32_t weight) {
  const unsigned valid_mask = __ballot_sync(kFullWarp, valid);
  if (!valid) {
    return;
  }
  const unsigned peer_mask = __match_any_sync(valid_mask, bin);
  const uint32_t group_weight = warp_group_sum_u32(peer_mask, weight);
  if ((static_cast<int>(threadIdx.x) & 31) == __ffs(peer_mask) - 1) {
    atomicAdd(&histogram[bin], group_weight);
  }
}

// Reserve one slot per matching lane in a shared append buffer with a single
// atomicAdd per warp; returns the slot (or -1). The physical order of the
// packed records differs from the per-lane atomic version, which is fine for
// the threshold bucket: Phase C only reads (weight, index) records and the
// stable tie order is produced by the final logical scan.
__device__ __forceinline__ int warp_compact_reserve(bool match,
                                                    int32_t *global_count) {
  const unsigned match_mask = __ballot_sync(kFullWarp, match);
  const int lane = static_cast<int>(threadIdx.x) & 31;
  int base = 0;
  if (lane == 0 && match_mask != 0u) {
    base = atomicAdd(global_count, __popc(match_mask));
  }
  base = __shfl_sync(kFullWarp, base, 0);
  if (!match) {
    return -1;
  }
  const unsigned lanes_before = (1u << lane) - 1u;
  return base + __popc(match_mask & lanes_before);
}

// Threshold-bin search over a 256-bin weighted histogram, executed by one
// full warp (8 bins per lane, descending bin order). Returns the highest bin
// b with sum(hist[b+1..255]) <= remaining < sum(hist[b..255]) and writes the
// weight strictly above it; -1 when the whole histogram fits. Identical to
// the single-thread descending scan it replaces, minus ~256 dependent
// shared-memory round trips on the kernel's critical path.
__device__ __forceinline__ int warp_threshold_bin(const uint32_t *histogram,
                                                  uint32_t remaining,
                                                  uint32_t *above_out) {
  constexpr int kPerLane = kRadix / 32;
  const int lane = static_cast<int>(threadIdx.x) & 31;
  uint32_t w[kPerLane];
  uint32_t local = 0;
#pragma unroll
  for (int i = 0; i < kPerLane; ++i) {
    w[i] = histogram[kRadix - 1 - (lane * kPerLane + i)];
    local += w[i];
  }
  uint32_t inclusive = local;
#pragma unroll
  for (int offset = 1; offset < 32; offset <<= 1) {
    const uint32_t up = __shfl_up_sync(kFullWarp, inclusive, offset);
    if (lane >= offset) {
      inclusive += up;
    }
  }
  uint32_t above = inclusive - local;
  int chosen = -1;
  uint32_t chosen_above = 0;
#pragma unroll
  for (int i = 0; i < kPerLane; ++i) {
    if (chosen < 0 && above + w[i] > remaining) {
      chosen = kRadix - 1 - (lane * kPerLane + i);
      chosen_above = above;
    }
    above += w[i];
  }
  const unsigned found = __ballot_sync(kFullWarp, chosen >= 0);
  if (found == 0u) {
    return -1;
  }
  const int src = __ffs(found) - 1; // lowest lane holds the highest bins
  *above_out = __shfl_sync(kFullWarp, chosen_above, src);
  return __shfl_sync(kFullWarp, chosen, src);
}

struct Guard {
  int32_t sink_end;
  int32_t tail_start;
};

__device__ __forceinline__ Guard make_guard(int32_t seq, int sink, int tail) {
  Guard g;
  g.sink_end = min(max(sink, 0), max(seq, 0));
  g.tail_start = max(g.sink_end, seq - tail);
  return g;
}

// Clipped [first, first+len) of a leaf inside the non-guard interval.
__device__ __forceinline__ int32_t clipped_leaf(int32_t start, int32_t length,
                                                const Guard &g,
                                                int32_t *first_out) {
  if (length <= 0) {
    *first_out = start;
    return 0;
  }
  const int32_t first = min(max(start, g.sink_end), g.tail_start);
  const int32_t last = min(max(start + length, g.sink_end), g.tail_start);
  *first_out = first;
  return max(last - first, 0);
}

using LengthScan = cub::BlockScan<int32_t, kThreads>;

__global__ __launch_bounds__(kThreads) void weighted_prefix_fallback_kernel(
    const float *__restrict__ scores, const int32_t *__restrict__ lengths,
    const int32_t *__restrict__ starts, const int32_t *__restrict__ num_leaves,
    const int32_t *__restrict__ seq_len, int32_t *__restrict__ selected_prefix,
    int capacity, int budget, int sink, int tail) {
  __shared__ uint32_t histogram[kRadix];
  __shared__ uint64_t prefix_mask;
  __shared__ uint64_t prefix_value;
  __shared__ uint64_t crossing_key;
  __shared__ int32_t remaining;
  __shared__ int32_t running;
  __shared__ int32_t select_all;
  __shared__ typename LengthScan::TempStorage scan_storage;

  const int tid = threadIdx.x;
  const int n = min(max(*num_leaves, 0), capacity);
  const Guard g = make_guard(*seq_len, sink, tail);
  const int32_t guard_tokens = g.sink_end + max(*seq_len - g.tail_start, 0);

  if (tid == 0) {
    prefix_mask = 0;
    prefix_value = 0;
    crossing_key = 0;
    remaining = max(budget - guard_tokens, 0);
    running = 0;
    select_all = 0;
  }
  __syncthreads();

#pragma unroll
  for (int pass = 0; pass < kKeyPasses; ++pass) {
    if (tid < kRadix) {
      histogram[tid] = 0;
    }
    __syncthreads();

    const int shift = (kKeyPasses - 1 - pass) * 8;
    for (int index = tid; index < n; index += kThreads) {
      int32_t first;
      const int32_t length =
          clipped_leaf(starts[index], lengths[index], g, &first);
      if (length <= 0) {
        continue;
      }
      const uint64_t key = stable_leaf_key(scores[index], index);
      if ((key & prefix_mask) == prefix_value) {
        const int bin = static_cast<int>((key >> shift) & 0xffu);
        atomicAdd(&histogram[bin], static_cast<uint32_t>(length));
      }
    }
    __syncthreads();

    if (tid == 0) {
      uint32_t above = 0;
      int chosen = -1;
      for (int bin = kRadix - 1; bin >= 0; --bin) {
        const uint32_t weight = histogram[bin];
        if (above + weight > static_cast<uint32_t>(remaining)) {
          chosen = bin;
          remaining -= static_cast<int32_t>(above);
          const uint64_t byte_mask = 0xffull << shift;
          prefix_mask |= byte_mask;
          prefix_value =
              (prefix_value & ~byte_mask) |
              (static_cast<uint64_t>(chosen) << shift);
          break;
        }
        above += weight;
      }
      // The whole unresolved bucket fits: every clipped leaf is selected.
      if (chosen < 0) {
        select_all = 1;
      }
    }
    __syncthreads();
    if (select_all) {
      break;
    }
  }

  if (tid == 0) {
    // After six passes the crossing key is fully resolved; ``remaining`` is
    // the budget left for that leaf (strictly less than its clipped length).
    crossing_key = prefix_value;
  }
  __syncthreads();

  // Deterministic compaction order: leaf index is logical start order.
  // selected_prefix is inclusive and monotone, with plateaus at omitted
  // leaves; the expansion kernel can upper_bound it directly.
  for (int base = 0; base < n; base += kThreads) {
    const int index = base + tid;
    int32_t selected_length = 0;
    if (index < n) {
      int32_t first;
      const int32_t length =
          clipped_leaf(starts[index], lengths[index], g, &first);
      if (length > 0) {
        const uint64_t key = stable_leaf_key(scores[index], index);
        if (select_all || key > crossing_key) {
          selected_length = length;
        } else if (key == crossing_key) {
          selected_length = min(remaining, length);
        }
      }
    }

    int32_t exclusive = 0;
    int32_t aggregate = 0;
    LengthScan(scan_storage).ExclusiveSum(selected_length, exclusive,
                                           aggregate);
    __syncthreads();
    const int32_t stripe_base = running;
    if (index < n) {
      selected_prefix[index] = stripe_base + exclusive + selected_length;
    }
    __syncthreads();
    if (tid == 0) {
      running += aggregate;
    }
    __syncthreads();
  }
}

// Production fast path: one full weighted histogram, one full pack of the
// threshold byte, then three radix refinements over that packed bucket only.
// A final logical-order scan exactly reproduces stable score-desc/index-asc
// selection, including equal-score ties and a partially selected leaf.
__global__ __launch_bounds__(kThreads) void weighted_prefix_fast_kernel(
    const float *__restrict__ scores, const int32_t *__restrict__ lengths,
    const int32_t *__restrict__ starts, const int32_t *__restrict__ num_leaves,
    const int32_t *__restrict__ seq_len, int32_t *__restrict__ selected_prefix,
    int32_t *__restrict__ profile_stats, int capacity, int budget, int sink,
    int tail) {
  __shared__ uint32_t histogram[kRadix];
  // Upper 32 bits hold clipped token weight; lower 32 bits hold leaf index.
  __shared__ uint64_t threshold_candidates[kFastMaxLeaves];
  __shared__ uint32_t threshold_key;
  __shared__ uint32_t prefix_mask;
  __shared__ int32_t threshold_byte;
  __shared__ int32_t candidate_count;
  __shared__ int32_t remaining;
  __shared__ int32_t equal_running;
  __shared__ int32_t equal_stripe_base;
  __shared__ int32_t running;
  __shared__ int32_t select_all;
  __shared__ typename LengthScan::TempStorage scan_storage;

  const int tid = threadIdx.x;
  const int n = min(max(*num_leaves, 0), capacity);
  const Guard g = make_guard(*seq_len, sink, tail);
  const int32_t guard_tokens = g.sink_end + max(*seq_len - g.tail_start, 0);

  if (tid == 0) {
    threshold_key = 0;
    prefix_mask = 0;
    threshold_byte = -1;
    candidate_count = 0;
    remaining = max(budget - guard_tokens, 0);
    equal_running = 0;
    equal_stripe_base = 0;
    running = 0;
    select_all = 0;
  }
  if (tid < kRadix) {
    histogram[tid] = 0;
  }
  __syncthreads();

  // Phase A: exact-FP32 ordered-key high-byte histogram over all leaves.
#if (ADAPTIVE_HISA_WS_WARP_OPT & 1)
  // Whole-block stripes keep every lane in the warp intrinsics; `index < n`
  // is folded into the predicate instead of leaving the loop early.
  for (int base = 0; base < n; base += kThreads) {
    const int index = base + tid;
    int32_t length = 0;
    uint32_t key = 0;
    if (index < n) {
      int32_t first;
      length = clipped_leaf(starts[index], lengths[index], g, &first);
      if (length > 0) {
        key = ordered_float_key(scores[index]);
      }
    }
    warp_aggregated_hist_add(histogram, length > 0, static_cast<int>(key >> 24),
                             static_cast<uint32_t>(length));
  }
#else
  for (int index = tid; index < n; index += kThreads) {
    int32_t first;
    const int32_t length =
        clipped_leaf(starts[index], lengths[index], g, &first);
    if (length > 0) {
      const int bin = static_cast<int>(ordered_float_key(scores[index]) >> 24);
      atomicAdd(&histogram[bin], static_cast<uint32_t>(length));
    }
  }
#endif
  __syncthreads();

  if (tid < 32) {
    uint32_t above = 0;
    const int chosen = warp_threshold_bin(
        histogram, static_cast<uint32_t>(remaining), &above);
    if (tid == 0) {
      if (chosen >= 0) {
        remaining -= static_cast<int32_t>(above);
        threshold_byte = chosen;
        threshold_key = static_cast<uint32_t>(chosen) << 24;
        prefix_mask = 0xff000000u;
      } else {
        // The complete non-guard interval fits in the remaining budget.
        select_all = 1;
      }
    }
  }
  __syncthreads();

  if (!select_all) {
    // Phase B: second and last full score scan. Pack only leaves in the
    // threshold high-byte bucket; all higher/lower buckets are already
    // resolved and are revisited only by the final logical-order output scan.
#if (ADAPTIVE_HISA_WS_WARP_OPT & 2)
    // One candidate_count atomic per warp per stripe (ballot + lane prefix)
    // instead of one per matching leaf. candidate_count still counts every
    // matching leaf; only the physical order of the packed records changes.
    for (int base = 0; base < n; base += kThreads) {
      const int index = base + tid;
      int32_t length = 0;
      bool match = false;
      if (index < n) {
        int32_t first;
        length = clipped_leaf(starts[index], lengths[index], g, &first);
        match = length > 0 &&
                static_cast<int32_t>(ordered_float_key(scores[index]) >> 24) ==
                    threshold_byte;
      }
      const int pos = warp_compact_reserve(match, &candidate_count);
      if (match && pos < kFastMaxLeaves) {
        threshold_candidates[pos] =
            (static_cast<uint64_t>(static_cast<uint32_t>(length)) << 32) |
            static_cast<uint32_t>(index);
      }
    }
#else
    for (int index = tid; index < n; index += kThreads) {
      int32_t first;
      const int32_t length =
          clipped_leaf(starts[index], lengths[index], g, &first);
      if (length <= 0) {
        continue;
      }
      const uint32_t key = ordered_float_key(scores[index]);
      if (static_cast<int32_t>(key >> 24) == threshold_byte) {
        const int pos = atomicAdd(&candidate_count, 1);
        if (pos < kFastMaxLeaves) {
          threshold_candidates[pos] =
              (static_cast<uint64_t>(static_cast<uint32_t>(length)) << 32) |
              static_cast<uint32_t>(index);
        }
      }
    }
#endif
    __syncthreads();

    // Phase C: refine only the packed threshold bucket. The first byte was
    // consumed above, so exactly three bytes of the ordered FP32 key remain.
#pragma unroll
    for (int shift = 16; shift >= 0; shift -= 8) {
      if (tid < kRadix) {
        histogram[tid] = 0;
      }
      __syncthreads();

      const int packed_n = min(candidate_count, kFastMaxLeaves);
#if (ADAPTIVE_HISA_WS_WARP_OPT & 4)
      for (int base = 0; base < packed_n; base += kThreads) {
        const int pos = base + tid;
        bool valid = false;
        uint32_t weight = 0;
        uint32_t key = 0;
        if (pos < packed_n) {
          const uint64_t packed = threshold_candidates[pos];
          const int index = static_cast<int>(static_cast<uint32_t>(packed));
          weight = static_cast<uint32_t>(packed >> 32);
          key = ordered_float_key(scores[index]);
          valid = (key & prefix_mask) == threshold_key;
        }
        warp_aggregated_hist_add(histogram, valid,
                                 static_cast<int>((key >> shift) & 0xffu),
                                 weight);
      }
#else
      for (int pos = tid; pos < packed_n; pos += kThreads) {
        const uint64_t packed = threshold_candidates[pos];
        const int index = static_cast<int>(static_cast<uint32_t>(packed));
        const uint32_t weight = static_cast<uint32_t>(packed >> 32);
        const uint32_t key = ordered_float_key(scores[index]);
        if ((key & prefix_mask) == threshold_key) {
          const int bin = static_cast<int>((key >> shift) & 0xffu);
          atomicAdd(&histogram[bin], weight);
        }
      }
#endif
      __syncthreads();

      if (tid < 32) {
        uint32_t above = 0;
        const int chosen = warp_threshold_bin(
            histogram, static_cast<uint32_t>(remaining), &above);
        if (tid == 0 && chosen >= 0) {
          remaining -= static_cast<int32_t>(above);
          const uint32_t byte_mask = 0xffu << shift;
          threshold_key = (threshold_key & ~byte_mask) |
                          (static_cast<uint32_t>(chosen) << shift);
          prefix_mask |= byte_mask;
        }
      }
      __syncthreads();
    }
  }

  if (tid == 0 && profile_stats != nullptr) {
    const int ratio =
        n > 0 ? min(kRatioBins - 1, candidate_count * 100 / n) : 0;
    atomicAdd(profile_stats + ratio, 1);
    atomicAdd(profile_stats + kStatCalls, 1);
    atomicAdd(profile_stats + kStatSumN, n);
    atomicAdd(profile_stats + kStatSumM, candidate_count);
    atomicMax(profile_stats + kStatMaxM, candidate_count);
  }
  __syncthreads();

  // Final logical-order scan. For score ties, equal_running is the total
  // clipped weight of earlier equal-score leaves, so only the first
  // `remaining` tied tokens are selected and the crossing leaf is partial.
  for (int base = 0; base < n; base += kThreads) {
    const int index = base + tid;
    int32_t length = 0;
    uint32_t key = 0;
    if (index < n) {
      int32_t first;
      length = clipped_leaf(starts[index], lengths[index], g, &first);
      if (length > 0) {
        key = ordered_float_key(scores[index]);
      }
    }

    if (tid == 0) {
      equal_stripe_base = equal_running;
    }
    __syncthreads();

    const int32_t equal_length =
        (!select_all && length > 0 && key == threshold_key) ? length : 0;
    int32_t equal_exclusive = 0;
    int32_t equal_aggregate = 0;
    LengthScan(scan_storage).ExclusiveSum(equal_length, equal_exclusive,
                                           equal_aggregate);
    __syncthreads();

    int32_t selected_length = 0;
    if (length > 0) {
      if (select_all || key > threshold_key) {
        selected_length = length;
      } else if (key == threshold_key) {
        selected_length =
            min(max(remaining - equal_stripe_base - equal_exclusive, 0),
                length);
      }
    }
    if (tid == 0) {
      equal_running += equal_aggregate;
    }
    __syncthreads();

    int32_t exclusive = 0;
    int32_t aggregate = 0;
    LengthScan(scan_storage).ExclusiveSum(selected_length, exclusive,
                                           aggregate);
    __syncthreads();
    const int32_t stripe_base = running;
    if (index < n) {
      selected_prefix[index] = stripe_base + exclusive + selected_length;
    }
    __syncthreads();
    if (tid == 0) {
      running += aggregate;
    }
    __syncthreads();
  }
}

__global__ void expand_selected_kernel(
    const int32_t *__restrict__ selected_prefix,
    const int32_t *__restrict__ starts, const int32_t *__restrict__ lengths,
    const int32_t *__restrict__ seq_len, const int32_t *__restrict__ num_leaves,
    int32_t *__restrict__ candidates, int32_t *__restrict__ count,
    int capacity, int budget, int sink, int tail) {
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  constexpr int kWarps = kThreads / 32;
  const int n = min(max(*num_leaves, 0), capacity);
  const int32_t seq = *seq_len;
  const Guard g = make_guard(seq, sink, tail);
  const int32_t sink_tokens = min(g.sink_end, budget);
  const int32_t leaf_tokens = n > 0 ? selected_prefix[n - 1] : 0;
  const int32_t leaf_end = min(sink_tokens + leaf_tokens, budget);
  const int32_t tail_tokens = max(seq - g.tail_start, 0);
  const int32_t total = min(leaf_end + tail_tokens, budget);

  // Guards are contiguous and can be written directly.
  for (int pos = tid; pos < sink_tokens; pos += kThreads) {
    candidates[pos] = pos;
  }
  const int32_t tail_take = max(total - leaf_end, 0);
  for (int pos = tid; pos < tail_take; pos += kThreads) {
    candidates[leaf_end + pos] = g.tail_start + pos;
  }
  for (int pos = total + tid; pos < budget; pos += kThreads) {
    candidates[pos] = -1;
  }

  // One warp owns each selected leaf interval. This reverses the old
  // 8192-times binary search: intervals write their known output ranges.
  // Each warp inspects 32 leaves per round with coalesced loads (most leaves
  // are not selected) and then expands the selected ones one at a time via
  // ballot, instead of one dependent global round trip per leaf per warp.
  for (int base = warp * 32; base < n; base += kWarps * 32) {
    const int leaf = base + lane;
    int32_t taken = 0;
    int32_t first = 0;
    int32_t output = 0;
    if (leaf < n) {
      const int32_t previous = leaf > 0 ? selected_prefix[leaf - 1] : 0;
      taken = selected_prefix[leaf] - previous;
      if (taken > 0) {
        clipped_leaf(starts[leaf], lengths[leaf], g, &first);
        output = sink_tokens + previous;
      }
    }
    unsigned pending = __ballot_sync(kFullWarp, taken > 0);
    while (pending != 0u) {
      const int src = __ffs(pending) - 1;
      pending &= pending - 1u;
      const int32_t s_first = __shfl_sync(kFullWarp, first, src);
      const int32_t s_taken = __shfl_sync(kFullWarp, taken, src);
      const int32_t s_output = __shfl_sync(kFullWarp, output, src);
      for (int j = lane; j < s_taken; j += 32) {
        candidates[s_output + j] = s_first + j;
      }
    }
  }
  if (tid == 0) {
    *count = total;
  }
}

// Interval expansion runs in one CTA: pack contiguous (start, length,
// output_offset) records for sink + selected leaves + tail, then expand
// token IDs with the same contract as expand_selected_kernel.
__global__ __launch_bounds__(kThreads) void expand_intervals_kernel(
    const int32_t *__restrict__ selected_prefix,
    const int32_t *__restrict__ starts, const int32_t *__restrict__ lengths,
    const int32_t *__restrict__ seq_len, const int32_t *__restrict__ num_leaves,
    int32_t *__restrict__ candidates, int32_t *__restrict__ count,
    int32_t *__restrict__ seg_starts, int32_t *__restrict__ seg_lengths,
    int32_t *__restrict__ seg_offsets, int32_t *__restrict__ seg_count,
    int capacity, int budget, int sink, int tail) {
  const int tid = threadIdx.x;
  const int n = min(max(*num_leaves, 0), capacity);
  const int32_t seq = *seq_len;
  const Guard g = make_guard(seq, sink, tail);
  const int32_t sink_tokens = min(g.sink_end, budget);
  const int32_t leaf_tokens = n > 0 ? selected_prefix[n - 1] : 0;
  const int32_t leaf_end = min(sink_tokens + leaf_tokens, budget);
  const int32_t tail_tokens = max(seq - g.tail_start, 0);
  const int32_t total = min(leaf_end + tail_tokens, budget);

  __shared__ int32_t n_seg;
  if (tid == 0) {
    int32_t cursor = 0;
    if (sink_tokens > 0) {
      seg_starts[0] = 0;
      seg_lengths[0] = sink_tokens;
      seg_offsets[0] = 0;
      cursor = 1;
    }
    for (int i = 0; i < n; ++i) {
      const int32_t previous = i > 0 ? selected_prefix[i - 1] : 0;
      const int32_t taken = selected_prefix[i] - previous;
      if (taken <= 0) {
        continue;
      }
      int32_t first;
      clipped_leaf(starts[i], lengths[i], g, &first);
      seg_starts[cursor] = first;
      seg_lengths[cursor] = taken;
      seg_offsets[cursor] = sink_tokens + previous;
      ++cursor;
    }
    if (tail_tokens > 0 && leaf_end < budget) {
      const int32_t take = min(tail_tokens, budget - leaf_end);
      seg_starts[cursor] = g.tail_start;
      seg_lengths[cursor] = take;
      seg_offsets[cursor] = leaf_end;
      ++cursor;
    }
    n_seg = cursor;
    *seg_count = cursor;
    *count = total;
  }
  __syncthreads();

  // Direct interval expansion: one warp writes each known contiguous range.
  const int warp = tid >> 5;
  const int lane = tid & 31;
  constexpr int kWarps = kThreads / 32;
  for (int seg = warp; seg < n_seg; seg += kWarps) {
    const int32_t start = seg_starts[seg];
    const int32_t length = seg_lengths[seg];
    const int32_t output = seg_offsets[seg];
    for (int j = lane; j < length; j += 32) {
      candidates[output + j] = start + j;
    }
  }
  for (int pos = total + tid; pos < budget; pos += kThreads) {
    candidates[pos] = -1;
  }
}

__global__ void record_fallback_profile_kernel(
    const int32_t *__restrict__ num_leaves,
    int32_t *__restrict__ profile_stats, int capacity) {
  if (threadIdx.x == 0) {
    const int n = min(max(*num_leaves, 0), capacity);
    atomicAdd(profile_stats + kStatCalls, 1);
    atomicAdd(profile_stats + kStatSumN, n);
    atomicAdd(profile_stats + kStatFallback, 1);
  }
}

} // namespace

static void launch_prefix(const at::Tensor &scores, const at::Tensor &lengths,
                          const at::Tensor &starts, const at::Tensor &num_leaves,
                          const at::Tensor &seq_len, at::Tensor &selected_prefix,
                          int capacity, int token_budget, int sink_tokens,
                          int tail_tokens, cudaStream_t stream,
                          bool force_fallback = false,
                          int32_t *profile_stats = nullptr) {
  if (!force_fallback && capacity <= kFastMaxLeaves) {
    weighted_prefix_fast_kernel<<<1, kThreads, 0, stream>>>(
        scores.data_ptr<float>(), lengths.data_ptr<int32_t>(),
        starts.data_ptr<int32_t>(), num_leaves.data_ptr<int32_t>(),
        seq_len.data_ptr<int32_t>(), selected_prefix.data_ptr<int32_t>(),
        profile_stats, capacity, token_budget, sink_tokens, tail_tokens);
  } else {
    weighted_prefix_fallback_kernel<<<1, kThreads, 0, stream>>>(
        scores.data_ptr<float>(), lengths.data_ptr<int32_t>(),
        starts.data_ptr<int32_t>(), num_leaves.data_ptr<int32_t>(),
        seq_len.data_ptr<int32_t>(), selected_prefix.data_ptr<int32_t>(),
        capacity, token_budget, sink_tokens, tail_tokens);
    if (profile_stats != nullptr) {
      record_fallback_profile_kernel<<<1, 1, 0, stream>>>(
          num_leaves.data_ptr<int32_t>(), profile_stats, capacity);
    }
  }
  TORCH_CHECK(cudaGetLastError() == cudaSuccess,
              "weighted prefix kernel launch failed");
}

static void weighted_select_impl(
    const at::Tensor &scores, const at::Tensor &lengths,
    const at::Tensor &starts, const at::Tensor &num_leaves,
    const at::Tensor &seq_len, at::Tensor &selected_prefix,
    at::Tensor &candidates, at::Tensor &count, int64_t budget, int64_t sink,
    int64_t tail, bool force_fallback) {
  TORCH_CHECK(scores.is_cuda() && lengths.is_cuda() && starts.is_cuda(),
              "selector inputs must be CUDA");
  TORCH_CHECK(num_leaves.is_cuda() && seq_len.is_cuda(),
              "selector scalars must be CUDA");
  TORCH_CHECK(scores.scalar_type() == at::kFloat, "scores must be f32");
  TORCH_CHECK(lengths.scalar_type() == at::kInt &&
                  starts.scalar_type() == at::kInt &&
                  num_leaves.scalar_type() == at::kInt &&
                  seq_len.scalar_type() == at::kInt,
              "selector metadata must be i32");
  TORCH_CHECK(scores.is_contiguous() && lengths.is_contiguous() &&
                  starts.is_contiguous() && selected_prefix.is_contiguous(),
              "selector vectors must be contiguous");
  TORCH_CHECK(scores.numel() == lengths.numel() &&
                  scores.numel() == starts.numel() &&
                  scores.numel() == selected_prefix.numel(),
              "selector vectors must have the same capacity");
  TORCH_CHECK(scores.numel() > 0 && scores.numel() <= 65535,
              "selector capacity must be in [1, 65535]");
  TORCH_CHECK(budget > 0 && candidates.numel() == budget,
              "candidate output must match budget");
  TORCH_CHECK(sink >= 0 && tail >= 0 && sink + tail <= budget,
              "sink + tail guards must fit the budget");
  TORCH_CHECK(candidates.scalar_type() == at::kInt &&
                  count.scalar_type() == at::kInt,
              "selector outputs must be i32");

  const int capacity = static_cast<int>(scores.numel());
  const int token_budget = static_cast<int>(budget);
  const int sink_tokens = static_cast<int>(sink);
  const int tail_tokens = static_cast<int>(tail);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  launch_prefix(scores, lengths, starts, num_leaves, seq_len, selected_prefix,
                capacity, token_budget, sink_tokens, tail_tokens, stream,
                force_fallback);
  expand_selected_kernel<<<1, kThreads, 0, stream>>>(
      selected_prefix.data_ptr<int32_t>(), starts.data_ptr<int32_t>(),
      lengths.data_ptr<int32_t>(), seq_len.data_ptr<int32_t>(),
      num_leaves.data_ptr<int32_t>(), candidates.data_ptr<int32_t>(),
      count.data_ptr<int32_t>(), capacity, token_budget, sink_tokens,
      tail_tokens);
  TORCH_CHECK(cudaGetLastError() == cudaSuccess,
              "expand_selected_kernel launch failed");
}

void weighted_select_interface(
    const at::Tensor &scores, const at::Tensor &lengths,
    const at::Tensor &starts, const at::Tensor &num_leaves,
    const at::Tensor &seq_len, at::Tensor &selected_prefix,
    at::Tensor &candidates, at::Tensor &count, int64_t budget, int64_t sink,
    int64_t tail) {
  weighted_select_impl(scores, lengths, starts, num_leaves, seq_len,
                       selected_prefix, candidates, count, budget, sink, tail,
                       false);
}

void weighted_select_legacy_interface(
    const at::Tensor &scores, const at::Tensor &lengths,
    const at::Tensor &starts, const at::Tensor &num_leaves,
    const at::Tensor &seq_len, at::Tensor &selected_prefix,
    at::Tensor &candidates, at::Tensor &count, int64_t budget, int64_t sink,
    int64_t tail) {
  weighted_select_impl(scores, lengths, starts, num_leaves, seq_len,
                       selected_prefix, candidates, count, budget, sink, tail,
                       true);
}

void weighted_prefix_profile_interface(
    const at::Tensor &scores, const at::Tensor &lengths,
    const at::Tensor &starts, const at::Tensor &num_leaves,
    const at::Tensor &seq_len, at::Tensor &selected_prefix,
    at::Tensor &profile_stats, int64_t budget, int64_t sink, int64_t tail) {
  TORCH_CHECK(scores.is_cuda() && lengths.is_cuda() && starts.is_cuda() &&
                  num_leaves.is_cuda() && seq_len.is_cuda() &&
                  selected_prefix.is_cuda() && profile_stats.is_cuda(),
              "profile selector tensors must be CUDA");
  TORCH_CHECK(scores.scalar_type() == at::kFloat &&
                  lengths.scalar_type() == at::kInt &&
                  starts.scalar_type() == at::kInt &&
                  num_leaves.scalar_type() == at::kInt &&
                  seq_len.scalar_type() == at::kInt &&
                  selected_prefix.scalar_type() == at::kInt &&
                  profile_stats.scalar_type() == at::kInt,
              "profile selector dtypes must be f32/i32");
  TORCH_CHECK(scores.is_contiguous() && lengths.is_contiguous() &&
                  starts.is_contiguous() && selected_prefix.is_contiguous() &&
                  profile_stats.is_contiguous(),
              "profile selector tensors must be contiguous");
  TORCH_CHECK(scores.numel() == lengths.numel() &&
                  scores.numel() == starts.numel() &&
                  scores.numel() == selected_prefix.numel(),
              "profile selector vectors must have the same capacity");
  TORCH_CHECK(profile_stats.numel() >= kStatsSize,
              "profile_stats must contain at least 106 int32 values");
  const int capacity = static_cast<int>(scores.numel());
  TORCH_CHECK(capacity > 0 && capacity <= 65535,
              "selector capacity must be in [1, 65535]");
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  launch_prefix(
      scores, lengths, starts, num_leaves, seq_len, selected_prefix, capacity,
      static_cast<int>(budget), static_cast<int>(sink),
      static_cast<int>(tail), stream, false,
      profile_stats.data_ptr<int32_t>());
}

void expand_selected_interface(
    const at::Tensor &selected_prefix, const at::Tensor &starts,
    const at::Tensor &lengths, const at::Tensor &seq_len,
    const at::Tensor &num_leaves, at::Tensor &candidates, at::Tensor &count,
    int64_t budget, int64_t sink, int64_t tail) {
  TORCH_CHECK(selected_prefix.is_cuda() && starts.is_cuda() &&
                  lengths.is_cuda() && seq_len.is_cuda() &&
                  num_leaves.is_cuda() && candidates.is_cuda() &&
                  count.is_cuda(),
              "expansion tensors must be CUDA");
  TORCH_CHECK(selected_prefix.scalar_type() == at::kInt &&
                  starts.scalar_type() == at::kInt &&
                  lengths.scalar_type() == at::kInt &&
                  seq_len.scalar_type() == at::kInt &&
                  num_leaves.scalar_type() == at::kInt &&
                  candidates.scalar_type() == at::kInt &&
                  count.scalar_type() == at::kInt,
              "expansion tensors must be i32");
  TORCH_CHECK(selected_prefix.numel() == starts.numel() &&
                  starts.numel() == lengths.numel(),
              "expansion vectors must have the same capacity");
  TORCH_CHECK(candidates.numel() == budget,
              "candidate output must match budget");
  const int capacity = static_cast<int>(starts.numel());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  expand_selected_kernel<<<1, kThreads, 0, stream>>>(
      selected_prefix.data_ptr<int32_t>(), starts.data_ptr<int32_t>(),
      lengths.data_ptr<int32_t>(), seq_len.data_ptr<int32_t>(),
      num_leaves.data_ptr<int32_t>(), candidates.data_ptr<int32_t>(),
      count.data_ptr<int32_t>(), capacity, static_cast<int>(budget),
      static_cast<int>(sink), static_cast<int>(tail));
  TORCH_CHECK(cudaGetLastError() == cudaSuccess,
              "expand_selected_kernel launch failed");
}

void weighted_select_intervals_interface(
    const at::Tensor &scores, const at::Tensor &lengths,
    const at::Tensor &starts, const at::Tensor &num_leaves,
    const at::Tensor &seq_len, at::Tensor &selected_prefix,
    at::Tensor &candidates, at::Tensor &count, at::Tensor &seg_starts,
    at::Tensor &seg_lengths, at::Tensor &seg_offsets, at::Tensor &seg_count,
    int64_t budget, int64_t sink, int64_t tail) {
  TORCH_CHECK(scores.is_cuda() && lengths.is_cuda() && starts.is_cuda(),
              "selector inputs must be CUDA");
  TORCH_CHECK(num_leaves.is_cuda() && seq_len.is_cuda(),
              "selector scalars must be CUDA");
  TORCH_CHECK(scores.scalar_type() == at::kFloat, "scores must be f32");
  TORCH_CHECK(seg_starts.numel() >= scores.numel() + 2 &&
                  seg_lengths.numel() == seg_starts.numel() &&
                  seg_offsets.numel() == seg_starts.numel(),
              "segment buffers must hold capacity+2 rows");
  TORCH_CHECK(seg_starts.scalar_type() == at::kInt &&
                  seg_lengths.scalar_type() == at::kInt &&
                  seg_offsets.scalar_type() == at::kInt &&
                  seg_count.scalar_type() == at::kInt,
              "segment outputs must be i32");
  const int capacity = static_cast<int>(scores.numel());
  const int token_budget = static_cast<int>(budget);
  const int sink_tokens = static_cast<int>(sink);
  const int tail_tokens = static_cast<int>(tail);
  TORCH_CHECK(token_budget > 0 && candidates.numel() == budget,
              "candidate output must match budget");
  TORCH_CHECK(sink >= 0 && tail >= 0 && sink + tail <= budget,
              "sink + tail guards must fit the budget");
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  launch_prefix(scores, lengths, starts, num_leaves, seq_len, selected_prefix,
                capacity, token_budget, sink_tokens, tail_tokens, stream);
  expand_intervals_kernel<<<1, kThreads, 0, stream>>>(
      selected_prefix.data_ptr<int32_t>(), starts.data_ptr<int32_t>(),
      lengths.data_ptr<int32_t>(), seq_len.data_ptr<int32_t>(),
      num_leaves.data_ptr<int32_t>(), candidates.data_ptr<int32_t>(),
      count.data_ptr<int32_t>(), seg_starts.data_ptr<int32_t>(),
      seg_lengths.data_ptr<int32_t>(), seg_offsets.data_ptr<int32_t>(),
      seg_count.data_ptr<int32_t>(), capacity, token_budget, sink_tokens,
      tail_tokens);
  TORCH_CHECK(cudaGetLastError() == cudaSuccess,
              "expand_intervals_kernel launch failed");
}

TORCH_LIBRARY(adaptive_hisa_select, m) {
  m.def("weighted_select(Tensor scores, Tensor lengths, Tensor starts, "
        "Tensor num_leaves, Tensor seq_len, "
        "Tensor(a!) selected_prefix, Tensor(b!) candidates, "
        "Tensor(c!) count, int budget, int sink, int tail) -> ()");
  m.def("weighted_select_legacy(Tensor scores, Tensor lengths, Tensor starts, "
        "Tensor num_leaves, Tensor seq_len, "
        "Tensor(a!) selected_prefix, Tensor(b!) candidates, "
        "Tensor(c!) count, int budget, int sink, int tail) -> ()");
  m.def("weighted_prefix_profile(Tensor scores, Tensor lengths, Tensor starts, "
        "Tensor num_leaves, Tensor seq_len, Tensor(a!) selected_prefix, "
        "Tensor(b!) profile_stats, int budget, int sink, int tail) -> ()");
  m.def("expand_selected(Tensor selected_prefix, Tensor starts, Tensor lengths, "
        "Tensor seq_len, Tensor num_leaves, Tensor(a!) candidates, "
        "Tensor(b!) count, int budget, int sink, int tail) -> ()");
  m.def("weighted_select_intervals(Tensor scores, Tensor lengths, Tensor starts, "
        "Tensor num_leaves, Tensor seq_len, "
        "Tensor(a!) selected_prefix, Tensor(b!) candidates, "
        "Tensor(c!) count, Tensor(d!) seg_starts, Tensor(e!) seg_lengths, "
        "Tensor(f!) seg_offsets, Tensor(g!) seg_count, "
        "int budget, int sink, int tail) -> ()");
}

TORCH_LIBRARY_IMPL(adaptive_hisa_select, CUDA, m) {
  m.impl("weighted_select", weighted_select_interface);
  m.impl("weighted_select_legacy", weighted_select_legacy_interface);
  m.impl("weighted_prefix_profile", weighted_prefix_profile_interface);
  m.impl("expand_selected", expand_selected_interface);
  m.impl("weighted_select_intervals", weighted_select_intervals_interface);
}

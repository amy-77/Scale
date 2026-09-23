/**
 * Exact k-th selection for the P-key partition (split repair + target merge).
 *
 * Both selections previously ran a full ``torch.sort`` only to read one
 * threshold: the k cheapest merge edges (fp64 Ward cost, +inf = ineligible)
 * and the ``deficit`` best split candidates (lexicographic int64 staircase
 * keys). Sorting writes the permuted keys (and indices) back to HBM several
 * times; radix *selection* reads the keys once per 11-bit digit, keeps every
 * histogram in shared memory and writes a single scalar (or one byte per
 * element for the mask variant).
 *
 * Semantics are exact and identical to the sort-based code:
 *   kth_threshold_f64: successor of the k-th smallest value (``cost < thr``
 *                      admits the k cheapest incl. boundary ties), -inf for
 *                      k <= 0, k clamped to n.
 *   lex_select_mask:   mask[i] = valid[i] && key(i) <=_lex T where T is the
 *                      k-th smallest valid key tuple; all-zero for k <= 0.
 *
 * ``k`` is a device scalar: no host sync, CUDA-graph capturable. Small inputs
 * (n <= kSmallMax) run all digit passes inside one block in a single launch;
 * larger inputs run one launch per digit with a last-block finaliser.
 *
 * Histogram atomics are warp-aggregated (``__match_any_sync``): real cost
 * arrays are heavily skewed (most high digits identical), which would
 * otherwise serialise every thread on one shared-memory bin.
 */
#include <ATen/cuda/CUDAContext.h>
#include <ATen/core/TensorBody.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/library.h>

#include <cstdint>
#include <cmath>
#include <optional>

namespace {

constexpr int kThreads = 1024;
constexpr int kDigitBits = 11;
constexpr int kRadix = 1 << kDigitBits;  // 2048 bins
constexpr int kBinsPerLane = kRadix / 32;
constexpr int kMaxKeys = 4;
constexpr int kSmallMax = 1 << 15;      // single-block path
constexpr int kItemsPerThread = 2;      // multi-block grid sizing
constexpr uint64_t kSignBit = 0x8000000000000000ull;

// scratch layout (int64 words): [0, kMaxKeys) selected keys, [kMaxKeys] k_rem,
// [kMaxKeys+1] block ticket, [kMaxKeys+2] valid total, then kRadix histogram.
constexpr int kStateSel = 0;
constexpr int kStateK = kMaxKeys;
constexpr int kStateTicket = kMaxKeys + 1;
constexpr int kStateTotal = kMaxKeys + 2;
constexpr int kStateHist = kMaxKeys + 3;
constexpr int kScratchWords = kStateHist + kRadix;  // 2055

__device__ __forceinline__ uint64_t ord_f64(double x) {
  uint64_t b = static_cast<uint64_t>(__double_as_longlong(x));
  return (b >> 63) ? ~b : (b | kSignBit);
}

__device__ __forceinline__ double unord_f64(uint64_t u) {
  uint64_t b = (u >> 63) ? (u & ~kSignBit) : ~u;
  return __longlong_as_double(static_cast<long long>(b));
}

struct F64Fetch {
  const double* v;
  __device__ __forceinline__ bool valid(int) const { return true; }
  __device__ __forceinline__ uint64_t key(int i, int) const { return ord_f64(v[i]); }
};

struct LexFetch {
  const long long* keys;
  const uint8_t* mask;  // may be nullptr
  long stride_n;
  long stride_k;
  bool flip;  // key_bits == 64: arbitrary sign, flip the sign bit; else keys >= 0
  __device__ __forceinline__ bool valid(int i) const { return mask == nullptr || mask[i] != 0; }
  __device__ __forceinline__ uint64_t key(int i, int j) const {
    const uint64_t u = static_cast<uint64_t>(keys[i * stride_n + j * stride_k]);
    return flip ? (u ^ kSignBit) : u;
  }
};

__host__ __device__ __forceinline__ int first_shift_for(int key_bits) {
  const int passes = (key_bits + kDigitBits - 1) / kDigitBits;
  return (passes - 1) * kDigitBits;
}

// Does element i take part in the digit pass of key j at bit offset `shift`?
template <typename Fetch>
__device__ __forceinline__ bool participates(
    const Fetch& f, int i, int j, const uint64_t* sel, uint64_t prefix, int shift) {
  if (!f.valid(i)) return false;
  for (int jj = 0; jj < j; ++jj) {
    if (f.key(i, jj) != sel[jj]) return false;
  }
  const int hi = shift + kDigitBits;
  const uint64_t high_mask = hi >= 64 ? 0ull : (~0ull << hi);
  return ((f.key(i, j) ^ prefix) & high_mask) == 0;
}

// Warp-aggregated shared-memory histogram over [base0, n) with `stride`.
template <typename Fetch>
__device__ __forceinline__ void warp_histogram(
    const Fetch& f, int n, int j, const uint64_t* sel, uint64_t prefix, int shift,
    int base0, int stride, uint32_t* hist) {
  const int lane = threadIdx.x & 31;
  for (int base = base0; base < n; base += stride) {
    const int i = base + lane;
    const bool active = i < n && participates(f, i, j, sel, prefix, shift);
    const int bin = active ? static_cast<int>((f.key(i, j) >> shift) & (kRadix - 1)) : 0;
    const unsigned mask = __ballot_sync(0xffffffffu, active);
    if (active) {
      const unsigned peers = __match_any_sync(mask, bin);
      if (lane == __ffs(peers) - 1) atomicAdd(&hist[bin], static_cast<uint32_t>(__popc(peers)));
    }
  }
}

// Warp 0 scans the histogram and picks the digit holding k_rem (1-based).
// k_rem becomes relative to the chosen bin. `total_out` (optional) gets the
// number of participating elements (used by the first pass to clamp k).
template <typename T>
__device__ __forceinline__ void pick_digit(
    const T* hist, long long* k_rem, int* bin_out, long long* total_out) {
  const int lane = threadIdx.x & 31;
  const T* mine = hist + lane * kBinsPerLane;
  unsigned long long sum = 0;
  for (int t = 0; t < kBinsPerLane; ++t) sum += mine[t];  // no register array: avoid spills
  unsigned long long incl = sum;
#pragma unroll
  for (int off = 1; off < 32; off <<= 1) {
    unsigned long long up = __shfl_up_sync(0xffffffffu, incl, off);
    if (lane >= off) incl += up;
  }
  const unsigned long long excl = incl - sum;
  const unsigned long long total = __shfl_sync(0xffffffffu, incl, 31);
  long long k = *k_rem;
  __syncwarp();  // every lane has read k_rem before the owner lane rewrites it
  if (total_out != nullptr && lane == 0) *total_out = static_cast<long long>(total);
  if (k < 1) k = 1;
  if (k > static_cast<long long>(total)) k = static_cast<long long>(total);
  const unsigned long long ku = static_cast<unsigned long long>(k);
  if (ku > excl && ku <= incl) {
    unsigned long long run = excl;
    for (int t = 0; t < kBinsPerLane; ++t) {
      const unsigned long long c = mine[t];
      if (ku > run && ku <= run + c) {
        *bin_out = lane * kBinsPerLane + t;
        *k_rem = k - static_cast<long long>(run);
        break;
      }
      run += c;
    }
  }
}

template <typename Fetch>
__device__ __forceinline__ bool lex_le(const Fetch& f, int i, int n_keys, const uint64_t* sel) {
  for (int j = 0; j < n_keys; ++j) {
    const uint64_t kj = f.key(i, j);
    if (kj < sel[j]) return true;
    if (kj > sel[j]) return false;
  }
  return true;
}

// ---------------------------------------------------------------------------
// Small path: every digit of every key inside one block, one launch.
// ---------------------------------------------------------------------------
template <typename Fetch>
__device__ __forceinline__ void select_block(
    const Fetch& f, int n, int n_keys, int key_bits, long long k_in,
    uint64_t* sel_out, bool* nonempty_out) {
  __shared__ uint32_t hist[kRadix];
  __shared__ uint64_t sel[kMaxKeys];
  __shared__ long long k_rem;
  __shared__ long long total;
  __shared__ int bin;
  if (threadIdx.x == 0) {
    k_rem = k_in;
    total = -1;
    for (int j = 0; j < kMaxKeys; ++j) sel[j] = 0;
  }
  __syncthreads();
  const int warp = threadIdx.x >> 5;
  const int first_shift = first_shift_for(key_bits);
  bool first = true;
  for (int j = 0; j < n_keys; ++j) {
    uint64_t prefix = 0;
    for (int shift = first_shift; shift >= 0; shift -= kDigitBits) {
      for (int b = threadIdx.x; b < kRadix; b += blockDim.x) hist[b] = 0;
      __syncthreads();
      warp_histogram(f, n, j, sel, prefix, shift, warp * 32, blockDim.x, hist);
      __syncthreads();
      if (threadIdx.x < 32) {
        if (threadIdx.x == 0) bin = 0;
        __syncwarp();
        pick_digit(hist, &k_rem, &bin, first ? &total : nullptr);
      }
      __syncthreads();
      prefix |= static_cast<uint64_t>(bin) << shift;
      if (threadIdx.x == 0) sel[j] = prefix;
      first = false;
      __syncthreads();
    }
  }
  for (int j = 0; j < kMaxKeys; ++j) sel_out[j] = sel[j];
  *nonempty_out = total > 0;
}

__global__ __launch_bounds__(kThreads) void kth_f64_small_kernel(
    const double* __restrict__ values, const long long* __restrict__ k_ptr,
    double* __restrict__ thr_out, int n) {
  F64Fetch f{values};
  const long long k = *k_ptr;
  if (k <= 0) {  // idle merge round: answer is -inf, skip the digit passes
    if (threadIdx.x == 0) *thr_out = -INFINITY;
    return;
  }
  uint64_t sel[kMaxKeys];
  bool nonempty;
  select_block(f, n, 1, 64, k, sel, &nonempty);
  if (threadIdx.x == 0) {
    double thr = -INFINITY;
    if (nonempty) thr = nextafter(unord_f64(sel[0]), INFINITY);
    *thr_out = thr;
  }
}

__global__ __launch_bounds__(kThreads) void lex_small_kernel(
    const long long* __restrict__ keys, const uint8_t* __restrict__ valid,
    const long long* __restrict__ k_ptr, uint8_t* __restrict__ mask_out,
    int n, int n_keys, int key_bits, long stride_n, long stride_k) {
  LexFetch f{keys, valid, stride_n, stride_k, key_bits == 64};
  const long long k = *k_ptr;
  if (k <= 0) {  // nothing to pop: all-zero mask without the digit passes
    for (int i = threadIdx.x; i < n; i += blockDim.x) mask_out[i] = 0;
    return;
  }
  uint64_t sel[kMaxKeys];
  bool nonempty;
  select_block(f, n, n_keys, key_bits, k, sel, &nonempty);
  const bool any = nonempty;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    mask_out[i] = (any && f.valid(i) && lex_le(f, i, n_keys, sel)) ? 1 : 0;
  }
}

// ---------------------------------------------------------------------------
// Large path: one launch per digit; the last block to finish finalises.
// ---------------------------------------------------------------------------
template <typename Fetch>
__device__ __forceinline__ void pass_grid(
    const Fetch& f, int n, int j, int shift, bool first,
    const long long* __restrict__ k_ptr, long long* __restrict__ state) {
  __shared__ uint32_t hist[kRadix];
  __shared__ uint64_t sel[kMaxKeys];
  __shared__ uint64_t prefix_sh;
  __shared__ int ticket;
  __shared__ long long k_rem;
  __shared__ int bin;
  for (int b = threadIdx.x; b < kRadix; b += blockDim.x) hist[b] = 0;
  if (threadIdx.x == 0) {
    for (int jj = 0; jj < kMaxKeys; ++jj) sel[jj] = static_cast<uint64_t>(__ldcg(&state[kStateSel + jj]));
    prefix_sh = static_cast<uint64_t>(__ldcg(&state[kStateSel + j]));
  }
  __syncthreads();
  const uint64_t prefix = prefix_sh;
  const int warp = threadIdx.x >> 5;
  const int stride = gridDim.x * blockDim.x;
  warp_histogram(f, n, j, sel, prefix, shift, blockIdx.x * blockDim.x + warp * 32, stride, hist);
  __syncthreads();
  unsigned long long* ghist = reinterpret_cast<unsigned long long*>(state + kStateHist);
  for (int b = threadIdx.x; b < kRadix; b += blockDim.x) {
    if (hist[b] != 0) atomicAdd(&ghist[b], static_cast<unsigned long long>(hist[b]));
  }
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) {
    ticket = static_cast<int>(atomicAdd(reinterpret_cast<unsigned long long*>(&state[kStateTicket]), 1ull));
  }
  __syncthreads();
  if (ticket != static_cast<int>(gridDim.x) - 1) return;
  __threadfence();
  // last block: pull the global histogram, pick the digit, reset scratch
  __shared__ unsigned long long ghist_sh[kRadix];
  for (int b = threadIdx.x; b < kRadix; b += blockDim.x) ghist_sh[b] = __ldcg(&ghist[b]);
  __syncthreads();
  if (threadIdx.x < 32) {
    if (threadIdx.x == 0) {
      k_rem = first ? *k_ptr : __ldcg(&state[kStateK]);
      bin = 0;
    }
    __syncwarp();
    long long total_tmp = 0;
    pick_digit(ghist_sh, &k_rem, &bin, first ? &total_tmp : nullptr);
    __syncwarp();
    if (threadIdx.x == 0) {
      if (first) state[kStateTotal] = total_tmp;
      state[kStateK] = k_rem;
      state[kStateSel + j] = static_cast<long long>(prefix | (static_cast<uint64_t>(bin) << shift));
      state[kStateTicket] = 0;
    }
  }
  __syncthreads();
  for (int b = threadIdx.x; b < kRadix; b += blockDim.x) ghist[b] = 0ull;
  __threadfence();
}

__global__ __launch_bounds__(kThreads) void kth_f64_pass_kernel(
    const double* __restrict__ values, const long long* __restrict__ k_ptr,
    long long* __restrict__ state, int n, int shift, int first) {
  if (*k_ptr <= 0) return;  // finish kernel emits -inf for k <= 0 regardless of state
  F64Fetch f{values};
  pass_grid(f, n, 0, shift, first != 0, k_ptr, state);
}

__global__ void kth_f64_finish_kernel(
    const long long* __restrict__ k_ptr, const long long* __restrict__ state,
    double* __restrict__ thr_out) {
  const long long k = *k_ptr;
  const long long total = state[kStateTotal];
  double thr = -INFINITY;
  if (k > 0 && total > 0) {
    thr = nextafter(unord_f64(static_cast<uint64_t>(state[kStateSel])), INFINITY);
  }
  *thr_out = thr;
}

__global__ __launch_bounds__(kThreads) void lex_pass_kernel(
    const long long* __restrict__ keys, const uint8_t* __restrict__ valid,
    const long long* __restrict__ k_ptr, long long* __restrict__ state,
    int n, int j, int shift, int first, long stride_n, long stride_k, int flip) {
  if (*k_ptr <= 0) return;  // mask kernel emits all-zero for k <= 0 regardless of state
  LexFetch f{keys, valid, stride_n, stride_k, flip != 0};
  pass_grid(f, n, j, shift, first != 0, k_ptr, state);
}

__global__ __launch_bounds__(kThreads) void lex_mask_kernel(
    const long long* __restrict__ keys, const uint8_t* __restrict__ valid,
    const long long* __restrict__ k_ptr, const long long* __restrict__ state,
    uint8_t* __restrict__ mask_out, int n, int n_keys, long stride_n, long stride_k, int flip) {
  LexFetch f{keys, valid, stride_n, stride_k, flip != 0};
  __shared__ uint64_t sel[kMaxKeys];
  __shared__ bool any;
  if (threadIdx.x == 0) {
    for (int j = 0; j < kMaxKeys; ++j) sel[j] = static_cast<uint64_t>(state[kStateSel + j]);
    any = (*k_ptr > 0) && (state[kStateTotal] > 0);
  }
  __syncthreads();
  const int stride = gridDim.x * blockDim.x;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
    mask_out[i] = (any && f.valid(i) && lex_le(f, i, n_keys, sel)) ? 1 : 0;
  }
}

int grid_for(int n) {
  int g = (n + kThreads * kItemsPerThread - 1) / (kThreads * kItemsPerThread);
  return g < 1 ? 1 : g;
}

void check_scratch(const at::Tensor& scratch) {
  TORCH_CHECK(scratch.is_cuda() && scratch.scalar_type() == at::kLong && scratch.is_contiguous());
  TORCH_CHECK(scratch.numel() >= kScratchWords, "radix_select scratch too small");
}

// thr_out[0] = successor of k-th smallest (k clamped to [1, n]); -inf if k <= 0.
void kth_threshold_f64(
    const at::Tensor& values, const at::Tensor& k, at::Tensor& thr_out, at::Tensor& scratch) {
  TORCH_CHECK(values.is_cuda() && values.scalar_type() == at::kDouble && values.dim() == 1 && values.is_contiguous());
  TORCH_CHECK(k.is_cuda() && k.scalar_type() == at::kLong && k.numel() == 1);
  TORCH_CHECK(thr_out.is_cuda() && thr_out.scalar_type() == at::kDouble && thr_out.numel() == 1);
  const int n = static_cast<int>(values.numel());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  const double* v = values.data_ptr<double>();
  const long long* kp = reinterpret_cast<const long long*>(k.data_ptr<int64_t>());
  double* out = thr_out.data_ptr<double>();
  if (n <= kSmallMax) {
    kth_f64_small_kernel<<<1, kThreads, 0, stream>>>(v, kp, out, n);
    return;
  }
  check_scratch(scratch);
  long long* state = reinterpret_cast<long long*>(scratch.data_ptr<int64_t>());
  const int grid = grid_for(n);
  bool first = true;
  for (int shift = first_shift_for(64); shift >= 0; shift -= kDigitBits) {
    kth_f64_pass_kernel<<<grid, kThreads, 0, stream>>>(v, kp, state, n, shift, first ? 1 : 0);
    first = false;
  }
  kth_f64_finish_kernel<<<1, 1, 0, stream>>>(kp, state, out);
}

// mask_out[i] = valid[i] && keys[i] <=_lex (k-th smallest valid tuple); 0 if k <= 0.
void lex_select_mask(
    const at::Tensor& keys, const std::optional<at::Tensor>& valid, const at::Tensor& k,
    at::Tensor& mask_out, at::Tensor& scratch, int64_t key_bits) {
  TORCH_CHECK(keys.is_cuda() && keys.scalar_type() == at::kLong && keys.dim() == 2);
  const int n = static_cast<int>(keys.size(0));
  const int n_keys = static_cast<int>(keys.size(1));
  TORCH_CHECK(n_keys >= 1 && n_keys <= kMaxKeys, "lex_select_mask supports 1..4 keys");
  TORCH_CHECK(key_bits >= 1 && key_bits <= 64);
  TORCH_CHECK(k.is_cuda() && k.scalar_type() == at::kLong && k.numel() == 1);
  TORCH_CHECK(mask_out.is_cuda() && mask_out.scalar_type() == at::kByte && mask_out.numel() == n && mask_out.is_contiguous());
  const uint8_t* vptr = nullptr;
  if (valid.has_value()) {
    TORCH_CHECK(valid->is_cuda() && valid->numel() == n && valid->is_contiguous());
    TORCH_CHECK(valid->scalar_type() == at::kByte || valid->scalar_type() == at::kBool);
    vptr = static_cast<const uint8_t*>(valid->data_ptr());
  }
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  const long long* kp_keys = reinterpret_cast<const long long*>(keys.data_ptr<int64_t>());
  const long long* kp = reinterpret_cast<const long long*>(k.data_ptr<int64_t>());
  uint8_t* out = mask_out.data_ptr<uint8_t>();
  const long sn = static_cast<long>(keys.stride(0));
  const long sk = static_cast<long>(keys.stride(1));
  const int kb = static_cast<int>(key_bits);
  if (n <= kSmallMax) {
    lex_small_kernel<<<1, kThreads, 0, stream>>>(kp_keys, vptr, kp, out, n, n_keys, kb, sn, sk);
    return;
  }
  check_scratch(scratch);
  long long* state = reinterpret_cast<long long*>(scratch.data_ptr<int64_t>());
  const int grid = grid_for(n);
  const int first_shift = first_shift_for(kb);
  const int flip = kb == 64 ? 1 : 0;
  bool first = true;
  for (int j = 0; j < n_keys; ++j) {
    for (int shift = first_shift; shift >= 0; shift -= kDigitBits) {
      lex_pass_kernel<<<grid, kThreads, 0, stream>>>(
          kp_keys, vptr, kp, state, n, j, shift, first ? 1 : 0, sn, sk, flip);
      first = false;
    }
  }
  lex_mask_kernel<<<grid, kThreads, 0, stream>>>(kp_keys, vptr, kp, state, out, n, n_keys, sn, sk, flip);
}

}  // namespace

// Python mirrors kScratchWords (kMaxKeys + 3 + kRadix = 2055).
TORCH_LIBRARY(adaptive_hisa_radix_select, m) {
  m.def("kth_threshold_f64(Tensor values, Tensor k, Tensor(a!) thr_out, Tensor(b!) scratch) -> ()");
  m.def("lex_select_mask(Tensor keys, Tensor? valid, Tensor k, Tensor(a!) mask_out, "
        "Tensor(b!) scratch, int key_bits) -> ()");
}

TORCH_LIBRARY_IMPL(adaptive_hisa_radix_select, CUDA, m) {
  m.impl("kth_threshold_f64", kth_threshold_f64);
  m.impl("lex_select_mask", lex_select_mask);
}

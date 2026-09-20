#include <cuda_runtime.h>
#include <torch/extension.h>

#include <algorithm>
#include <cfloat>

// One block scores one (query row, leaf). q is [rows, H, D], mean is [leaves, D].
__global__ void score_leaves_kernel(
    const float* __restrict__ q,
    const float* __restrict__ weights,
    const float* __restrict__ mean,
    float* __restrict__ out,
    int rows,
    int leaves,
    int heads,
    int dim) {
    int leaf = blockIdx.x;
    int row = blockIdx.y;
    if (leaf >= leaves || row >= rows) {
        return;
    }
    float acc = 0.f;
    for (int head = threadIdx.x; head < heads; head += blockDim.x) {
        const float* qh = q + (static_cast<long long>(row) * heads + head) * dim;
        const float* mh = mean + static_cast<long long>(leaf) * dim;
        float dot = 0.f;
        for (int d = 0; d < dim; ++d) {
            dot += qh[d] * mh[d];
        }
        acc += weights[row * heads + head] * fmaxf(dot, 0.f);
    }
    __shared__ float buf[128];
    buf[threadIdx.x] = acc;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            buf[threadIdx.x] += buf[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        out[row * leaves + leaf] = buf[0];
    }
}

// Gather R logical tokens from the paged uint8 index buffer.
// buf: [pages, stride] uint8, stride = page_size * (dim + 4)
// token_ids: [R] int64 logical ids, -1 skipped
// page_table: [max_pages] int32
__global__ void gather_compact_kernel(
    const uint8_t* __restrict__ buf,
    const int64_t* __restrict__ token_ids,
    const int32_t* __restrict__ page_table,
    uint8_t* __restrict__ keys,
    float* __restrict__ scale,
    int count,
    int page_size,
    int dim,
    int stride) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) {
        return;
    }
    int64_t token = token_ids[i];
    if (token < 0) {
        return;
    }
    int logical_page = static_cast<int>(token / page_size);
    int off = static_cast<int>(token % page_size);
    int physical = page_table[logical_page];
    const uint8_t* page = buf + static_cast<long long>(physical) * stride;
    const uint8_t* src = page + off * dim;
    uint8_t* dst = keys + static_cast<long long>(i) * dim;
    for (int d = 0; d < dim; ++d) {
        dst[d] = src[d];
    }
    const float* scales = reinterpret_cast<const float*>(page + page_size * dim);
    scale[i] = scales[off];
}

std::vector<torch::Tensor> score_leaves(
    torch::Tensor q,
    torch::Tensor weights,
    torch::Tensor mean) {
    TORCH_CHECK(q.is_cuda() && mean.is_cuda() && weights.is_cuda(), "score_leaves tensors must be CUDA");
    TORCH_CHECK(q.dim() == 3 && mean.dim() == 2, "q [rows,heads,dim], mean [leaves,dim]");
    auto qq = q.contiguous().to(torch::kFloat);
    auto ww = weights.contiguous().to(torch::kFloat);
    auto mm = mean.contiguous().to(torch::kFloat);
    int rows = qq.size(0);
    int heads = qq.size(1);
    int dim = qq.size(2);
    int leaves = mm.size(0);
    auto out = torch::empty({rows, leaves}, qq.options());
    if (rows == 0 || leaves == 0) {
        return {out};
    }
    score_leaves_kernel<<<dim3(leaves, rows), 64>>>(
        qq.data_ptr<float>(),
        ww.data_ptr<float>(),
        mm.data_ptr<float>(),
        out.data_ptr<float>(),
        rows,
        leaves,
        heads,
        dim);
    return {out};
}

std::vector<torch::Tensor> gather_compact(
    torch::Tensor buf,
    torch::Tensor token_ids,
    torch::Tensor page_table,
    int64_t page_size,
    int64_t dim) {
    TORCH_CHECK(buf.is_cuda() && token_ids.is_cuda() && page_table.is_cuda(), "gather_compact requires CUDA");
    auto ids = token_ids.contiguous().to(torch::kLong);
    auto table = page_table.contiguous().to(torch::kInt);
    int count = ids.size(0);
    auto keys = torch::zeros({count, dim}, buf.options());
    auto scale = torch::zeros({count}, buf.options().dtype(torch::kFloat));
    if (count == 0) {
        return {keys, scale};
    }
    int threads = 128;
    int blocks = (count + threads - 1) / threads;
    gather_compact_kernel<<<blocks, threads>>>(
        buf.data_ptr<uint8_t>(),
        ids.data_ptr<int64_t>(),
        table.data_ptr<int>(),
        keys.data_ptr<uint8_t>(),
        scale.data_ptr<float>(),
        count,
        static_cast<int>(page_size),
        static_cast<int>(dim),
        static_cast<int>(buf.size(1)));
    return {keys, scale};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("score_leaves", &score_leaves, "Adaptive-HISA leaf scores");
    m.def("gather_compact", &gather_compact, "Adaptive-HISA compact K gather");
}

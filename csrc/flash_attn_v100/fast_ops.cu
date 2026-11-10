// ---------------------------------------------------------------------------
// fast_ops.cu — 训练链路上的融合小算子 (V100 / sm70)
//
// 目前包含:
//   rope_fwd / rope_bwd : 融合 RoPE。原实现每层要跑 neg*2 + cat*2 + mul*4 +
//                         add*2 + 若干 cast (~10 个 kernel/tensor)，这里 1 个。
//                         数学上与原 eager 路径**逐位等价**:
//                           out[d] = q[d]*cos[d] + rot(q)[d]*sin[d]
//                           rot(q)[d] = -q[d+32] (d<32), q[d-32] (d>=32)
//                         全部用 fp32 乘加 (显式 __fmul_rn/__fadd_rn 禁止 fma 收缩),
//                         最后一次性 __float2half_rn, 与 "fp32 计算 + .half()" 相同。
//
// 只支持 head_dim == 64 (模型固定 64)、最后一维连续、各维跨度偶数 (__half2 对齐)。
// cos/sin 为 fp32 [S,64]，行方向允许 padding (本模型的 cos_cached 是 [1,S,128]
// 切片出来的视图，行 stride = 128)。
// ---------------------------------------------------------------------------
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace fastops {

namespace {

constexpr int D_ROPE = 64;
constexpr int ROWS_PER_BLOCK = 4;
constexpr int NTHREADS_ROPE = 32 * ROWS_PER_BLOCK;

__device__ __forceinline__ void rope_row(
    const half* __restrict__ src, half* __restrict__ dst,
    const float* __restrict__ cs, const float* __restrict__ sn, int lane) {
    const int d = 2 * lane;
    // lane < 16 => d < 32, 伙伴在 +32, 符号 -1; 否则伙伴在 -32, 符号 +1
    const int dp = (lane < 16) ? (d + 32) : (d - 32);
    const float sgn = (lane < 16) ? -1.f : 1.f;

    const __half2 x = *reinterpret_cast<const __half2*>(src + d);
    const __half2 xp = *reinterpret_cast<const __half2*>(src + dp);
    const float c0 = cs[d], c1 = cs[d + 1];
    const float s0 = sn[d], s1 = sn[d + 1];

    const float o0 = __fadd_rn(__fmul_rn(__half2float(x.x), c0),
                               __fmul_rn(sgn * __half2float(xp.x), s0));
    const float o1 = __fadd_rn(__fmul_rn(__half2float(x.y), c1),
                               __fmul_rn(sgn * __half2float(xp.y), s1));
    *reinterpret_cast<__half2*>(dst + d) = __floats2half2_rn(o0, o1);
}

__device__ __forceinline__ void rope_row_bwd(
    const half* __restrict__ dsrc, half* __restrict__ ddst,
    const float* __restrict__ cs, const float* __restrict__ sn, int lane) {
    const int d = 2 * lane;
    const int dp = (lane < 16) ? (d + 32) : (d - 32);
    const float sgn = (lane < 16) ? 1.f : -1.f;

    const __half2 g = *reinterpret_cast<const __half2*>(dsrc + d);
    const __half2 gp = *reinterpret_cast<const __half2*>(dsrc + dp);
    const float c0 = cs[d], c1 = cs[d + 1];
    const float s0 = sn[dp], s1 = sn[dp + 1];

    const float o0 = __fadd_rn(__fmul_rn(__half2float(g.x), c0),
                               __fmul_rn(sgn * __half2float(gp.x), s0));
    const float o1 = __fadd_rn(__fmul_rn(__half2float(g.y), c1),
                               __fmul_rn(sgn * __half2float(gp.y), s1));
    *reinterpret_cast<__half2*>(ddst + d) = __floats2half2_rn(o0, o1);
}

template <bool BACKWARD>
__global__ void rope_kernel(const half* __restrict__ q, const half* __restrict__ k,
                            const float* __restrict__ cos, const float* __restrict__ sin,
                            half* __restrict__ qo, half* __restrict__ ko,
                            int S, int H,
                            long qsb, long qsh, long qss,
                            long ksb, long ksh, long kss,
                            long oqb, long oqh, long oqs,
                            long okb, long okh, long oks,
                            long cs0, long sn0) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int s = blockIdx.x * ROWS_PER_BLOCK + warp;
    if (s >= S) return;

    const long bh = blockIdx.y;
    const long b = bh / H, h = bh % H;
    const float* cs = cos + (long)s * cs0;
    const float* sn = sin + (long)s * sn0;

    const long qi = b * qsb + h * qsh + (long)s * qss;
    const long ki = b * ksb + h * ksh + (long)s * kss;
    const long qoi = b * oqb + h * oqh + (long)s * oqs;
    const long koi = b * okb + h * okh + (long)s * oks;
    if (BACKWARD) {
        rope_row_bwd(q + qi, qo + qoi, cs, sn, lane);
        rope_row_bwd(k + ki, ko + koi, cs, sn, lane);
    } else {
        rope_row(q + qi, qo + qoi, cs, sn, lane);
        rope_row(k + ki, ko + koi, cs, sn, lane);
    }
}

template <bool BACKWARD>
void launch(const torch::Tensor& q, const torch::Tensor& k,
            const torch::Tensor& cos, const torch::Tensor& sin,
            torch::Tensor& qo, torch::Tensor& ko) {
    const int B = q.size(0), H = q.size(1), S = q.size(2);
    if (B == 0 || H == 0 || S == 0) return;
    dim3 grid((S + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK, B * H);
    auto stream = at::cuda::getCurrentCUDAStream();
    rope_kernel<BACKWARD><<<grid, NTHREADS_ROPE, 0, stream>>>(
        reinterpret_cast<const half*>(q.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(k.data_ptr<at::Half>()),
        cos.data_ptr<float>(), sin.data_ptr<float>(),
        reinterpret_cast<half*>(qo.data_ptr<at::Half>()),
        reinterpret_cast<half*>(ko.data_ptr<at::Half>()),
        S, H,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        qo.stride(0), qo.stride(1), qo.stride(2),
        ko.stride(0), ko.stride(1), ko.stride(2),
        cos.stride(0), sin.stride(0));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

inline bool is_dense(at::IntArrayRef sizes, at::IntArrayRef strides) {
    long need = 1, numel = 1;
    for (size_t i = 0; i < sizes.size(); ++i) {
        numel *= sizes[i];
        need += (sizes[i] - 1) * strides[i];
    }
    return need == numel;
}

// eager 路径里 TensorIterator 会把输出分配成"和输入相同内存顺序的 dense 张量":
// 本模型 q/k 来自 fused-qkv 的 (B,S,H,D) 视图, 所以输出 stride 是
// (S*H*D, D, H*D, 1)。这里复刻这个选择 (直接用输入 stride 会在 fused-qkv
// 视图下分配 3 倍内存)。
// 注意: 输出 stride 只影响布局/内存, 不影响数值, 所以任何 dense 布局都安全;
// 这里只是为了和 eager 保持一致 + 不浪费内存。非 fused-qkv 顺序的视图退回
// 标准 (B,H,S,D) 连续布局。
inline std::vector<int64_t> out_strides(const torch::Tensor& x) {
    if (is_dense(x.sizes(), x.strides()))
        return x.strides().vec();
    const int64_t B = x.size(0), H = x.size(1), S = x.size(2), D = x.size(3);
    const bool fused_qkv_order = (x.stride(1) == D && x.stride(2) >= H * D);
    if (fused_qkv_order)
        return {S * H * D, D, H * D, 1};
    return {H * S * D, S * D, D, 1};
}

// kernel 用 __half2 成对读写 (d, d+1), 因此要求指针 4 字节对齐且各维跨度是偶数。
// 本模型的 fused-qkv 视图 (跨度 64 / 2304 / S*2304) 天然满足。
inline bool half2_aligned(const torch::Tensor& t) {
    if (reinterpret_cast<uintptr_t>(t.data_ptr()) % 4 != 0)
        return false;
    for (int d = 0; d < 3; ++d)
        if (t.stride(d) % 2 != 0)
            return false;
    return true;
}

inline void check(const torch::Tensor& q, const torch::Tensor& k,
                  const torch::Tensor& cos, const torch::Tensor& sin) {
    TORCH_CHECK(q.is_cuda() && k.is_cuda(), "rope: inputs must be CUDA");
    TORCH_CHECK(q.dtype() == torch::kHalf && k.dtype() == torch::kHalf, "rope: fp16 only");
    TORCH_CHECK(q.dim() == 4 && q.size(3) == D_ROPE, "rope: head_dim must be 64");
    TORCH_CHECK(k.is_cuda() && k.device() == q.device(), "rope: q/k device mismatch");
    TORCH_CHECK(q.sizes() == k.sizes(), "rope: q/k shape mismatch");
    TORCH_CHECK(q.stride(3) == 1 && k.stride(3) == 1, "rope: last dim must be contiguous");
    TORCH_CHECK(cos.is_cuda() && sin.is_cuda() && cos.dtype() == torch::kFloat
                    && sin.dtype() == torch::kFloat,
                "rope: cos/sin must be fp32");
    TORCH_CHECK(cos.device() == q.device() && sin.device() == q.device(),
                "rope: cos/sin must be on the same device as q/k");
    TORCH_CHECK(half2_aligned(q) && half2_aligned(k),
                "rope: q/k must be 4-byte aligned with even strides");
    // cos/sin 允许行方向有 padding (本模型的 cos_cached 是 [1,S,128] 切片出的
    // [S,64] 视图, 行 stride = 128), 只要能按 (stride0, 1) 寻址即可。
    TORCH_CHECK(cos.dim() == 2 && cos.size(0) == q.size(2) && cos.size(1) == D_ROPE,
                "rope: cos/sin must be [S, 64]");
    TORCH_CHECK(cos.stride(1) == 1 && sin.stride(1) == 1,
                "rope: cos/sin last dim must be contiguous");
    TORCH_CHECK(cos.sizes() == sin.sizes(), "rope: cos/sin shape mismatch");
}

}  // namespace

std::vector<torch::Tensor> rope_fwd(torch::Tensor q, torch::Tensor k,
                                    torch::Tensor cos, torch::Tensor sin) {
    check(q, k, cos, sin);
    // 输出布局与 eager 路径一致 (dense, 保持输入的内存顺序), 下游 FlashAttention
    // 可以零拷贝消费 (stride 天然 32B 对齐)。
    auto qo = torch::empty_strided(q.sizes(), out_strides(q), q.options());
    auto ko = torch::empty_strided(k.sizes(), out_strides(k), k.options());
    launch<false>(q, k, cos, sin, qo, ko);
    return {qo, ko};
}

std::vector<torch::Tensor> rope_bwd(torch::Tensor dqo, torch::Tensor dko,
                                    torch::Tensor cos, torch::Tensor sin,
                                    torch::Tensor q, torch::Tensor k) {
    check(q, k, cos, sin);
    TORCH_CHECK(dqo.sizes() == q.sizes() && dko.sizes() == k.sizes(), "rope_bwd: grad shape");
    TORCH_CHECK(dqo.is_cuda() && dko.is_cuda()
                    && dqo.dtype() == torch::kHalf && dko.dtype() == torch::kHalf
                    && dqo.dim() == 4 && dko.dim() == 4,
                "rope_bwd: grads must be CUDA fp16 4-D");
    TORCH_CHECK(dqo.stride(3) == 1 && dko.stride(3) == 1, "rope_bwd: grad last dim contiguous");
    TORCH_CHECK(half2_aligned(dqo) && half2_aligned(dko),
                "rope_bwd: grads must be 4-byte aligned with even strides");
    auto dq = torch::empty_strided(q.sizes(), out_strides(q), q.options());
    auto dk = torch::empty_strided(k.sizes(), out_strides(k), k.options());
    launch<true>(dqo, dko, cos, sin, dq, dk);
    return {dq, dk};
}

}  // namespace fastops

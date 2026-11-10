// FlashAttention-2 for NVIDIA V100 / SM70 (Volta tensor cores, HMMA.884)
// 专用实现: head_dim = 64, fp16, causal 可选, dropout 可选.
//
// 设计要点 (为什么这样写):
//   * nvcuda::wmma 在 sm70 上可用 (内部走 HMMA.884 张量核心), 而 Triton 在
//     Volta 上会退化成 FMA -> 这就是必须手写 CUDA 的原因.
//   * wmma 的 accumulator fragment 元素布局虽然文档写"未指定", 但在 sm70 上
//     实测是固定映射 (探针: /tmp/opencode/probe_layout.cu):
//         lane l 持有行 base 与 base+2, 其中
//         base = ((l>>2)&1)*8 + ((l>>4)&1)*4 + (l&1)
//         列   = c, c+1, c+4, c+5,  c = 2*((l>>1)&1) + 8*((l>>3)&1)
//     "S 的累加器" 和 "O 的累加器" 是同一个 mma shape, 所以行映射完全一致:
//     本 lane 算出的 softmax 行统计量就是本 lane 需要 rescale 的行, 无需任何
//     shared 广播, 也无需把 acc_o 写回 shared 做 rescale (这是性能关键).
//   * 因此 S 的 fp32 tile 不需要落到 shared: 直接在寄存器里 mask/scale/求
//     行最大值/exp, 再按上面的列映射写进 sP (fp16, 给 PV 的 mma 用).
//     shared 用量 25.3KB -> 3 CTA/SM, 寄存器 ~120.
//   * Q tile 先落到 shared 再 load fragment, 避免越界读和 32B 对齐问题.
//   * K/V/Q 全局加载用 uint4 (16B) 向量化.
//   * dropout mask 用 (seed,row,col,bh) 的纯函数 hash, 前向/反向完全一致, 不需要
//     保存 N x N mask, 也不需要 philox 状态.
//
// 实测 (B=3,H=12,S=4096,causal): mem_eff 2.74ms (28 TFLOP/s), 本 kernel 见 README.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <mma.h>
#include <cuda_runtime.h>
#include <vector>

#include "fa_common.cuh"

using namespace nvcuda;

namespace fav100 {

// ---------------------------------------------------------------------------
// Forward
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(NTHREADS)
fa_fwd_kernel(const half* __restrict__ Q, const half* __restrict__ K,
              const half* __restrict__ V, half* __restrict__ O,
              float* __restrict__ LSE,
              int qsb, int qsh, int qss,
              int ksb, int ksh, int kss,
              int vsb, int vsh, int vss,
              int osb, int osh, int oss,
              long lsb, long lsh,
              int N, float scale2, float dropout_p, unsigned int seed,
              int H, int causal) {
    // K|V 相邻 (16KB), 循环结束后整体复用为 fp32 输出缓冲 (也是 16KB)
    __shared__ __align__(32) half  sKV[2 * BC * D_HEAD];
    __shared__ __align__(32) half  sP[NWARPS][RPW][BC_PAD];
    __shared__ __align__(32) float sInv[BR];

    half (*sK)[D_HEAD] = reinterpret_cast<half (*)[D_HEAD]>(sKV);
    half (*sV)[D_HEAD] = reinterpret_cast<half (*)[D_HEAD]>(sKV + BC * D_HEAD);
    float* sOut = reinterpret_cast<float*>(sKV);  // [NWARPS][RPW][D_HEAD]

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int b = blockIdx.y / H;
    const int h = blockIdx.y % H;
    const int m0 = blockIdx.x * BR;
    const int row0 = m0 + warp * RPW;

    const half* Qb = Q + (long)b * qsb + (long)h * qsh;
    const half* Kb = K + (long)b * ksb + (long)h * ksh;
    const half* Vb = V + (long)b * vsb + (long)h * vsh;
    half* Ob = O + (long)b * osb + (long)h * osh;

    // ---- Q tile (64 x 64) 落到 shared, 再从 shared 载入 fragment ----
    {
        half (*sQ)[D_HEAD] = reinterpret_cast<half (*)[D_HEAD]>(sKV);
#pragma unroll
        for (int i = tid; i < BR * SEGS; i += NTHREADS) {
            const int r = i / SEGS, seg = (i % SEGS) * VEC;
            const int gr = m0 + r;
            uint4 v = make_uint4(0, 0, 0, 0);
            if (gr < N) v = *reinterpret_cast<const uint4*>(Qb + (long)gr * qss + seg);
            *reinterpret_cast<uint4*>(&sQ[r][seg]) = v;
        }
        __syncthreads();
    }
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> qf[4];
    {
        half (*sQ)[D_HEAD] = reinterpret_cast<half (*)[D_HEAD]>(sKV);
#pragma unroll
        for (int i = 0; i < 4; ++i)
            wmma::load_matrix_sync(qf[i], &sQ[warp * RPW][i * 16], D_HEAD);
    }
    __syncthreads();

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_o[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) wmma::fill_fragment(acc_o[j], 0.f);

    // 本 lane 在累加器 tile 里持有的两行/四列
    const int acc_r0 = acc_row_base(lane);
    const int acc_c0 = acc_col_base(lane);
    const int rowA = row0 + acc_r0;
    const int rowB = rowA + 2;
    const bool lse_writer = ((lane & 2) == 0) && ((lane & 8) == 0);

    float mA = -INFINITY, lA = 0.f;   // 行 rowA 的 running max / sum
    float mB = -INFINITY, lB = 0.f;

    const int hi = causal ? min(N, m0 + BR) : N;
    const bool do_drop = dropout_p > 0.f;
    const unsigned int bh = (unsigned int)(blockIdx.y);
    const float inv_keep = 1.f / (1.f - dropout_p);

    for (int n0 = 0; n0 < hi; n0 += BC) {
        const bool nfull = (n0 + BC <= N);
        const bool need_mask = causal && (n0 + BC > row0);

        if (nfull) {
#pragma unroll
            for (int i = tid; i < BC * SEGS; i += NTHREADS) {
                const int r = i / SEGS, seg = (i % SEGS) * VEC;
                const long off = (long)(n0 + r);
                *reinterpret_cast<uint4*>(&sK[r][seg]) =
                    *reinterpret_cast<const uint4*>(Kb + off * kss + seg);
                *reinterpret_cast<uint4*>(&sV[r][seg]) =
                    *reinterpret_cast<const uint4*>(Vb + off * vss + seg);
            }
        } else {
            for (int i = tid; i < BC * SEGS; i += NTHREADS) {
                const int r = i / SEGS, seg = (i % SEGS) * VEC;
                if (n0 + r < N) {
                    const long off = (long)(n0 + r);
                    *reinterpret_cast<uint4*>(&sK[r][seg]) =
                        *reinterpret_cast<const uint4*>(Kb + off * kss + seg);
                    *reinterpret_cast<uint4*>(&sV[r][seg]) =
                        *reinterpret_cast<const uint4*>(Vb + off * vss + seg);
                } else {
                    *reinterpret_cast<uint4*>(&sK[r][seg]) = make_uint4(0, 0, 0, 0);
                    *reinterpret_cast<uint4*>(&sV[r][seg]) = make_uint4(0, 0, 0, 0);
                }
            }
        }
        __syncthreads();

        // S = Q @ K^T -> 16 x 64 fp32, 直接留在寄存器里 (乘 scale2)
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> sf[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            wmma::fill_fragment(sf[j], 0.f);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> kb;
                wmma::load_matrix_sync(kb, &sK[j * 16][i * 16], D_HEAD);
                wmma::mma_sync(sf[j], qf[i], kb, sf[j]);
            }
        }

        // mask + scale, 同时求行最大值 (跨 4 个 col-tile 与 4 个同组 lane)
        float mxA = -INFINITY, mxB = -INFINITY;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float* x = sf[j].x;
            const int cb = n0 + j * 16 + acc_c0;
            const int c0 = cb, c1 = cb + 1, c4 = cb + 4, c5 = cb + 5;
#pragma unroll
            for (int e = 0; e < 8; ++e) x[e] *= scale2;
            // 越界 (tail) 与 causal mask
            if (c0 >= N) { x[0] = -INFINITY; x[2] = -INFINITY; }
            if (c1 >= N) { x[1] = -INFINITY; x[3] = -INFINITY; }
            if (c4 >= N) { x[4] = -INFINITY; x[6] = -INFINITY; }
            if (c5 >= N) { x[5] = -INFINITY; x[7] = -INFINITY; }
            if (need_mask) {
                if (c0 > rowA) x[0] = -INFINITY;
                if (c1 > rowA) x[1] = -INFINITY;
                if (c4 > rowA) x[4] = -INFINITY;
                if (c5 > rowA) x[5] = -INFINITY;
                if (c0 > rowB) x[2] = -INFINITY;
                if (c1 > rowB) x[3] = -INFINITY;
                if (c4 > rowB) x[6] = -INFINITY;
                if (c5 > rowB) x[7] = -INFINITY;
            }
            mxA = fmaxf(mxA, fmaxf(fmaxf(x[0], x[1]), fmaxf(x[4], x[5])));
            mxB = fmaxf(mxB, fmaxf(fmaxf(x[2], x[3]), fmaxf(x[6], x[7])));
        }
        // 同一行的 4 个 lane 是 lane^2 和 lane^8
        mxA = fmaxf(mxA, __shfl_xor_sync(0xffffffffu, mxA, 2));
        mxA = fmaxf(mxA, __shfl_xor_sync(0xffffffffu, mxA, 8));
        mxB = fmaxf(mxB, __shfl_xor_sync(0xffffffffu, mxB, 2));
        mxB = fmaxf(mxB, __shfl_xor_sync(0xffffffffu, mxB, 8));

        const float mNewA = fmaxf(mA, mxA);
        const float mNewB = fmaxf(mB, mxB);
        const float alphaA = (mNewA == -INFINITY) ? 0.f : exp2f(mA - mNewA);
        const float alphaB = (mNewB == -INFINITY) ? 0.f : exp2f(mB - mNewB);
        const float subA = (mNewA == -INFINITY) ? 0.f : mNewA;
        const float subB = (mNewB == -INFINITY) ? 0.f : mNewB;

        // P = exp2(S - m), 按累加器布局直接写 fp16 的 sP
        float sumA = 0.f, sumB = 0.f;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float* x = sf[j].x;
            const int cb = j * 16 + acc_c0;
            float p0 = exp2f(x[0] - subA), p1 = exp2f(x[1] - subA);
            float p2 = exp2f(x[2] - subB), p3 = exp2f(x[3] - subB);
            float p4 = exp2f(x[4] - subA), p5 = exp2f(x[5] - subA);
            float p6 = exp2f(x[6] - subB), p7 = exp2f(x[7] - subB);
            // 分母必须是"未 dropout"的 exp 和: dropout 是 softmax 之后才作用在 P 上,
            // 不能再按 mask 后的和重新归一化, 否则 lse 也不是未 mask 的 logsumexp,
            // 反向会算错.
            sumA += p0 + p1 + p4 + p5;
            sumB += p2 + p3 + p6 + p7;
            if (do_drop) {
                const int gc = n0 + cb;
                if (!dropout_keep(seed, rowA, gc, bh, dropout_p)) { p0 = 0.f; }
                else p0 *= inv_keep;
                if (!dropout_keep(seed, rowA, gc + 1, bh, dropout_p)) { p1 = 0.f; }
                else p1 *= inv_keep;
                if (!dropout_keep(seed, rowB, gc, bh, dropout_p)) { p2 = 0.f; }
                else p2 *= inv_keep;
                if (!dropout_keep(seed, rowB, gc + 1, bh, dropout_p)) { p3 = 0.f; }
                else p3 *= inv_keep;
                if (!dropout_keep(seed, rowA, gc + 4, bh, dropout_p)) { p4 = 0.f; }
                else p4 *= inv_keep;
                if (!dropout_keep(seed, rowA, gc + 5, bh, dropout_p)) { p5 = 0.f; }
                else p5 *= inv_keep;
                if (!dropout_keep(seed, rowB, gc + 4, bh, dropout_p)) { p6 = 0.f; }
                else p6 *= inv_keep;
                if (!dropout_keep(seed, rowB, gc + 5, bh, dropout_p)) { p7 = 0.f; }
                else p7 *= inv_keep;
            }
            half* pr = &sP[warp][acc_r0][cb];
            *reinterpret_cast<__half2*>(pr) = __floats2half2_rn(p0, p1);
            *reinterpret_cast<__half2*>(pr + 4) = __floats2half2_rn(p4, p5);
            half* pr2 = pr + 2 * BC_PAD;
            *reinterpret_cast<__half2*>(pr2) = __floats2half2_rn(p2, p3);
            *reinterpret_cast<__half2*>(pr2 + 4) = __floats2half2_rn(p6, p7);
        }
        sumA += __shfl_xor_sync(0xffffffffu, sumA, 2);
        sumA += __shfl_xor_sync(0xffffffffu, sumA, 8);
        sumB += __shfl_xor_sync(0xffffffffu, sumB, 2);
        sumB += __shfl_xor_sync(0xffffffffu, sumB, 8);
        lA = lA * alphaA + sumA;
        lB = lB * alphaB + sumB;
        mA = mNewA;
        mB = mNewB;

        // rescale 累加器 (本 lane 的两行, 布局已知)
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float* x = acc_o[j].x;
            x[0] *= alphaA; x[1] *= alphaA; x[4] *= alphaA; x[5] *= alphaA;
            x[2] *= alphaB; x[3] *= alphaB; x[6] *= alphaB; x[7] *= alphaB;
        }
        __syncwarp();

        // acc_o += P @ V
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> pf;
            wmma::load_matrix_sync(pf, &sP[warp][0][i * 16], BC_PAD);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> vb;
                wmma::load_matrix_sync(vb, &sV[i * 16][j * 16], D_HEAD);
                wmma::mma_sync(acc_o[j], pf, vb, acc_o[j]);
            }
        }
        __syncthreads();
    }

    // ---- 输出: acc -> sOut (复用 sKV) -> global ----
    __syncthreads();
#pragma unroll
    for (int j = 0; j < 4; ++j)
        wmma::store_matrix_sync(&sOut[warp * RPW * D_HEAD + j * 16], acc_o[j],
                                D_HEAD, wmma::mem_row_major);
    if (lse_writer) {
        const int lrowA = warp * RPW + acc_r0;   // sInv 用 CTA 内行号
        sInv[lrowA] = (lA > 0.f) ? (1.f / lA) : 0.f;
        sInv[lrowA + 2] = (lB > 0.f) ? (1.f / lB) : 0.f;
        if (rowA < N)
            LSE[(long)b * lsb + (long)h * lsh + rowA] =
                (mA == -INFINITY) ? -INFINITY : (mA + log2f(lA)) * LN2_F;
        if (rowB < N)
            LSE[(long)b * lsb + (long)h * lsh + rowB] =
                (mB == -INFINITY) ? -INFINITY : (mB + log2f(lB)) * LN2_F;
    }
    __syncthreads();

#pragma unroll
    for (int i = tid; i < BR * SEGS; i += NTHREADS) {
        const int r = i / SEGS, seg = (i % SEGS) * VEC;
        if (m0 + r < N) {
            const float* src = &sOut[r * D_HEAD + seg];
            const float invl = sInv[r];
            const __half2 h0 = __floats2half2_rn(src[0] * invl, src[1] * invl);
            const __half2 h1 = __floats2half2_rn(src[2] * invl, src[3] * invl);
            const __half2 h2 = __floats2half2_rn(src[4] * invl, src[5] * invl);
            const __half2 h3 = __floats2half2_rn(src[6] * invl, src[7] * invl);
            uint4 out;
            reinterpret_cast<__half2*>(&out)[0] = h0;
            reinterpret_cast<__half2*>(&out)[1] = h1;
            reinterpret_cast<__half2*>(&out)[2] = h2;
            reinterpret_cast<__half2*>(&out)[3] = h3;
            *reinterpret_cast<uint4*>(Ob + (long)(m0 + r) * oss + seg) = out;
        }
    }
}

// ---------------------------------------------------------------------------
// host 侧
// ---------------------------------------------------------------------------
static inline void check_fwd(const torch::Tensor& q, const torch::Tensor& k,
                             const torch::Tensor& v) {
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(), "inputs must be CUDA");
    TORCH_CHECK(q.dtype() == torch::kHalf, "only fp16 supported");
    TORCH_CHECK(q.dim() == 4, "q must be (B,H,S,D)");
    TORCH_CHECK(q.size(3) == 64, "head_dim must be 64");
    TORCH_CHECK(q.stride(3) == 1 && k.stride(3) == 1 && v.stride(3) == 1,
                "last dim must be contiguous");
    TORCH_CHECK(q.sizes() == k.sizes() && q.sizes() == v.sizes(), "q/k/v shape mismatch");
    for (int t = 0; t < 3; ++t) {
        const torch::Tensor& x = (t == 0 ? q : (t == 1 ? k : v));
        // uint4 向量化 + wmma load_matrix_sync 都要求 16B/32B 对齐
        TORCH_CHECK((x.stride(2) * 2) % 32 == 0,
                    "stride(2)*2 must be 32B aligned for wmma");
        TORCH_CHECK((x.stride(1) * 2) % 32 == 0,
                    "stride(1)*2 must be 32B aligned for vectorized loads");
        TORCH_CHECK((x.stride(0) * 2) % 32 == 0,
                    "stride(0)*2 must be 32B aligned for vectorized loads");
        TORCH_CHECK((reinterpret_cast<uintptr_t>(x.data_ptr()) % 32) == 0,
                    "tensor base must be 32B aligned");
    }
}

std::vector<torch::Tensor> fa_fwd(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                                  double dropout_p, bool causal, double softmax_scale,
                                  int64_t seed) {
    check_fwd(q, k, v);
    const int B = q.size(0), H = q.size(1), N = q.size(2);
    auto o = torch::empty({B, H, N, D_HEAD}, q.options());
    auto lse = torch::empty({B, H, N}, q.options().dtype(torch::kFloat));

    const int grid_x = (N + BR - 1) / BR;
    dim3 grid(grid_x, B * H);
    const float scale2 = (float)(softmax_scale * (double)LOG2E_F);

    auto stream = at::cuda::getCurrentCUDAStream();
    fa_fwd_kernel<<<grid, NTHREADS, 0, stream>>>(
        reinterpret_cast<const half*>(q.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(k.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(v.data_ptr<at::Half>()),
        reinterpret_cast<half*>(o.data_ptr<at::Half>()),
        lse.data_ptr<float>(),
        (int)q.stride(0), (int)q.stride(1), (int)q.stride(2),
        (int)k.stride(0), (int)k.stride(1), (int)k.stride(2),
        (int)v.stride(0), (int)v.stride(1), (int)v.stride(2),
        (int)o.stride(0), (int)o.stride(1), (int)o.stride(2),
        (long)lse.stride(0), (long)lse.stride(1),
        N, scale2, (float)dropout_p, (unsigned int)seed, H, causal ? 1 : 0);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {o, lse};
}

}  // namespace fav100

namespace fav100 {
// 实现在 flash_attn_v100_bwd.cu
std::vector<torch::Tensor> fa_bwd(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                                  torch::Tensor o, torch::Tensor lse,
                                  torch::Tensor do_, double dropout_p, bool causal,
                                  double softmax_scale, int64_t seed,
                                  int64_t mode);
}  // namespace fav100

// fast_ops.cu: 融合 RoPE (逐位等价于 eager 实现)
namespace fastops {
std::vector<torch::Tensor> rope_fwd(torch::Tensor q, torch::Tensor k,
                                    torch::Tensor cos, torch::Tensor sin);
std::vector<torch::Tensor> rope_bwd(torch::Tensor dqo, torch::Tensor dko,
                                    torch::Tensor cos, torch::Tensor sin,
                                    torch::Tensor q, torch::Tensor k);
}  // namespace fastops

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fa_fwd", &fav100::fa_fwd, "FA2 sm70 forward");
    m.def("fa_bwd", &fav100::fa_bwd, "FA2 sm70 backward",
          pybind11::arg("q"), pybind11::arg("k"), pybind11::arg("v"),
          pybind11::arg("o"), pybind11::arg("lse"), pybind11::arg("do"),
          pybind11::arg("dropout_p"), pybind11::arg("causal"),
          pybind11::arg("softmax_scale"), pybind11::arg("seed"),
          // mode 位掩码: 1=delta, 2=dq, 4=dkv, 7=全部 (默认)
          pybind11::arg("mode") = 7);
    m.def("rope_fwd", &fastops::rope_fwd, "fused RoPE forward");
    m.def("rope_bwd", &fastops::rope_bwd, "fused RoPE backward");
}

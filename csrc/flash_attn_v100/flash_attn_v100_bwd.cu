// FlashAttention-2 backward for NVIDIA V100 / SM70.
//
// FA2 风格的两个 kernel:
//   1. fa_bwd_dkv_kernel : 每个 CTA 负责一个 key block (64 keys), 循环所有 query block
//                          dV += P^T dO,  dK += dS^T Q
//   2. fa_bwd_dq_kernel  : 每个 CTA 负责一个 query block (64 q), 循环所有 key block
//                          dQ += dS K
// 外加一个 preprocess kernel 求 delta_i = rowsum(O_i * dO_i).
//
// 关键公式 (S2 = scale*log2e*(Q.K), lse 为 fp32 natural log):
//   P    = exp2(S2 - lse*log2e)
//   P_d  = mask * P / (1-p)                     (dropout 用与前向同一个 hash)
//   dS   = P_d * dP - P * delta                 (等价于 P*(mask*dP/(1-p) - delta))
//   dV   = P_d^T dO,  dP = dO V^T,  dQ = scale*dS V,  dK = scale*dS^T Q
//
// 矩阵布局 (shared tile 都是 row-major, 用 wmma 读):
//   dK/dV kernel 全部在 (key x q) 视角:
//     S  = K Q^T   : A=sK row_major, B=sQ col_major
//     dP = V dO^T  : A=sV row_major, B=sDO col_major
//     dV += P^T dO : A=sP row_major, B=sDO row_major   (sP[key][q] 即 P^T)
//     dK += dS^T Q : A=sP(改成 dS) row_major, B=sQ row_major
//   dQ kernel 全部在 (q x key) 视角:
//     S  = Q K^T   : A=sQ row_major, B=sK col_major
//     dP = dO V^T  : A=sDO row_major, B=sV col_major
//     dQ += dS K   : A=sDS row_major, B=sK row_major

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <mma.h>
#include <cuda_runtime.h>
#include <vector>

#include "fa_common.cuh"

using namespace nvcuda;

namespace fav100 {

// ---------------------------------------------------------------------------
// delta_i = rowsum(O_i * dO_i);  每行 8 线程, 每 block 64 行
// ---------------------------------------------------------------------------
__global__ void fa_bwd_delta_kernel(const half* __restrict__ O,
                                    const half* __restrict__ dO,
                                    float* __restrict__ DELTA,
                                    int N) {
    const int lane8 = threadIdx.x & 7;
    const int row = blockIdx.x * 64 + (threadIdx.x >> 3);
    if (row >= N) return;
    const long base = ((long)blockIdx.y * N + row) * D_HEAD + lane8 * VEC;
    const uint4 ov = *reinterpret_cast<const uint4*>(O + base);
    const uint4 dv = *reinterpret_cast<const uint4*>(dO + base);
    const half* oh = reinterpret_cast<const half*>(&ov);
    const half* dh = reinterpret_cast<const half*>(&dv);
    float sum = 0.f;
#pragma unroll
    for (int t = 0; t < VEC; ++t) sum += __half2float(oh[t]) * __half2float(dh[t]);
    sum += __shfl_xor_sync(0xffffffffu, sum, 1);
    sum += __shfl_xor_sync(0xffffffffu, sum, 2);
    sum += __shfl_xor_sync(0xffffffffu, sum, 4);
    if (lane8 == 0) DELTA[(long)blockIdx.y * N + row] = sum;
}

// ---------------------------------------------------------------------------
// dQ kernel
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(NTHREADS)
fa_bwd_dq_kernel(const half* __restrict__ Q, const half* __restrict__ K,
                 const half* __restrict__ V, const half* __restrict__ dO,
                 const float* __restrict__ LSE, const float* __restrict__ DELTA,
                 half* __restrict__ dQ,
                 int qsb, int qsh, int qss,
                 int ksb, int ksh, int kss,
                 int vsb, int vsh, int vss,
                 int oqb, int oqh, int oqs,
                 int N, float scale2, float scale, float dropout_p,
                 unsigned int seed, int H, int causal) {
    __shared__ __align__(32) half sKV[2 * BC * D_HEAD];       // K, V
    __shared__ __align__(32) half sQD[2 * BR * D_HEAD];       // Q, dO (只加载一次)
    __shared__ __align__(32) half sDS[NWARPS][RPW][BC_PAD];   // dS

    half (*sK)[D_HEAD] = reinterpret_cast<half (*)[D_HEAD]>(sKV);
    half (*sV)[D_HEAD] = reinterpret_cast<half (*)[D_HEAD]>(sKV + BC * D_HEAD);
    half (*sQ)[D_HEAD] = reinterpret_cast<half (*)[D_HEAD]>(sQD);
    half (*sDO)[D_HEAD] = reinterpret_cast<half (*)[D_HEAD]>(sQD + BR * D_HEAD);
    float* sOut = reinterpret_cast<float*>(sKV);

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int b = blockIdx.y / H, h = blockIdx.y % H;
    const int m0 = blockIdx.x * BR;
    const int row0 = m0 + warp * RPW;
    const int bh = blockIdx.y;

    load_tile(sQ, Q + (long)b * qsb + (long)h * qsh, qss, N, m0, tid, NTHREADS);
    // dO 由 Python 侧保证连续 (B,H,N,D)
    load_tile(sDO, dO + (long)bh * N * D_HEAD, D_HEAD, N, m0, tid, NTHREADS);
    __syncthreads();

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> qf[4];
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> dof[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        wmma::load_matrix_sync(qf[i], &sQ[warp * RPW][i * 16], D_HEAD);
        wmma::load_matrix_sync(dof[i], &sDO[warp * RPW][i * 16], D_HEAD);
    }
    __syncthreads();

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_dq[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) wmma::fill_fragment(acc_dq[j], 0.f);

    const int acc_r0 = acc_row_base(lane);
    const int acc_c0 = acc_col_base(lane);
    const int rowA = row0 + acc_r0, rowB = rowA + 2;
    const long lbase = (long)bh * N;
    const float lseA = (rowA < N) ? LSE[lbase + rowA] * LOG2E_F : 0.f;
    const float lseB = (rowB < N) ? LSE[lbase + rowB] * LOG2E_F : 0.f;
    const float delA = (rowA < N) ? DELTA[lbase + rowA] : 0.f;
    const float delB = (rowB < N) ? DELTA[lbase + rowB] : 0.f;

    const int hi = causal ? min(N, m0 + BR) : N;
    const bool do_drop = dropout_p > 0.f;
    const float inv_keep = 1.f / (1.f - dropout_p);

    for (int n0 = 0; n0 < hi; n0 += BC) {
        const bool need_mask = causal && (n0 + BC > row0);

        load_tile(sK, K + (long)b * ksb + (long)h * ksh, kss, N, n0, tid, NTHREADS);
        load_tile(sV, V + (long)b * vsb + (long)h * vsh, vss, N, n0, tid, NTHREADS);
        __syncthreads();

        // S = Q K^T ; dP = dO V^T (都是 16 x 64)
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> sf[4], dpf[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            wmma::fill_fragment(sf[j], 0.f);
            wmma::fill_fragment(dpf[j], 0.f);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> kb, vb;
                wmma::load_matrix_sync(kb, &sK[j * 16][i * 16], D_HEAD);
                wmma::mma_sync(sf[j], qf[i], kb, sf[j]);
                wmma::load_matrix_sync(vb, &sV[j * 16][i * 16], D_HEAD);
                wmma::mma_sync(dpf[j], dof[i], vb, dpf[j]);
            }
        }

        // dS = P * (mask*dP/(1-p) - delta) -> sDS
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float* s = sf[j].x;
            const float* y = dpf[j].x;
            const int cb = j * 16 + acc_c0;
            const int gc = n0 + cb;
            const bool in0 = (gc < N), in1 = (gc + 1 < N), in4 = (gc + 4 < N), in5 = (gc + 5 < N);
            float p0 = (in0 && (!need_mask || gc <= rowA)) ? exp2f(s[0] * scale2 - lseA) : 0.f;
            float p1 = (in1 && (!need_mask || gc + 1 <= rowA)) ? exp2f(s[1] * scale2 - lseA) : 0.f;
            float p2 = (in0 && (!need_mask || gc <= rowB)) ? exp2f(s[2] * scale2 - lseB) : 0.f;
            float p3 = (in1 && (!need_mask || gc + 1 <= rowB)) ? exp2f(s[3] * scale2 - lseB) : 0.f;
            float p4 = (in4 && (!need_mask || gc + 4 <= rowA)) ? exp2f(s[4] * scale2 - lseA) : 0.f;
            float p5 = (in5 && (!need_mask || gc + 5 <= rowA)) ? exp2f(s[5] * scale2 - lseA) : 0.f;
            float p6 = (in4 && (!need_mask || gc + 4 <= rowB)) ? exp2f(s[6] * scale2 - lseB) : 0.f;
            float p7 = (in5 && (!need_mask || gc + 5 <= rowB)) ? exp2f(s[7] * scale2 - lseB) : 0.f;
            float k0 = 1.f, k1 = 1.f, k2 = 1.f, k3 = 1.f;
            float k4 = 1.f, k5 = 1.f, k6 = 1.f, k7 = 1.f;
            if (do_drop) {
                k0 = dropout_keep(seed, rowA, gc, bh, dropout_p) ? inv_keep : 0.f;
                k1 = dropout_keep(seed, rowA, gc + 1, bh, dropout_p) ? inv_keep : 0.f;
                k2 = dropout_keep(seed, rowB, gc, bh, dropout_p) ? inv_keep : 0.f;
                k3 = dropout_keep(seed, rowB, gc + 1, bh, dropout_p) ? inv_keep : 0.f;
                k4 = dropout_keep(seed, rowA, gc + 4, bh, dropout_p) ? inv_keep : 0.f;
                k5 = dropout_keep(seed, rowA, gc + 5, bh, dropout_p) ? inv_keep : 0.f;
                k6 = dropout_keep(seed, rowB, gc + 4, bh, dropout_p) ? inv_keep : 0.f;
                k7 = dropout_keep(seed, rowB, gc + 5, bh, dropout_p) ? inv_keep : 0.f;
            }
            const float d0 = p0 * (k0 * y[0] - delA);
            const float d1 = p1 * (k1 * y[1] - delA);
            const float d2 = p2 * (k2 * y[2] - delB);
            const float d3 = p3 * (k3 * y[3] - delB);
            const float d4 = p4 * (k4 * y[4] - delA);
            const float d5 = p5 * (k5 * y[5] - delA);
            const float d6 = p6 * (k6 * y[6] - delB);
            const float d7 = p7 * (k7 * y[7] - delB);
            half* pr = &sDS[warp][acc_r0][cb];
            *reinterpret_cast<__half2*>(pr) = __floats2half2_rn(d0, d1);
            *reinterpret_cast<__half2*>(pr + 4) = __floats2half2_rn(d4, d5);
            half* pr2 = pr + 2 * BC_PAD;
            *reinterpret_cast<__half2*>(pr2) = __floats2half2_rn(d2, d3);
            *reinterpret_cast<__half2*>(pr2 + 4) = __floats2half2_rn(d6, d7);
        }
        __syncwarp();

        // dQ += dS K  (S = Q K^T, 所以 dQ = scale * dS K)
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> sa;
            wmma::load_matrix_sync(sa, &sDS[warp][0][i * 16], BC_PAD);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> kb;
                wmma::load_matrix_sync(kb, &sK[i * 16][j * 16], D_HEAD);
                wmma::mma_sync(acc_dq[j], sa, kb, acc_dq[j]);
            }
        }
        __syncthreads();
    }

    // 输出 dQ (乘 scale)
#pragma unroll
    for (int j = 0; j < 4; ++j) {
#pragma unroll
        for (int e = 0; e < acc_dq[j].num_elements; ++e) acc_dq[j].x[e] *= scale;
        wmma::store_matrix_sync(&sOut[warp * RPW * D_HEAD + j * 16], acc_dq[j],
                                D_HEAD, wmma::mem_row_major);
    }
    __syncthreads();
    store_out(dQ + (long)b * oqb + (long)h * oqh, sOut, oqs, N, m0, tid, NTHREADS);
}

// ---------------------------------------------------------------------------
// dK/dV kernel
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(NTHREADS)
fa_bwd_dkv_kernel(const half* __restrict__ Q, const half* __restrict__ K,
                  const half* __restrict__ V, const half* __restrict__ dO,
                  const float* __restrict__ LSE, const float* __restrict__ DELTA,
                  half* __restrict__ dK, half* __restrict__ dV,
                  int qsb, int qsh, int qss,
                  int ksb, int ksh, int kss,
                  int vsb, int vsh, int vss,
                  int N, float scale2, float scale, float dropout_p,
                  unsigned int seed, int H, int causal) {
    __shared__ __align__(32) half sKV[2 * BC * D_HEAD];      // K, V (本 key block, 只加载一次)
    __shared__ __align__(32) half sQD[2 * BR * D_HEAD];      // Q, dO (每个 q block)
    __shared__ __align__(32) half sP[NWARPS][RPW][BC_PAD];   // P_d -> dS
    __shared__ __align__(32) float sLSE[BR];
    __shared__ __align__(32) float sDELTA[BR];

    half (*sK)[D_HEAD] = reinterpret_cast<half (*)[D_HEAD]>(sKV);
    half (*sV)[D_HEAD] = reinterpret_cast<half (*)[D_HEAD]>(sKV + BC * D_HEAD);
    half (*sQ)[D_HEAD] = reinterpret_cast<half (*)[D_HEAD]>(sQD);
    half (*sDO)[D_HEAD] = reinterpret_cast<half (*)[D_HEAD]>(sQD + BR * D_HEAD);
    float* sOutK = reinterpret_cast<float*>(sKV);
    float* sOutV = reinterpret_cast<float*>(sQD);

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int b = blockIdx.y / H, h = blockIdx.y % H;
    const int n0 = blockIdx.x * BC;
    const int krow0 = n0 + warp * RPW;
    const int bh = blockIdx.y;
    const long lbase = (long)bh * N;

    // K/V tile: 只加载一次
    load_tile(sK, K + (long)b * ksb + (long)h * ksh, kss, N, n0, tid, NTHREADS);
    load_tile(sV, V + (long)b * vsb + (long)h * vsh, vss, N, n0, tid, NTHREADS);
    __syncthreads();

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_dk[4], acc_dv[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        wmma::fill_fragment(acc_dk[j], 0.f);
        wmma::fill_fragment(acc_dv[j], 0.f);
    }

    const int acc_r0 = acc_row_base(lane);
    const int acc_c0 = acc_col_base(lane);
    const int keyA = krow0 + acc_r0, keyB = keyA + 2;

    const bool do_drop = dropout_p > 0.f;
    const float inv_keep = 1.f / (1.f - dropout_p);
    const int qstart = causal ? (n0 / BR) * BR : 0;

    for (int m0 = qstart; m0 < N; m0 += BR) {
        const bool need_mask = causal && (m0 < n0 + BC);

        load_tile(sQ, Q + (long)b * qsb + (long)h * qsh, qss, N, m0, tid, NTHREADS);
        load_tile(sDO, dO + (long)lbase * D_HEAD, D_HEAD, N, m0, tid, NTHREADS);
        for (int i = tid; i < BR; i += NTHREADS) {
            const int g = m0 + i;
            sLSE[i] = (g < N) ? LSE[lbase + g] : 0.f;
            sDELTA[i] = (g < N) ? DELTA[lbase + g] : 0.f;
        }
        __syncthreads();

        // S = K Q^T (16 keys x 64 q); A 片段只依赖 i, 提到外层避免重复加载
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> sf[4];
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> ka[4];
#pragma unroll
        for (int i = 0; i < 4; ++i)
            wmma::load_matrix_sync(ka[i], &sK[warp * RPW][i * 16], D_HEAD);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            wmma::fill_fragment(sf[j], 0.f);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> qb;
                wmma::load_matrix_sync(qb, &sQ[j * 16][i * 16], D_HEAD);
                wmma::mma_sync(sf[j], ka[i], qb, sf[j]);
            }
        }

        // P_d = mask*exp2(S*scale2 - lse)/(1-p) -> sP
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float* s = sf[j].x;
            const int cb = j * 16 + acc_c0;
            const int gc = m0 + cb;
            const float l0 = sLSE[cb], l1 = sLSE[cb + 1], l4 = sLSE[cb + 4], l5 = sLSE[cb + 5];
            const bool in0 = (gc < N), in1 = (gc + 1 < N), in4 = (gc + 4 < N), in5 = (gc + 5 < N);
            float p0 = (in0 && (!need_mask || gc >= keyA)) ? exp2f(s[0] * scale2 - l0 * LOG2E_F) : 0.f;
            float p1 = (in1 && (!need_mask || gc + 1 >= keyA)) ? exp2f(s[1] * scale2 - l1 * LOG2E_F) : 0.f;
            float p2 = (in0 && (!need_mask || gc >= keyB)) ? exp2f(s[2] * scale2 - l0 * LOG2E_F) : 0.f;
            float p3 = (in1 && (!need_mask || gc + 1 >= keyB)) ? exp2f(s[3] * scale2 - l1 * LOG2E_F) : 0.f;
            float p4 = (in4 && (!need_mask || gc + 4 >= keyA)) ? exp2f(s[4] * scale2 - l4 * LOG2E_F) : 0.f;
            float p5 = (in5 && (!need_mask || gc + 5 >= keyA)) ? exp2f(s[5] * scale2 - l5 * LOG2E_F) : 0.f;
            float p6 = (in4 && (!need_mask || gc + 4 >= keyB)) ? exp2f(s[6] * scale2 - l4 * LOG2E_F) : 0.f;
            float p7 = (in5 && (!need_mask || gc + 5 >= keyB)) ? exp2f(s[7] * scale2 - l5 * LOG2E_F) : 0.f;
            s[0] = p0; s[1] = p1; s[2] = p2; s[3] = p3;   // 保留未 mask 的 P 给 dS
            s[4] = p4; s[5] = p5; s[6] = p6; s[7] = p7;
            if (do_drop) {
                p0 *= dropout_keep(seed, gc, keyA, bh, dropout_p) ? inv_keep : 0.f;
                p1 *= dropout_keep(seed, gc + 1, keyA, bh, dropout_p) ? inv_keep : 0.f;
                p2 *= dropout_keep(seed, gc, keyB, bh, dropout_p) ? inv_keep : 0.f;
                p3 *= dropout_keep(seed, gc + 1, keyB, bh, dropout_p) ? inv_keep : 0.f;
                p4 *= dropout_keep(seed, gc + 4, keyA, bh, dropout_p) ? inv_keep : 0.f;
                p5 *= dropout_keep(seed, gc + 5, keyA, bh, dropout_p) ? inv_keep : 0.f;
                p6 *= dropout_keep(seed, gc + 4, keyB, bh, dropout_p) ? inv_keep : 0.f;
                p7 *= dropout_keep(seed, gc + 5, keyB, bh, dropout_p) ? inv_keep : 0.f;
            }
            half* pr = &sP[warp][acc_r0][cb];
            *reinterpret_cast<__half2*>(pr) = __floats2half2_rn(p0, p1);
            *reinterpret_cast<__half2*>(pr + 4) = __floats2half2_rn(p4, p5);
            half* pr2 = pr + 2 * BC_PAD;
            *reinterpret_cast<__half2*>(pr2) = __floats2half2_rn(p2, p3);
            *reinterpret_cast<__half2*>(pr2 + 4) = __floats2half2_rn(p6, p7);
        }
        __syncwarp();

        // dV += P_d^T dO
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> pa;
            wmma::load_matrix_sync(pa, &sP[warp][0][i * 16], BC_PAD);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> vb;
                wmma::load_matrix_sync(vb, &sDO[i * 16][j * 16], D_HEAD);
                wmma::mma_sync(acc_dv[j], pa, vb, acc_dv[j]);
            }
        }

        // dP = V dO^T (16 keys x 64 q); A 片段同样只依赖 i
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> dpf[4];
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> va[4];
#pragma unroll
        for (int i = 0; i < 4; ++i)
            wmma::load_matrix_sync(va[i], &sV[warp * RPW][i * 16], D_HEAD);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            wmma::fill_fragment(dpf[j], 0.f);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> db;
                wmma::load_matrix_sync(db, &sDO[j * 16][i * 16], D_HEAD);
                wmma::mma_sync(dpf[j], va[i], db, dpf[j]);
            }
        }
        __syncwarp();

        // dS = P_d*dP - P*delta  (P_d 从 shared 读回, 与 dV 用的完全一致) -> sP
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float* s = sf[j].x;      // 未 mask 的 P
            const float* y = dpf[j].x;
            const int cb = j * 16 + acc_c0;
            const float d0 = sDELTA[cb], d1 = sDELTA[cb + 1], d4 = sDELTA[cb + 4], d5 = sDELTA[cb + 5];
            const __half2* pp = reinterpret_cast<const __half2*>(&sP[warp][acc_r0][cb]);
            const float2 f0 = __half22float2(pp[0]);
            const float2 f4 = __half22float2(pp[2]);
            const __half2* pp2 = reinterpret_cast<const __half2*>(&sP[warp][acc_r0 + 2][cb]);
            const float2 f2 = __half22float2(pp2[0]);
            const float2 f6 = __half22float2(pp2[2]);
            half* pr = &sP[warp][acc_r0][cb];
            *reinterpret_cast<__half2*>(pr) =
                __floats2half2_rn(f0.x * y[0] - s[0] * d0, f0.y * y[1] - s[1] * d1);
            *reinterpret_cast<__half2*>(pr + 4) =
                __floats2half2_rn(f4.x * y[4] - s[4] * d4, f4.y * y[5] - s[5] * d5);
            half* pr2 = pr + 2 * BC_PAD;
            *reinterpret_cast<__half2*>(pr2) =
                __floats2half2_rn(f2.x * y[2] - s[2] * d0, f2.y * y[3] - s[3] * d1);
            *reinterpret_cast<__half2*>(pr2 + 4) =
                __floats2half2_rn(f6.x * y[6] - s[6] * d4, f6.y * y[7] - s[7] * d5);
        }
        __syncwarp();

        // dK += dS^T Q
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> sa;
            wmma::load_matrix_sync(sa, &sP[warp][0][i * 16], BC_PAD);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> qb;
                wmma::load_matrix_sync(qb, &sQ[i * 16][j * 16], D_HEAD);
                wmma::mma_sync(acc_dk[j], sa, qb, acc_dk[j]);
            }
        }
        __syncthreads();
    }

    // 输出 dK, dV (乘 scale 的只有 dK)
#pragma unroll
    for (int j = 0; j < 4; ++j) {
#pragma unroll
        for (int e = 0; e < acc_dk[j].num_elements; ++e) acc_dk[j].x[e] *= scale;
        wmma::store_matrix_sync(&sOutK[warp * RPW * D_HEAD + j * 16], acc_dk[j],
                                D_HEAD, wmma::mem_row_major);
        wmma::store_matrix_sync(&sOutV[warp * RPW * D_HEAD + j * 16], acc_dv[j],
                                D_HEAD, wmma::mem_row_major);
    }
    __syncthreads();
    // dK / dV 都是 (B,H,N,D) 连续
    store_out(dK + (long)bh * N * D_HEAD, sOutK, D_HEAD, N, n0, tid, NTHREADS);
    store_out(dV + (long)bh * N * D_HEAD, sOutV, D_HEAD, N, n0, tid, NTHREADS);
}

// ---------------------------------------------------------------------------
// host 侧
// ---------------------------------------------------------------------------
std::vector<torch::Tensor> fa_bwd(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                                  torch::Tensor o, torch::Tensor lse,
                                  torch::Tensor do_, double dropout_p, bool causal,
                                  double softmax_scale, int64_t seed,
                                  int64_t mode = 7) {
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(), "inputs must be CUDA");
    TORCH_CHECK(q.dtype() == torch::kHalf, "only fp16 supported");
    TORCH_CHECK(q.dim() == 4 && q.size(3) == D_HEAD, "q must be (B,H,S,64)");
    TORCH_CHECK(q.sizes() == k.sizes() && q.sizes() == v.sizes(), "q/k/v shape mismatch");
    TORCH_CHECK(q.stride(3) == 1 && k.stride(3) == 1 && v.stride(3) == 1,
                "last dim must be contiguous");
    {
        const torch::Tensor* ts[3] = {&q, &k, &v};
        for (int t = 0; t < 3; ++t) {
            const torch::Tensor& x = *ts[t];
            TORCH_CHECK((x.stride(2) * 2) % 32 == 0 && (x.stride(1) * 2) % 32 == 0
                            && (x.stride(0) * 2) % 32 == 0,
                        "strides must be 32B aligned for vectorized loads");
            TORCH_CHECK((reinterpret_cast<uintptr_t>(x.data_ptr()) % 32) == 0,
                        "tensor base must be 32B aligned");
        }
    }

    auto do_c = do_.is_contiguous() ? do_ : do_.contiguous();
    auto o_c = o.is_contiguous() ? o : o.contiguous();
    const int B = q.size(0), H = q.size(1), N = q.size(2);
    const int BH = B * H;
    auto opts = q.options();
    auto dq = torch::empty({B, H, N, D_HEAD}, opts);
    auto dk = torch::empty({B, H, N, D_HEAD}, opts);
    auto dv = torch::empty({B, H, N, D_HEAD}, opts);
    auto delta = torch::empty({(long)BH, N}, opts.dtype(torch::kFloat));

    const float scale2 = (float)(softmax_scale * (double)LOG2E_F);
    const float sc = (float)softmax_scale;
    const unsigned int sd = (unsigned int)seed;
    auto stream = at::cuda::getCurrentCUDAStream();

    const half* qp = reinterpret_cast<const half*>(q.data_ptr<at::Half>());
    const half* kp = reinterpret_cast<const half*>(k.data_ptr<at::Half>());
    const half* vp = reinterpret_cast<const half*>(v.data_ptr<at::Half>());
    const half* dop = reinterpret_cast<const half*>(do_c.data_ptr<at::Half>());

    dim3 dgrid((N + 63) / 64, BH);
    if (mode & 1)
        fa_bwd_delta_kernel<<<dgrid, 512, 0, stream>>>(
            reinterpret_cast<const half*>(o_c.data_ptr<at::Half>()), dop,
            delta.data_ptr<float>(), N);

    dim3 qgrid((N + BR - 1) / BR, BH);
    if (mode & 2)
        fa_bwd_dq_kernel<<<qgrid, NTHREADS, 0, stream>>>(
        qp, kp, vp, dop, lse.data_ptr<float>(), delta.data_ptr<float>(),
        reinterpret_cast<half*>(dq.data_ptr<at::Half>()),
        (int)q.stride(0), (int)q.stride(1), (int)q.stride(2),
        (int)k.stride(0), (int)k.stride(1), (int)k.stride(2),
        (int)v.stride(0), (int)v.stride(1), (int)v.stride(2),
        (int)dq.stride(0), (int)dq.stride(1), (int)dq.stride(2),
        N, scale2, sc, (float)dropout_p, sd, H, causal ? 1 : 0);

    dim3 kgrid((N + BC - 1) / BC, BH);
    if (mode & 4)
        fa_bwd_dkv_kernel<<<kgrid, NTHREADS, 0, stream>>>(
        qp, kp, vp, dop, lse.data_ptr<float>(), delta.data_ptr<float>(),
        reinterpret_cast<half*>(dk.data_ptr<at::Half>()),
        reinterpret_cast<half*>(dv.data_ptr<at::Half>()),
        (int)q.stride(0), (int)q.stride(1), (int)q.stride(2),
        (int)k.stride(0), (int)k.stride(1), (int)k.stride(2),
        (int)v.stride(0), (int)v.stride(1), (int)v.stride(2),
        N, scale2, sc, (float)dropout_p, sd, H, causal ? 1 : 0);

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {dq, dk, dv};
}

}  // namespace fav100

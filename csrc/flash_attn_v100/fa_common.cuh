// flash_attn_v100 前向/反向共用的常量与工具函数.
// 注意: dropout_keep 必须被前向和反向以完全相同的参数调用, 否则 mask 会不一致.
#pragma once

#include <cuda_fp16.h>

namespace fav100 {

constexpr int D_HEAD = 64;
constexpr int BR = 64;            // queries per CTA (fwd/dQ) 或 keys per CTA (dK/dV)
constexpr int BC = 64;            // keys per block (fwd) 或 queries per block (dK/dV)
constexpr int NWARPS = 4;
constexpr int RPW = BR / NWARPS;  // 16 rows per warp
constexpr int NTHREADS = NWARPS * 32;
constexpr int VEC = 8;            // halves per uint4
constexpr int SEGS = D_HEAD / VEC;
constexpr int BC_PAD = BC + 8;    // sP 行跨度, 减少 bank conflict
constexpr float LOG2E_F = 1.4426950408889634f;
constexpr float LN2_F = 0.6931471805599453f;

__device__ __forceinline__ unsigned int hash_u32(unsigned int x) {
    x ^= x >> 16; x *= 0x7feb352dU;
    x ^= x >> 15; x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}

// dropout keep mask: (seed, query row, key col, batch*head) 的纯函数 hash.
// 前向/反向都调用它, 因此不需要保存 N x N 的 mask.
__device__ __forceinline__ bool dropout_keep(unsigned int seed, int row, int col,
                                             unsigned int bh, float p) {
    unsigned int h = seed * 0x9E3779B1u;
    h ^= (unsigned int)(row + 1) * 0x85EBCA6Bu;
    h ^= (unsigned int)(col + 1) * 0xC2B2AE35u;
    h ^= (bh + 1) * 0x27D4EB2Fu;
    h = hash_u32(h);
    return ((h >> 8) * (1.0f / 16777216.0f)) >= p;
}

// wmma m16n16k16 fp32 accumulator 在 sm70 上的元素布局 (实测):
//   本 lane 持有行 base 与 base+2, 列为 c, c+1, c+4, c+5
__device__ __forceinline__ int acc_row_base(int lane) {
    return ((lane >> 2) & 1) * 8 + ((lane >> 4) & 1) * 4 + (lane & 1);
}

__device__ __forceinline__ int acc_col_base(int lane) {
    return 2 * ((lane >> 1) & 1) + 8 * ((lane >> 3) & 1);
}

// 把一个 (rows x 64) tile 从全局 (任意行跨度) 载入 shared, uint4 向量化, 越界补 0
__device__ __forceinline__ void load_tile(half (*dst)[D_HEAD], const half* src,
                                          int row_stride, int N, int g0,
                                          int tid, int nthreads) {
    for (int i = tid; i < BR * SEGS; i += nthreads) {
        const int r = i / SEGS, seg = (i % SEGS) * VEC;
        const int gr = g0 + r;
        uint4 v = make_uint4(0, 0, 0, 0);
        if (gr < N) v = *reinterpret_cast<const uint4*>(src + (long)gr * row_stride + seg);
        *reinterpret_cast<uint4*>(&dst[r][seg]) = v;
    }
}

// 把 shared 里的 fp32 结果 (BR x 64) 转 fp16 写回全局
__device__ __forceinline__ void store_out(half* dst, const float* src, int row_stride,
                                          int N, int g0, int tid, int nthreads) {
    for (int i = tid; i < BR * SEGS; i += nthreads) {
        const int r = i / SEGS, seg = (i % SEGS) * VEC;
        if (g0 + r < N) {
            const float* s = src + r * D_HEAD + seg;
            uint4 out;
            __half2* oh = reinterpret_cast<__half2*>(&out);
#pragma unroll
            for (int t = 0; t < 4; ++t)
                oh[t] = __floats2half2_rn(s[2 * t], s[2 * t + 1]);
            *reinterpret_cast<uint4*>(dst + (long)(g0 + r) * row_stride + seg) = out;
        }
    }
}

}  // namespace fav100

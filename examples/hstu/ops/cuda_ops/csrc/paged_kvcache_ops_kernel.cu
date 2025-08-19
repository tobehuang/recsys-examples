/******************************************************************************
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Implementation based on FlashInfer library.
# 
******************************************************************************/

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <driver_types.h>
#include <iostream>
#include "vec_dtypes.cuh"

/**
 * 参数介绍：
 *    head_dim: 这是一个变量，表示传入的头部维度（head dimension）的值，它在运行时确定。
 *    HEAD_DIM: 这是一个编译时常量，在宏内部通过constexpr定义。它的值根据head_dim的值而设定为对应的常数（64, 128, 256, 512）
 *    ...: 这是一个可变参数（__VA_ARGS__），表示用户在调用宏时可以传入任意代码片段。这些代码会在对应的case中执行
 * 
 * 执行过程：
 *    switch-case结构​：根据head_dim的值，选择不同的case分支。
 *    在每个case中，使用constexpr定义一个新的常量HEAD_DIM，其值等于该case对应的头维度
 *    在每个case中，执行用户传入的代码（通过__VA_ARGS__插入）
 */
#define DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, ...)       \
  switch (head_dim) {                                    \
    case 64: {                                           \
      constexpr size_t HEAD_DIM = 64;                    \
      __VA_ARGS__                                        \
      break;                                             \
    }                                                    \
    case 128: {                                          \
      constexpr size_t HEAD_DIM = 128;                   \
      __VA_ARGS__                                        \
      break;                                             \
    }                                                    \
    case 256: {                                          \
      constexpr size_t HEAD_DIM = 256;                   \
      __VA_ARGS__                                        \
      break;                                             \
    }                                                    \
    case 512: {                                          \
      constexpr size_t HEAD_DIM = 512;                   \
      __VA_ARGS__                                        \
      break;                                             \
    }                                                    \
    default: {                                           \
      std::cerr << "Unsupported head_dim: " << head_dim; \
      return cudaErrorInvalidValue;                      \
    }                                                    \
  }

/**
 * @brief Get the uint fastdiv msa object
 *  用于计算无符号整数除法的快速替换参数​（魔数、移位值和调整标志）。这些参数使得可以在不使用实际除法指令的情况下，通过乘法和移位操作高效地执行除法。
 *  这在需要高性能计算的场景中非常有用（如图形编程、游戏引擎、低延迟系统等）。
 * 
 *  函数输入参数 d 是除数（除数必须大于1），输出参数 m, s, a 用于后续的快速除法
 *  被除数 n 除以 d 可以近似为：result = ((n * m) >> (32+s)) + a; // 然后可能再调整
 * 
 * @param d 输入除数
 * @param m 魔数，输出参数（通过引用返回
 * @param s 移位次数，输出参数（通过引用返回
 * @param a 一个调整标志，输出参数（通过引用返回
 */
void get_uint_fastdiv_msa(uint32_t d, uint32_t &m, uint32_t &s, uint32_t &a) {
    unsigned int p, nc, delta, q1, r1, q2, r2;
    a = 0;
    nc = unsigned(-1) - unsigned(-d) % d;
    p = 31;
    q1 = 0x80000000 / nc;
    r1 = 0x80000000 - q1 * nc;
    q2 = 0x7FFFFFFF / d;
    r2 = 0x7FFFFFFF - q2 * d;
    do {
      p++;
      if (r1 >= nc - r1) {
        q1 = 2 * q1 + 1;
        r1 = 2 * r1 - nc;
      } else {
        q1 = 2 * q1;
        r1 = 2 * r1;
      }
      if (r2 + 1 >= d - r2) {
        if (q2 >= 0x7FFFFFFF) a = 1;
        q2 = 2 * q2 + 1;
        r2 = 2 * r2 + 1 - d;
      } else {
        if (q2 >= 0x80000000) a = 1;
        q2 = 2 * q2;
        r2 = 2 * r2 + 1;
      }
      delta = d - 1 - r2;
    } while (p < 64 && (q1 < delta || (q1 == delta && r1 == 0)));
    m = q2 + 1;
    s = p - 32;
}

/**
 * @brief 用于同时计算无符号整数的商（q）和余数（r）。
 * 该函数利用预先计算的魔术参数（m, s, a）来避免使用昂贵的除法指令。函数可以在主机（CPU）和设备（GPU）上执行。
 * 此函数实现了比直接使用除法快 ​5-20 倍​ 的性能提升，在并行计算场景中可带来显著的加速效果。
 * 
 * @param n 被除数（dividend）
 * @param d 除数（divisor）
 * @param m, s, a: 预先为除数d计算的魔术参数（通过get_uint_fastdiv_msa函数得到）
 * @param q 计算得到的商（quotient）
 * @param r 计算得到的余数（remainder）
 */
__host__ __device__ __forceinline__ void divmod(uint32_t n, uint32_t d,
                                                uint32_t m, uint32_t s, uint32_t a, 
                                                uint32_t& q, uint32_t& r) 
{
    if (d == 1) {
        q = n;
    } else {
#ifdef __CUDA_ARCH__
        q = __umulhi(m, n);
#else
        q = (((unsigned long long)((long long)m * (long long)n)) >> 32);
#endif
        q += a * n;
        q >>= s;
    }
    r = n - q * d;
}

/**
 * @brief 这是一个用于 ​分页键值缓存（Paged Key-Value Cache）​​ 的 CUDA 核函数，用于高效地将新生成的键值对追加到缓存中。
 *  这种技术在注意力机制（尤其是 Transformer 类模型）中非常关键，用于存储历史键值对，同时支持动态扩展。
 * 
 * @tparam head_dim 每个注意力头的维度（固定大小，编译时常量）
 * @tparam vec_size 向量化加载/存储的宽度（例如，一次拷贝多少个元素）
 * @tparam DType 数据类型（例如float, half）
 * @tparam IdType 索引类型（例如int, long）
 * @param k_data 指向现有k缓存的指针（位于GPU内存），我们将把新数据追加到这里。
 * @param v_data 指向现有v缓存的指针（位于GPU内存），我们将把新数据追加到这里。
 * @param indices 用于管理分页的索引指针(存储实际的页索引)
 * @param indptr 指向每个样本的页索引的起始位置
 * @param num_heads 注意力头的数量
 * @param page_size ？一页可以容纳多少个键（或值）向量（即一个页面中存储的token数量）
 * @param stride_page 在k_data/v_data中，页面之间的跨度（以元素为单位）
 * @param stride_n 在页面内，相邻token之间的跨度（通常为1，或按元素连续存储）
 * @param stride_h 在页面内，相邻头之间的跨度（即一个页面内，同一个token位置的不同头的数据间隔）
 * @param append_key 存储要追加的 k 数据的指针
 * @param append_value 存储要追加的 v 数据的指针
 * @param batch_indices 每个要追加的数据点对应的批次索引
 * @param positions 每个要追加的数据点在序列中的位置（即token在序列中的位置）
 * @param offsets 每个批次中已存在多少token（即追加的起始位置偏移
 * @param nnz_cuda 要追加的总数据点数（非零元素个数，即总token数）
 * @param append_k_stride_n 新数据 k 的步长信息
 * @param append_k_stride_h 
 * @param append_v_stride_n 
 * @param append_v_stride_h 
 * @param m 
 * @param s 
 * @param a 
 * @return __global__ 
 */
template <uint32_t head_dim, uint32_t vec_size, typename DType, typename IdType>
__global__ void AppendPagedKVCacheKernel(DType* k_data, // 指向现有k缓存的指针（位于GPU内存），我们将把新数据追加到这里
                                         DType* v_data, // 指向现有v缓存的指针（位于GPU内存），我们将把新数据追加到这里
                                         IdType* indices, // 用于管理分页的索引指针(存储实际的页索引)
                                         IdType* indptr, // 指向每个样本的页索引的起始位置
                                         uint32_t num_heads,
                                         uint32_t page_size,
                                         uint32_t stride_page,
                                         uint32_t stride_n,
                                         uint32_t stride_h,
                                         DType* __restrict__ append_key,
                                         DType* __restrict__ append_value,
                                         IdType* __restrict__ batch_indices,
                                         IdType* __restrict__ positions, 
                                         IdType* __restrict__ offsets,
                                         IdType* __restrict__ nnz_cuda,
                                         size_t append_k_stride_n, size_t append_k_stride_h,
                                         size_t append_v_stride_n, size_t append_v_stride_h,
                                         uint32_t m, uint32_t s, uint32_t a) {
  uint32_t tx = threadIdx.x, ty = threadIdx.y;
  uint32_t head_idx = ty;  // 负责的注意力头
  uint32_t cta_id = blockIdx.x; // 线程块全局ID
  uint32_t num_ctas = gridDim.x; // 总线程块数

  uint32_t nnz = nnz_cuda[0]; // 这里nnz_cuda是一个只有一个元素的数组，存储待处理token总数。

  // 主循环：每个线程块处理多个追加操作（循环步长为线程块总数，以覆盖所有追加操作）。
#pragma unroll 4
  for (uint32_t i = cta_id; i < nnz; i += num_ctas) {
    uint32_t page_iter, entry_idx;
    // Step 1: 计算页面位置（用魔术数快速计算 位置/页面大小 → (页索引, 页内偏移)）
    //    positions[i]（表示当前token在整个序列中的位置）
    //    page_size 表示一个page的可存储token的长度
    //    page_iter 表示page的索引编号
    //    entry_idx 表示page内的偏移
    divmod(positions[i], page_size, m, s, a, page_iter, entry_idx);

    // Step 2: 计算内存偏移量
    //  indices[...]：获取物理页编号
    //  通过 stride_* 组合定位精确位置
    //  __ldg()：常量内存读取（缓存优化）

    //   通过batch_indices[i]得到当前token的批次索引
    //   通过indptr[batch_indices[i]]得到该批次在indices数组中的起始位置
    //   加上page_iter（第几页）得到该页在indices数组中的索引
    //   然后读取该页的实际页索引（indices[indptr[batch_indices[i]]+page_iter]），这个页索引乘以页的跨度（stride_page）得到该页在k_data/v_data中的起始位置
    size_t elem_offset = __ldg(indices + indptr[batch_indices[i]] + page_iter) * stride_page + 
      head_idx * stride_h + 
      entry_idx * stride_n + 
      tx * vec_size;
    
    // Step 3: 获取目标指针
    DType* k_ptr = k_data + elem_offset;
    DType* v_ptr = v_data + elem_offset;

    // Step4: 向量化数据拷贝（单次拷贝 vec_size 元素（减少内存事务））
    //  batch_indices[i]：确定批次
    //  offsets[]：计算全局 token 位置
    //  通过 stride_* 定位具体数据
    vec_t<DType, vec_size>::memcpy(
        k_ptr, append_key + (i + offsets[batch_indices[i]]) * append_k_stride_n + head_idx * append_k_stride_h + tx * vec_size);
    vec_t<DType, vec_size>::memcpy(
        v_ptr, append_value + (i + offsets[batch_indices[i]]) * append_v_stride_n + head_idx * append_v_stride_h + tx * vec_size);
  }
}

template <typename DType, typename IdType>
cudaError_t AppendPagedKVCache(DType* k_data,
                               DType* v_data,
                               IdType* indices,
                               IdType* indptr,
                               uint32_t num_heads,
                               uint32_t head_dim,
                               uint32_t page_size,
                               uint32_t stride_page,
                               uint32_t stride_n,
                               uint32_t stride_h,
                               DType* append_key, DType* append_value, IdType* batch_indices,
                               IdType* positions, IdType* offsets,
                               IdType* nnz_cuda, uint32_t nnz,
                               size_t append_k_stride_n, size_t append_k_stride_h,
                               size_t append_v_stride_n, size_t append_v_stride_h,
                               cudaStream_t stream) {
  int dev_id = 0;
  int num_sms = 0;
  int num_blocks_per_sm = 0;
  cudaGetDevice(&dev_id);
  cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, dev_id);

  DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {
    constexpr uint32_t vec_size = std::max(16 / sizeof(DType), HEAD_DIM / 32);
    uint32_t bdx = HEAD_DIM / vec_size;
    uint32_t bdy = num_heads;
    uint32_t num_threads = bdx * bdy;
    uint32_t smem_size = 0;
    auto kernel = AppendPagedKVCacheKernel<HEAD_DIM, vec_size, DType, IdType>;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks_per_sm, kernel,
                                                  num_threads, smem_size);
    num_blocks_per_sm = min(num_blocks_per_sm, ((int(nnz) + num_sms - 1) / num_sms));
    dim3 nblks(num_blocks_per_sm * num_sms);
    dim3 nthrs(bdx, bdy);

    uint32_t m, s, a;
    get_uint_fastdiv_msa(page_size, m, s, a);

    void* args[] = {(void*)&k_data,            (void*)&v_data,            (void*)&indices,
                    (void*)&indptr,            (void*)&num_heads,         (void*)&page_size,
                    (void*)&stride_page,       (void*)&stride_n,          (void*)&stride_h,
                    (void*)&append_key,        (void*)&append_value,      (void*)&batch_indices,
                    (void*)&positions,         (void*)&offsets,           (void*)&nnz_cuda,
                    (void*)&append_k_stride_n, (void*)&append_k_stride_h, (void*)&append_v_stride_n,
                    (void*)&append_v_stride_h, (void*)&m,                 (void*)&s,
                    (void*)&a};
    cudaLaunchKernel((void*)kernel, nblks, nthrs, args, 0, stream);
  });
  return cudaSuccess;
}

template 
cudaError_t AppendPagedKVCache<nv_bfloat16, int32_t>(
  nv_bfloat16* k_data,
  nv_bfloat16* v_data,
  int32_t* indices,
  int32_t* indptr,
  uint32_t num_heads,
  uint32_t head_dim,
  uint32_t page_size,
  uint32_t stride_page,
  uint32_t stride_n,
  uint32_t stride_h,
  nv_bfloat16* append_key, nv_bfloat16* append_value, int32_t* batch_indices,
  int32_t* positions, int32_t* offsets,
  int32_t* nnz_cuda, uint32_t nnz,
  size_t append_k_stride_n, size_t append_k_stride_h,
  size_t append_v_stride_n, size_t append_v_stride_h,
  cudaStream_t stream);

template 
cudaError_t AppendPagedKVCache<nv_half, int32_t>(
  nv_half* k_data,
  nv_half* v_data,
  int32_t* indices,
  int32_t* indptr,
  uint32_t num_heads,
  uint32_t head_dim,
  uint32_t page_size,
  uint32_t stride_page,
  uint32_t stride_n,
  uint32_t stride_h,
  nv_half* append_key, nv_half* append_value, int32_t* batch_indices,
  int32_t* positions, int32_t* offsets,
  int32_t* nnz_cuda, uint32_t nnz,
  size_t append_k_stride_n, size_t append_k_stride_h,
  size_t append_v_stride_n, size_t append_v_stride_h,
  cudaStream_t stream);


template <uint32_t head_dim, uint32_t vec_size, typename DType, typename IdType>
__global__ void GatherPagedKVCacheKernel(DType* gather_kv,
                                         IdType* page_ids,
                                         uint32_t page_size,
                                         uint32_t stride_page,
                                         uint32_t stride_k2v,
                                         uint32_t stride_n,
                                         uint32_t stride_h,
                                         uint32_t nnz,
                                         DType* __restrict__ kv_cache,
                                         uint32_t m, uint32_t s, uint32_t a) {
  uint32_t tx = threadIdx.x, ty = threadIdx.y;
  uint32_t head_idx = ty;
  uint32_t cta_id = blockIdx.x;
  uint32_t num_ctas = gridDim.x;
  DType* gather_k = gather_kv;
  DType* gather_v = gather_kv + stride_k2v;
  DType* __restrict__ k_cache = kv_cache;
  DType* __restrict__ v_cache = kv_cache + stride_k2v;

#pragma unroll 4
  for (uint32_t i = cta_id; i < nnz; i += num_ctas) {
    uint32_t page_id_idx, entry_idx;
    divmod(i, page_size, m, s, a,
           page_id_idx, entry_idx);
    size_t inner_page_offset = head_idx * stride_h + entry_idx * stride_n + tx * vec_size;
    size_t src_offset = __ldg(page_ids + page_id_idx) * stride_page + inner_page_offset;
    size_t dst_offset = page_id_idx * stride_page + inner_page_offset;
    vec_t<DType, vec_size>::memcpy(
        gather_k + dst_offset, k_cache + src_offset);
    vec_t<DType, vec_size>::memcpy(
        gather_v + dst_offset, v_cache + src_offset);
  }
}

template <typename DType, typename IdType>
cudaError_t GatherPagedKVCache(DType* gather_kv,
                               IdType* page_ids,
                               uint32_t num_heads,
                               uint32_t head_dim,
                               uint32_t page_size,
                               uint32_t stride_page,
                               uint32_t stride_k2v,
                               uint32_t stride_n,
                               uint32_t stride_h,
                               DType* kv_cache,
                               uint32_t nnz,
                               cudaStream_t stream) {
  int dev_id = 0;
  int num_sms = 0;
  int num_blocks_per_sm = 0;
  cudaGetDevice(&dev_id);
  cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, dev_id);

  DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {
    constexpr uint32_t vec_size = std::max(16 / sizeof(DType), HEAD_DIM / 32);
    uint32_t bdx = HEAD_DIM / vec_size;
    uint32_t bdy = num_heads;
    uint32_t num_threads = bdx * bdy;
    uint32_t smem_size = 0;
    auto kernel = GatherPagedKVCacheKernel<HEAD_DIM, vec_size, DType, IdType>;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks_per_sm, kernel,
                                                  num_threads, smem_size);
    num_blocks_per_sm = min(num_blocks_per_sm, ((int(nnz) + num_sms - 1) / num_sms));
    dim3 nblks(num_blocks_per_sm * num_sms);
    dim3 nthrs(bdx, bdy);

    uint32_t m, s, a;
    get_uint_fastdiv_msa(page_size, m, s, a);

    void* args[] = {(void*)&gather_kv,     (void*)&page_ids,      (void*)&page_size,    
                    (void*)&stride_page,   (void*)&stride_k2v,    (void*)&stride_n,
                    (void*)&stride_h,      (void*)&nnz,           (void*)&kv_cache,
                    (void*)&m,             (void*)&s,             (void*)&a};
    cudaLaunchKernel((void*)kernel, nblks, nthrs, args, 0, stream);
  });
  return cudaSuccess;
}

template 
cudaError_t GatherPagedKVCache<nv_bfloat16, int32_t>(
  nv_bfloat16* gather_kv,
  int32_t* page_ids,
  uint32_t num_heads,
  uint32_t head_dim,
  uint32_t page_size,
  uint32_t stride_page,
  uint32_t stride_k2v,
  uint32_t stride_n,
  uint32_t stride_h,
  nv_bfloat16* kv_cache,
  uint32_t nnz,
  cudaStream_t stream);
  
template 
cudaError_t GatherPagedKVCache<nv_half, int32_t>(
  nv_half* gather_kv,
  int32_t* page_ids,
  uint32_t num_heads,
  uint32_t head_dim,
  uint32_t page_size,
  uint32_t stride_page,
  uint32_t stride_k2v,
  uint32_t stride_n,
  uint32_t stride_h,
  nv_half* kv_cache,
  uint32_t nnz,
  cudaStream_t stream);
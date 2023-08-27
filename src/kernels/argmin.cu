#include <bits/stdc++.h>
#include <cub/cub.cuh>
#include "kernels.cuh"
#include "../utils.cuh"
#include "../include/colors.h"

__device__ Pair shfl_xor_sync (Pair p, unsigned delta){
  return Pair{
    __shfl_xor_sync(ARGMIN_SHFL_MASK, p.v, delta),
    __shfl_xor_sync(ARGMIN_SHFL_MASK, p.i, delta),
  };
}

__device__ Pair argmin (Pair a, Pair b) {
  return a.v <= b.v ? a : b;
}

__device__ Pair warp_argmin (float a) {
  Pair t{a, (uint32_t)threadIdx.x & 31};
  t = argmin(t, shfl_xor_sync(t, 1));
  t = argmin(t, shfl_xor_sync(t, 2));
  t = argmin(t, shfl_xor_sync(t, 4));
  t = argmin(t, shfl_xor_sync(t, 8));
  t = argmin(t, shfl_xor_sync(t, 16));
  return t;
}

/**
 * @brief This kernel reduces each block (one per point) to find the closest centroid (min dist.) and writes back the centroid index incrementing the cluster length
 * 
 * @param n 
 * @param k 
 * @param d_distances 
 * @param points_clusters point-cluster associations
 * @param clusters_len length of clusters
 * @param warps_per_block used to avoid useless compoutations
 * @param infty max value for DATA_TYPE
 */
__global__ void clusters_argmin_shfl(const uint32_t n, const uint32_t k, DATA_TYPE* d_distances, uint32_t* points_clusters,  uint32_t* clusters_len, uint32_t warps_per_block, DATA_TYPE infty) {
  const uint32_t warpSizeLog2 = sizeof(uint32_t) * CHAR_BIT - clz(warpSize) - 1;
  extern __shared__ Pair shrd[];
  const uint32_t tid = threadIdx.x;
  const uint32_t lane = tid & (warpSize - 1);
  const uint32_t wid = tid >> warpSizeLog2;
  const uint32_t idx = blockIdx.x * k + tid;
  float val = tid < k ? d_distances[idx] : infty;

  Pair p = warp_argmin(val);

  if (lane == 0) {
    p.i += wid << warpSizeLog2; // Remap p.i
    shrd[wid] = p;
  }
  
  __syncthreads();


  if (tid == 0) { // Intra-block reduction
    Pair* tmp = shrd;
    float minV = tmp->v;
    uint32_t minI = tmp->i;
    for (uint32_t i = 1; i < warps_per_block; i++) {
      Pair* tmp = shrd + i;
      if (tmp->v < minV) {
        minV = tmp->v;
        minI = tmp->i;
      }
    }
    points_clusters[blockIdx.x] = minI;
    atomicAdd(&clusters_len[minI], 1);
  }
}

/**
 * @brief This function uses the library CUB to perform the argmin for each point/centers
 * 
 * @param d_distances 
 * @param n 
 * @param k 
 * @param h_points_clusters indicates the cluster of each point
 * @param d_points_clusters 
 * @param h_clusters_len indicates how many point belog to each cluster
 */
void clusters_argmin_cub(const DATA_TYPE* d_distances, const uint32_t n, const uint32_t k, uint32_t* h_points_clusters, uint32_t* d_points_clusters, uint64_t* h_clusters_len) {
  memset(h_clusters_len, 0, k * sizeof(uint64_t));
  for (size_t i = 0; i < n; i++) {
    cub::KeyValuePair<int32_t, DATA_TYPE> *d_argmin = NULL;
    CHECK_CUDA_ERROR(cudaMalloc(&d_argmin, sizeof(int32_t) + sizeof(DATA_TYPE)));
    // Allocate temporary storage
    void *d_temp_storage = NULL; size_t temp_storage_bytes = 0;
    cub::DeviceReduce::ArgMin(d_temp_storage, temp_storage_bytes, d_distances, d_argmin, k);
    CHECK_CUDA_ERROR(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    
    // Run argmin-reduction
    cub::DeviceReduce::ArgMin(d_temp_storage, temp_storage_bytes, d_distances + i * k, d_argmin, k);

    int32_t argmin_idx;
    DATA_TYPE argmin_val;
    CHECK_CUDA_ERROR(cudaMemcpy(&argmin_idx, &(d_argmin->key), sizeof(int32_t), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(&argmin_val, &(d_argmin->value), sizeof(DATA_TYPE), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaFree(d_temp_storage));
    CHECK_CUDA_ERROR(cudaFree(d_argmin));
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    ++h_clusters_len[argmin_idx];
    h_points_clusters[i] = argmin_idx;
  }
  CHECK_CUDA_ERROR(cudaMemcpy(d_points_clusters, h_points_clusters, n * sizeof(uint32_t), cudaMemcpyHostToDevice));
}

void schedule_argmin_kernel(const cudaDeviceProp *props, const uint32_t n, const uint32_t k, dim3 *grid, dim3 *block, uint32_t *warps_per_block, uint32_t *sh_mem) {
  dim3 argmin_grid_dim(n);
  dim3 argmin_block_dim(max(next_pow_2(k), props->warpSize));
  
  *grid   = argmin_grid_dim;
  *block  = argmin_block_dim;
  *warps_per_block = (k + props->warpSize - 1) / props->warpSize; // Ceil
  *sh_mem = (*warps_per_block) * sizeof(Pair);
}
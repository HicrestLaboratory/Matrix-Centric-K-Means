#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>

#include "kernels.cuh"
#include "../utils.cuh"

#define DEBUG_GEMM 0

/*** Warp oriented ***/

/**
 * @brief This kernel will use exactly one warp to compute the distance between a point and a centroid thus is bounded to d <= 32. It uses shuffle to perform the reduction.
 * 
 * @param distances distances will be written here
 * @param centroids 
 * @param points 
 * @param d_closest_2_pow passed as parameter to avoid useless computations
 */
__global__ void compute_distances_one_point_per_warp(DATA_TYPE* distances, const DATA_TYPE* centroids, const DATA_TYPE* points, const uint32_t d_closest_2_pow) {
  const uint64_t point_offset = blockIdx.x * blockDim.x + threadIdx.x;
  const uint64_t center_offset = blockIdx.y * blockDim.x + threadIdx.x;

  DATA_TYPE dist = points[point_offset] - centroids[center_offset];
  dist *= dist;
  
  for (int i = (d_closest_2_pow >> 2); i > 0; i >>= 1) {
    dist += __shfl_down_sync(DISTANCES_SHFL_MASK, dist, i);
  }

  if (threadIdx.x == 0) {
    distances[(blockIdx.x * gridDim.y) + blockIdx.y] = dist;
  }
}

/**
 * @brief This kernel fits as many points in one warp as possible, bounded to d <= 32. It uses shuffle to perform the reduction: similar to compute_distances_one_point_per_warp.
 * 
 * @param distances distances will be written here
 * @param centroids 
 * @param points 
 * @param points_n 
 * @param points_per_warp passed as parameter to avoid useless computations
 * @param d 
 * @param d_closest_2_pow_log2 passed as parameter to avoid useless computations
 */
__global__ void compute_distances_shfl(DATA_TYPE* distances, const DATA_TYPE* centroids, const DATA_TYPE* points, const uint32_t points_n, const uint32_t points_per_warp, const uint32_t d, const uint32_t d_closest_2_pow_log2) {
  const uint32_t point_i = (blockIdx.x * points_per_warp) + (threadIdx.x >> d_closest_2_pow_log2);
  const uint32_t center_i = blockIdx.y;
  const uint32_t d_i = threadIdx.x & (d_closest_2_pow_log2 >> 1);

  if (point_i < points_n && d_i < d) {
    DATA_TYPE dist = points[point_i * d + d_i] - centroids[center_i * d + d_i];
    dist *= dist;
    for (int i = (0b1 >> (d_closest_2_pow_log2 - 2)); i > 0; i >>= 1) {
      dist += __shfl_down_sync(DISTANCES_SHFL_MASK, dist, i);
    }
    if (d_i == 0) {
      distances[(point_i * gridDim.y) + center_i] = dist;
    }
  }
}

/*** END Warp oriented ***/



/*** Matrix multiplication ***/

/**
 * @brief Computes the associated matrices for points (row-major) and stores them in associated_matrices (column-major for cuBlas). Note: this kernel will only write relevant values (i.e. on top, left and diagonal), the other values must be already be set to 0.
 *
 * @param points in ROW major order
 * @param associated_matrices the associated matrices will be written here
 * @param d 
 * @param round to handle d > 32
 */
__global__ void compute_point_associated_matrices (const DATA_TYPE* points, DATA_TYPE* associated_matrices, const uint32_t d, const uint32_t round) {
  const uint32_t block_base = warpSize * round;
  const uint32_t p_i = blockIdx.x;
  const uint32_t d_i = block_base + threadIdx.x;
  const uint32_t d_i1 = d_i + 1;

  // If dim in the thread is greater than d, then return to avoid illegal writes
  if (d_i >= d) { return; } 

  DATA_TYPE c = points[p_i * d + d_i];
  DATA_TYPE c_11 = c * c;

  for (int i = warpSize >> 2; i > 0; i >>= 1) { // Reduce c_11
    c_11 += __shfl_down_sync(DISTANCES_SHFL_MASK, c_11, i);
  }

  const uint32_t d1 = d + 1;
  const uint32_t matrix_base_i = p_i * d1 * d1;
  if (threadIdx.x == 0) {
    atomicAdd(&associated_matrices[matrix_base_i], c_11); // Write reduced c_11
  }
  associated_matrices[matrix_base_i + d_i1] = -c;               // Write first column
  associated_matrices[matrix_base_i + (d_i1 * d1)] = -c;        // Write first row
  associated_matrices[matrix_base_i + (d_i1 * d1) + d_i1] = 1;  // Write diagonal
}

DATA_TYPE* d_tmp = NULL; // https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-gemm
uint32_t d_tmp_dim = 0;
/**
 * @brief Computes and writes to d_distances the distance of each point-center (row-major, in this order)
 * 
 * @param handle 
 * @param d1 
 * @param n 
 * @param k 
 * @param d_P the points associated matrices (n * d1d1)
 * @param d_C the matrix of centers (prefixed with 1s)
 * @param d_distances size: n * k
 */
void compute_gemm_distances (cublasHandle_t& handle, uint32_t d1, uint32_t n, uint32_t k, DATA_TYPE* d_P, DATA_TYPE* d_C, DATA_TYPE* d_distances) {
  DATA_TYPE alpha = (DATA_TYPE)1;
  DATA_TYPE beta = (DATA_TYPE)0;
  uint32_t d1d1 = d1 * d1;
  DATA_TYPE* P = d_P;
  uint32_t max_k_d1 = max(k, d1);
  DATA_TYPE h_distances[k * n];
  DATA_TYPE h_tmp[max_k_d1 * max_k_d1];
  if (d_tmp_dim <= 0 || max_k_d1 != d_tmp_dim) {
    CHECK_CUDA_ERROR(cudaMalloc(&d_tmp, max_k_d1 * max_k_d1 * sizeof(DATA_TYPE)));
  }

  for (uint32_t p_i = 0; p_i < n; ++p_i, P += d1d1) { // Iterate over points associated matrices
    #if DEBUG_GEMM
      printf("\nc\n");
      DATA_TYPE tmp_debug1[n * d1];
      CHECK_CUBLAS_ERROR(cublasGetMatrix(k, d1, sizeof(DATA_TYPE), d_C, k, tmp_debug1, k));
      printMatrixColMaj(tmp_debug1, k, d1);
      printf("\nP_%d associated matrix\n", p_i);
      DATA_TYPE tmp_debug[d1d1];
      CHECK_CUBLAS_ERROR(cublasGetMatrix(d1, d1, sizeof(DATA_TYPE), P, d1, tmp_debug, d1));
      printMatrixColMaj(tmp_debug, d1, d1);
      printf("\n");
    #endif
    
    CHECK_CUBLAS_ERROR(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, // c * P
                                    k, d1, d1, &alpha,
                                    d_C, k,
                                    P, d1,
                                    &beta, d_tmp, k));

    #if DEBUG_GEMM
      printf("\nc * P\n");
      DATA_TYPE tmp_debug2[k * d1];
      CHECK_CUBLAS_ERROR(cublasGetMatrix(k, d1, sizeof(DATA_TYPE), d_tmp, k, tmp_debug2, k));
      printMatrixColMaj(tmp_debug2, k, d1);
      printf("\n");
    #endif

    CHECK_CUBLAS_ERROR(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, // (c * P) * c^T
                                    k, k, d1, &alpha,
                                    d_tmp, k,
                                    d_C, k,
                                    &beta, d_tmp, k));
    
    
    for (size_t i = 0; i < k; i++) {
      CHECK_CUBLAS_ERROR(cublasGetMatrix(k, k, sizeof(DATA_TYPE), d_tmp, k, h_tmp, k));
      h_distances[p_i * k + i] = h_tmp[IDX2C(i, i, k)];
    }

    #if DEBUG_GEMM
      printf("Distances from P_%d\n", p_i);
      printMatrixColMaj(h_tmp, k, k);
      printf("\n----------\n");
    #endif
  }
  // Copy distances to GPU
  CHECK_CUDA_ERROR(cudaMemcpy(d_distances, h_distances, n * k * sizeof(DATA_TYPE), cudaMemcpyHostToDevice));
}

/*** END Matrix multiplication ***/
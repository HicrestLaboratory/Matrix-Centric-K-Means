#ifndef __KERNEL_ARGMIN__
#define __KERNEL_ARGMIN__

#include <stdint.h>
#include <cublas_v2.h>
#include <cusparse.h>

#include "../include/common.h"

#define DISTANCES_SHFL_MASK 0xFFFFFFFF
#define ARGMIN_SHFL_MASK    0xFFFFFFFF
#define CENTROIDS_SHFL_MASK 0xFFFFFFFF

struct Pair {
  float v;
  uint32_t i;
};

/*////// SCHEDULE FUNCTIONS ///////*/

void schedule_distances_kernel(const cudaDeviceProp *props, const uint32_t n, const uint32_t d, const uint32_t k, dim3 *grid, dim3 *block, uint32_t* max_points_per_warp);
void schedule_argmin_kernel   (const cudaDeviceProp *props, const uint32_t n, const uint32_t k, dim3 *grid, dim3 *block, uint32_t *warps_per_block, uint32_t *sh_mem);
void schedule_centroids_kernel(const cudaDeviceProp *props, const uint32_t n, const uint32_t d, const uint32_t k, dim3 *grid, dim3 *block);
void schedule_copy_diag(cudaDeviceProp * props, const int k, int * num_blocks, int * num_threads);


/*/////// KERNEL FUNCTIONS ////////*/

__global__ void compute_distances_one_point_per_warp(DATA_TYPE* distances, const DATA_TYPE* centroids, const DATA_TYPE* points, const uint32_t d, const uint32_t d_closest_2_pow, const uint32_t round);
__global__ void compute_distances_shfl(DATA_TYPE* distances, const DATA_TYPE* centroids, const DATA_TYPE* points, const uint32_t points_n, const uint32_t points_per_warp, const uint32_t d, const uint32_t d_closest_2_pow_log2);

/**
 * @brief Generates associated matrices for points. 
 * 
 * @param points row-major matrix
 * @param associated_matrices the function will store here the associated matrices
 * @param d 
 */
__global__ void compute_point_associated_matrices (const DATA_TYPE* points, DATA_TYPE* associated_matrices, const uint32_t d, const uint32_t round);

__global__ void compute_p_matrix(const DATA_TYPE * d_points, DATA_TYPE * d_P,
                                            const uint32_t d, const uint32_t n, const uint32_t k,
                                            const uint32_t rounds);

__global__ void compute_c_matrix_row_major(const DATA_TYPE * d_centroids, 
                                            DATA_TYPE * d_C,
                                            const uint32_t d, const uint32_t n, 
                                            const uint32_t k,const uint32_t rounds);

__global__ void compute_c_matrix_col_major(const DATA_TYPE * d_centroids, 
                                            DATA_TYPE * d_C,
                                            const uint32_t d, const uint32_t n, 
                                            const uint32_t k,const uint32_t rounds);


__global__ void compute_c_vec(const DATA_TYPE * d_centroid,
                              DATA_TYPE * d_c_vec,
                              const uint32_t d);

__global__ void ewise_min(const DATA_TYPE * tmp,
                          DATA_TYPE * buff,
                          const uint32_t n);


void compute_gemm_distances (cublasHandle_t& handle, cudaDeviceProp *deviceProps, 
    const uint32_t d1, const uint32_t n, const uint32_t k, 
     DATA_TYPE* d_P,  DATA_TYPE* d_C, DATA_TYPE* d_distances);

void compute_gemm_distances_fast (cublasHandle_t& handle, 
    const uint32_t d, const uint32_t n, const uint32_t k, 
     DATA_TYPE* d_P,  DATA_TYPE* d_C, DATA_TYPE* d_distances);

__global__ void copy_diag(const DATA_TYPE * d_tmp, DATA_TYPE * d_distances, const int k, const int offset);
void compute_gemm_distances_free ();

void compute_spgemm_distances (cublasHandle_t& handle, cudaDeviceProp *deviceProps, 
    const uint32_t d1, const uint32_t n, const uint32_t k, 
     DATA_TYPE* d_P,  DATA_TYPE* d_C, DATA_TYPE* d_distances) = delete;

__global__ void clusters_argmin_cub(const DATA_TYPE* d_distances, const uint32_t n, const uint32_t k,  uint32_t* d_points_clusters, uint32_t* d_clusters_len);

__global__ void clusters_argmin_shfl(const uint32_t n, const uint32_t k, DATA_TYPE* d_distances, uint32_t* points_clusters,  uint32_t* clusters_len, uint32_t warps_per_block, DATA_TYPE infty, bool is_row_major);

__global__ void compute_centroids_shfl(DATA_TYPE* centroids, const DATA_TYPE* points, const uint32_t* points_clusters, const uint32_t* clusters_len, const uint64_t n, const uint32_t d, const uint32_t k, const uint32_t round);


__global__ void compute_v_matrix(DATA_TYPE * d_V,
                                 const uint32_t * d_points_clusters,
                                 const uint32_t * d_clusters_len,
                                 const uint32_t n, const uint32_t k,
                                 const uint32_t rounds);


void compute_centroids_gemm(cublasHandle_t& handle,
                            const uint32_t d, const uint32_t n, const uint32_t k,
                            const DATA_TYPE * d_V, const DATA_TYPE * d_points,
                            DATA_TYPE * d_centroids);

//GLOBAL TODO: Handle case where we can't allocate enough blocks to make this work
template <typename ClusterIter>
__global__ void compute_v_sparse(DATA_TYPE * d_vals,
                                 int32_t * d_rowinds,
                                 int32_t * d_col_offsets,
                                 ClusterIter d_points_clusters,
                                 const uint32_t * d_clusters_len,
                                 const uint32_t n)
{
    const int32_t tid = threadIdx.x + blockDim.x * blockIdx.x; 
    if (tid<n) {
        d_vals[tid] = ((DATA_TYPE) 1) / (DATA_TYPE)(d_clusters_len[d_points_clusters[tid]]);
        d_rowinds[tid] = (int32_t)d_points_clusters[tid];
        d_col_offsets[tid] = tid;
    }
    d_col_offsets[n] = n; //This might be horrible
}


void compute_centroids_spmm(cusparseHandle_t& handle,
                            const uint32_t d, const uint32_t n, const uint32_t k,
                            const DATA_TYPE * d_V_vals, 
                            const int32_t * d_V_rowinds,
                            const int32_t * d_V_col_offsets,
                            DATA_TYPE * d_centroids,
                            cusparseSpMatDescr_t& V_descr,
                            cusparseDnMatDescr_t& P_descr,
                            cusparseDnMatDescr_t& C_descr);


void check_p_correctness(DATA_TYPE * P, DATA_TYPE * points, uint32_t n, uint32_t d);
void check_c_correctness(DATA_TYPE * C, DATA_TYPE * centroids, uint32_t k, uint32_t d);


void compute_row_norm_mtx(cublasHandle_t& handle,
                        const uint32_t m, const uint32_t n, const uint32_t k,
                        const DATA_TYPE * mtx,
                        DATA_TYPE * d_norms,
                        DATA_TYPE * norm_mtx);

void compute_col_norm_mtx(cublasHandle_t& handle,
                        const uint32_t m, const uint32_t n, const uint32_t k,
                        const DATA_TYPE * mtx,
                        DATA_TYPE * d_norms,
                        DATA_TYPE * norm_mtx);

__global__ void compute_norm_mtx(const uint32_t m, const uint32_t n,  
                                    const DATA_TYPE * mtx,
                                    const uint32_t d_closest_2_pow_log2,
                                    DATA_TYPE * d_norms,
                                    const uint32_t round);


__global__ void add_norm_mtx_row(const uint32_t m, const uint32_t n,
                             const DATA_TYPE * norm_mtx, DATA_TYPE * mtx);


__global__ void add_norm_mtx_col(const uint32_t m, const uint32_t n,
                             const DATA_TYPE * norm_mtx, DATA_TYPE * mtx);

void compute_gemm_distances_arizona(cublasHandle_t& handle,
                                    const uint32_t d, const uint32_t n, const uint32_t k,
                                    const DATA_TYPE * d_points, const DATA_TYPE * d_points_norms,
                                    const DATA_TYPE * d_centroids, const DATA_TYPE * d_centroids_norms,
                                    DATA_TYPE * d_distances);



__global__ void check_convergence( const DATA_TYPE * d_centroids,
                                    const DATA_TYPE * d_last_centroids,
                                    const uint32_t d,
                                    const uint32_t k,
                                    const uint32_t next_pow_of2,
                                    const DATA_TYPE tol,
                                    bool is_row_maj,
                                    int * result);

#endif

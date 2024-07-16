#ifndef __KERNEL_ARGMIN__
#define __KERNEL_ARGMIN__

#include <stdint.h>
#include <cublas_v2.h>
#include <cusparse.h>
#include <random>
#include <unordered_set>

#include "../include/common.h"
#include "../cuda_utils.cuh"

#include "kernel_functions.cuh"

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
 *cublasHandle,  @brief Generates associated matrices for points. 
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

__global__ void copy_diag(const DATA_TYPE * d_M, DATA_TYPE * d_output,
                          const int m, const int n);
__global__ void copy_diag_scal(const DATA_TYPE * d_M, DATA_TYPE * d_output,
                          const int m, const int n,
                          const DATA_TYPE alpha);

__global__ void scale_diag(DATA_TYPE * d_M, const uint32_t n, const DATA_TYPE alpha);

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

__global__ void prune_centroids(const DATA_TYPE * d_new_centroids,
                                DATA_TYPE * d_centroids,
                                const uint32_t * d_stationary,
                                const uint32_t * d_offsets,
                                const uint32_t d, const uint32_t k,
                                const uint32_t k_pruned);


template <typename KV>
__global__ void scale_clusters_and_argmin(KV d_clusters,
                                       const KV d_clusters_prev,
                                       uint32_t * d_offsets,
                                       const uint32_t n)
{
    const uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) {
        const uint32_t scaled_cluster = d_clusters[tid].key + d_offsets[d_clusters[tid].key];
        const uint32_t prev_cluster = d_clusters_prev[tid].key;
        d_clusters[tid].key = (d_clusters[tid].value < d_clusters_prev[tid].value) ? scaled_cluster : prev_cluster;
    }
}

template <typename KV>
__global__ void scale_clusters(KV d_clusters,
                               uint32_t * d_offsets,
                               const uint32_t n)
{
    const uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) {
        d_clusters[tid].key += d_offsets[d_clusters[tid].key];
    }
}



template <typename KV>
__global__ void kvpair_argmin(KV * vec1, const KV * vec2, const uint32_t n)
{
    const uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) {
        KV kv1 = vec1[tid];
        KV kv2 = vec2[tid];
        vec1[tid] = (kv1.key <= kv2.key) ? kv1 : kv2;
    }
}


void compute_centroids_gemm(cublasHandle_t& handle,
                            const uint32_t d, const uint32_t n, const uint32_t k,
                            const DATA_TYPE * d_V, const DATA_TYPE * d_points,
                            DATA_TYPE * d_centroids);

// CSC
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


// Blocked ELLPACK
// Each thread block should be responsible for points that live in a 2D partition of the 
// logical view of the matrix
template <typename ClusterIter>
__global__ void compute_v_blocked(const uint32_t block_size,
                                 uint32_t * d_block_inds,
                                 DATA_TYPE * d_block_vals,
                                 ClusterIter d_points_clusters,
                                 const uint32_t * d_clusters_len,
                                 const uint32_t n,
                                 const uint32_t k)
{
    __shared__ uint32_t block_nnz;
    block_nnz = 0;

    const uint32_t threads_per_block = blockDim.x * blockDim.y;
    const uint32_t tid = threadIdx.x + (blockIdx.x + (gridDim.x + blockDim.y)) * threads_per_block;


    if (tid < n) {
        DATA_TYPE val = 1 / (DATA_TYPE)(d_clusters_len[d_points_clusters[tid]]);
        d_block_vals[tid] = val;
        atomicAdd(&block_nnz, 1);
    }

    __syncthreads();

    if (threadIdx.x==0) {
        d_block_inds[blockIdx.y * blockDim.x + blockIdx.x] = block_nnz;
    }

}


template <typename Kernel>
void init_kernel_mtx(cublasHandle_t& cublasHandle,
                     cudaDeviceProp * deviceProps,
                     const uint32_t n,
                     const uint32_t k,
                     const uint32_t d,
                     const DATA_TYPE * d_points,
                     DATA_TYPE * d_B)
{
    float b_beta = 0.0;

    if (n<=1000) {
        float b_alpha = -2.0;
        CHECK_CUBLAS_ERROR(cublasSgemm(cublasHandle, 
                                        CUBLAS_OP_T,
                                        CUBLAS_OP_N,
                                        n, n, d,
                                        &b_alpha,
                                        d_points, d,
                                        d_points, d,
                                        &b_beta,
                                        d_B, n));

        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        Kernel::function(n, d, d_B);

    } else {
        float b_alpha = 1.0;

        DATA_TYPE * d_B_tmp;
        CHECK_CUDA_ERROR(cudaMalloc(&d_B_tmp, sizeof(DATA_TYPE)*n*n)); //TODO: Just allocate space for triangular region
        CHECK_CUDA_ERROR(cudaMemset(d_B_tmp, 0, sizeof(DATA_TYPE)*n*n)); //Just in case

        CHECK_CUBLAS_ERROR(cublasSsyrk(cublasHandle,
                                       CUBLAS_FILL_MODE_LOWER,
                                       CUBLAS_OP_T,
                                       n, d, 
                                       &b_alpha,
                                       d_points, d,
                                       &b_beta,
                                       d_B_tmp, n));

        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        Kernel::function(n, d, d_B);

        b_alpha = -2.0;
        b_beta = -2.0;
        CHECK_CUBLAS_ERROR(cublasSgeam(cublasHandle,
                                       CUBLAS_OP_T,
                                       CUBLAS_OP_N,
                                       n, n,
                                       &b_alpha,
                                       d_B_tmp, n,
                                       &b_beta,
                                       d_B_tmp, n,
                                       d_B, n));
        const uint32_t scale_diag_b_block_dim = std::min((uint32_t)deviceProps->maxThreadsPerBlock, n);
        const uint32_t scale_diag_b_grid_dim = static_cast<float>(n)/static_cast<float>(scale_diag_b_block_dim);

        scale_diag<<<scale_diag_b_grid_dim, scale_diag_b_block_dim>>>(d_B, n, 0.5);

        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        CHECK_CUDA_ERROR(cudaFree(d_B_tmp));
    }

    
}
                    


void compute_centroids_spmm(cusparseHandle_t& handle,
                            const uint32_t d, const uint32_t n, const uint32_t k,
                            DATA_TYPE * d_centroids,
                            cusparseSpMatDescr_t& V_descr,
                            cusparseDnMatDescr_t& P_descr,
                            cusparseDnMatDescr_t& C_descr);


void check_p_correctness(DATA_TYPE * P, DATA_TYPE * points, uint32_t n, uint32_t d);
void check_c_correctness(DATA_TYPE * C, DATA_TYPE * centroids, uint32_t k, uint32_t d);


template <typename Distribution>
void init_centroid_selector(const uint32_t s,
                            const uint32_t n,
                            const uint32_t d,
                            Distribution& distr,
                            DATA_TYPE * d_F_vals,
                            int32_t * d_F_colinds,
                            int32_t * d_F_rowptrs,
                            cusparseSpMatDescr_t * F_descr)
{
    /* Generate k distinct random point indices for the initial centroid set */
    std::unordered_set<uint32_t> found;
    found.reserve(s);
    std::random_device rd;
    std::mt19937 gen(rd());

    std::vector<uint32_t> colinds(s);
    while (found.size() < s) 
    {
        int curr = distr(gen);
        if (found.find(curr) == found.end()) {
            colinds[found.size()] = curr;
            found.insert(curr);
        }
    }


    std::vector<DATA_TYPE> vals(s);
    std::fill(vals.begin(), vals.end(), 1);

    std::vector<uint32_t> rowptrs(s+1);
    std::iota(rowptrs.begin(), rowptrs.end(), 0);

    (cudaMemcpy(d_F_vals, vals.data(), sizeof(DATA_TYPE)*s, cudaMemcpyHostToDevice));
    (cudaMemcpy(d_F_colinds, colinds.data(), sizeof(uint32_t)*s, cudaMemcpyHostToDevice));
    (cudaMemcpy(d_F_rowptrs, rowptrs.data(), sizeof(uint32_t)*(s+1), cudaMemcpyHostToDevice));
    
    (cusparseCreateCsr(F_descr,
                        s, n, s,
                        d_F_rowptrs,
                        d_F_colinds,
                        d_F_vals,
                        CUSPARSE_INDEX_32I,
                        CUSPARSE_INDEX_32I,
                        CUSPARSE_INDEX_BASE_ZERO,
                        CUDA_R_32F));

}



__global__ void find_stationary_clusters(const uint32_t n,
                              const uint32_t k,
                              const int32_t * d_clusters_mask, 
                              const uint32_t * d_clusters, const uint32_t * d_clusters_prev,
                              uint32_t * d_stationary_clusters);

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

__global__ void add_norms_centroids(const uint32_t m, const uint32_t n,
                                     const DATA_TYPE * norms, DATA_TYPE * mtx);

__global__ void add_norms_points(const uint32_t m, const uint32_t n,
                                     const DATA_TYPE * norms, DATA_TYPE * mtx);

void compute_gemm_distances_arizona(cublasHandle_t& handle,
                                    const uint32_t d, const uint32_t n, const uint32_t k,
                                    const DATA_TYPE * d_points, const DATA_TYPE * d_points_norms,
                                    const DATA_TYPE * d_centroids, const DATA_TYPE * d_centroids_norms,
                                    DATA_TYPE * d_distances);


void compute_distances_spmm(const cusparseHandle_t& handle,
                                        const uint32_t d, 
                                        const uint32_t n,
                                        const uint32_t k,
                                        const DATA_TYPE * d_points_row_norms,
                                        const DATA_TYPE * d_centroids_row_norms,
                                        const cusparseDnMatDescr_t& B,
                                        const cusparseSpMatDescr_t& V,
                                        cusparseDnMatDescr_t& D,
                                        DATA_TYPE * d_distances);

void compute_distances_spmm_no_centroids(const cusparseHandle_t& handle,
                                        const uint32_t d, 
                                        const uint32_t n,
                                        const uint32_t k,
                                        const DATA_TYPE * d_points_row_norms,
                                        DATA_TYPE * d_centroids_row_norms,
                                        const cusparseDnMatDescr_t& B,
                                        const cusparseSpMatDescr_t& V,
                                        cusparseDnMatDescr_t& D,
                                        cusparseDnMatDescr_t& C,
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

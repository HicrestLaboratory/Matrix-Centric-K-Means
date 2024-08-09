#ifndef __KERNEL_ARGMIN__
#define __KERNEL_ARGMIN__

#include <stdint.h>
#include <cublas_v2.h>
#include <cusparse.h>
#include <random>
#include <iostream>
#include <unordered_set>

#include "../include/common.h"
#include "../cuda_utils.cuh"
#include <thrust/device_ptr.h>
#include <thrust/binary_search.h>



#define DISTANCES_SHFL_MASK 0xFFFFFFFF
#define ARGMIN_SHFL_MASK    0xFFFFFFFF
#define CENTROIDS_SHFL_MASK 0xFFFFFFFF

struct Pair {
  float v;
  uint32_t i;
};

struct Kvpair
{
    uint32_t key;
    uint32_t value;
};


enum class KernelMtxMethod {
    KERNEL_MTX_NAIVE,
    KERNEL_MTX_GEMM,
    KERNEL_MTX_SYRK
};


__global__ void sigmoid(const uint32_t n,
                               DATA_TYPE * d_B,
                               const DATA_TYPE gamma,
                               const DATA_TYPE coef);

__global__ void polynomial(const uint32_t n,
                                   DATA_TYPE * d_B,
                                   const DATA_TYPE gamma,
                                   const DATA_TYPE coef,
                                   const DATA_TYPE deg);
__global__ void polynomial_inv(const uint32_t n,
                                   DATA_TYPE * d_B,
                                   const DATA_TYPE gamma,
                                   const DATA_TYPE coef,
                                   const uint32_t deg);

__global__ void rbf(const uint32_t n,
                    DATA_TYPE * d_B,
                    const DATA_TYPE gamma);


__global__ void linear(const uint32_t n,
                       DATA_TYPE * d_B);


/* Kernel structs */

struct LinearKernel 
{

    static std::pair<uint32_t, uint32_t> get_grid_params(const uint32_t n) 
    {
        const uint32_t max_tpb = 1024;
        const uint32_t tpb = std::min(n*n, max_tpb);
        const uint32_t blocks = std::ceil( static_cast<float>(n*n) / static_cast<float>(tpb) );
        return {blocks, tpb};
    }

    static void function(const uint32_t n,
                         const uint32_t d,
                         DATA_TYPE * d_B)
    {
        auto params = get_grid_params(n);

        linear<<<params.first, params.second>>>(n, d_B);
    }
};


//TODO: change other kernel functions to have a op struct like polynomial kernel 
struct SigmoidKernel 
{

    static std::pair<uint32_t, uint32_t> get_grid_params(const uint32_t n) 
    {
        const uint32_t max_tpb = 1024;
        const uint32_t tpb = std::min(n*n, max_tpb);
        const uint32_t blocks = std::ceil( static_cast<float>(n*n) / static_cast<float>(tpb) );
        return {blocks, tpb};
    }

    static void function(const uint32_t n,
                         const uint32_t d,
                         DATA_TYPE * d_B)
    {


        const DATA_TYPE gamma = 0.5;// / static_cast<float>(d);
        const DATA_TYPE coef = 1;

        auto params = get_grid_params(n);

        sigmoid<<<params.first, params.second>>>(n, d_B, gamma, coef);

    }

};



struct PolynomialKernel 
{
    static std::pair<uint64_t, uint64_t> get_grid_params(const uint32_t n) 
    {
        const uint32_t max_tpb = 1024;
        const uint64_t tpb = std::min(n*n, max_tpb);
        const uint64_t blocks = std::ceil( static_cast<double>(n*n) / static_cast<double>(tpb) );
        return {blocks, tpb};
    }

    struct PolynomialUnaryOp
    {
        __host__ __device__
        DATA_TYPE operator()(const DATA_TYPE& elem)
        {
            return -2.0*powf(elem + 1, 2);
        }
    };


    static void function(const unsigned long long n,
                         const uint32_t d,
                         DATA_TYPE * d_B)
    {

        //TODO: parameterize w/ gamma and c 
        thrust::device_ptr<DATA_TYPE> d_B_ptr(d_B);
        unsigned long long offset = static_cast<unsigned long long>(n)*static_cast<unsigned long long>(n);
        thrust::transform(d_B_ptr, d_B_ptr+offset, d_B_ptr, PolynomialUnaryOp());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    }

};


struct RBFKernel 
{
    static std::pair<uint32_t, uint32_t> get_grid_params(const uint32_t n) 
    {
        const uint32_t max_tpb = 1024;
        const uint32_t tpb = std::min(n*n, max_tpb);
        const uint32_t blocks = std::ceil( static_cast<float>(n*n) / static_cast<float>(tpb) );
        return {blocks, tpb};
    }

    static void function(const uint32_t n,
                         const uint32_t d,
                         DATA_TYPE * d_B)
    {


        const DATA_TYPE gamma = 1/(DATA_TYPE)d; /// static_cast<float>(d);

        auto params = get_grid_params(n);

        rbf<<<params.first, params.second>>>(n, d_B, gamma);

        cudaDeviceSynchronize();

    }

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



// CSC permuted 
template <typename ClusterIter>
__global__ void compute_v_sparse_csc_permuted(DATA_TYPE * d_vals,
                                             int32_t * d_rowinds,
                                             int32_t * d_col_offsets,
                                             ClusterIter d_points_clusters,
                                             uint32_t * d_clusters_len,
                                             uint32_t * d_perm_vec,
                                             const uint32_t n)
{
    const int32_t tid = threadIdx.x + blockDim.x * blockIdx.x; 
    if (tid < n) {
        unsigned int idx = d_perm_vec[tid];
        const uint32_t cluster = d_points_clusters[idx];
        d_vals[tid] = ((DATA_TYPE) 1) / (DATA_TYPE)(d_clusters_len[cluster]);
        d_rowinds[tid] = cluster;
        d_col_offsets[tid] = tid;
    }
    d_col_offsets[n] = n;
}


template <typename ClusterIter>
__global__ void compute_perm_vec(uint32_t * d_perm_vec,
                                 ClusterIter d_points_clusters,
                                 uint32_t * d_clusters_offsets,
                                 const uint32_t n)
{
    //TODO: Somehow avoid redundant/needless copies of rows of K
    const int32_t tid = threadIdx.x + blockDim.x * blockIdx.x; 
    if (tid < n) {
        const uint32_t cluster = d_points_clusters[tid];
        unsigned int idx = atomicAdd(d_clusters_offsets + cluster, 1);
        d_perm_vec[idx] = tid;
    }
}



__global__ void init_z(const uint32_t n, const uint32_t k,
                       const DATA_TYPE * d_distances,
                       const int32_t * V_rowinds,
                       DATA_TYPE * d_z_vals);

__global__ void init_z_permuted(const uint32_t n, const uint32_t k,
                               const DATA_TYPE * d_distances,
                               const uint32_t * d_clusters,
                               const int32_t * V_rowinds,
                               const uint32_t * d_perm_vec,
                               DATA_TYPE * d_z_vals);

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

void compute_distances_popcorn_naive(const uint32_t d, 
                                     const uint32_t n,
                                     const uint32_t k,
                                     const DATA_TYPE * d_B,
                                     int32_t * d_clusters,
                                     const uint32_t * d_clusters_len,
                                     DATA_TYPE * d_c_norms,
                                     DATA_TYPE * d_distances);

void compute_distances_popcorn_spmm(const cusparseHandle_t& handle,
                                        const uint32_t d, 
                                        const uint32_t n,
                                        const uint32_t k,
                                        const DATA_TYPE * d_points_row_norms,
                                        const cusparseDnMatDescr_t& B,
                                        const cusparseSpMatDescr_t& V,
                                        cusparseDnMatDescr_t& D,
                                        cusparseDnMatDescr_t& C,
                                        const int32_t * d_clusters,
                                        DATA_TYPE * d_distances,
                                        int level);

void compute_distances_popcorn_spmv(const cusparseHandle_t& handle,
                                        const uint32_t d, 
                                        const uint32_t n,
                                        const uint32_t k,
                                        const DATA_TYPE * d_points_row_norms,
                                        const cusparseDnMatDescr_t& B,
                                        const cusparseSpMatDescr_t& V,
                                        cusparseDnMatDescr_t& D,
                                        cusparseDnVecDescr_t& c_tilde,
                                        cusparseDnVecDescr_t& z,
                                        const uint32_t * d_perm_vec,
                                        const int32_t * d_clusters,
                                        DATA_TYPE * d_distances,
                                        bool do_reorder);

__global__ void scale_diag(DATA_TYPE * d_M, const uint32_t n, const DATA_TYPE alpha);


__global__ void compute_kernel_matrix_naive(DATA_TYPE * d_K, 
                                            const DATA_TYPE * d_P, 
                                            const unsigned long long n, 
                                            const uint32_t d, 
                                            const uint32_t d_closest_2_pow);

__global__ void compute_kernel_matrix_naive_blockreduce(DATA_TYPE * d_K, 
                                                        const DATA_TYPE * d_P, 
                                                        const unsigned long long n, 
                                                        const unsigned long long d, 
                                                        const unsigned long long d_closest_2_pow);



template <typename Kernel>
void init_kernel_mtx_naive(cublasHandle_t& cublasHandle,
                         cudaDeviceProp * deviceProps,
                         const unsigned long long n,
                         const uint32_t k,
                         const uint32_t d,
                         const DATA_TYPE * d_points,
                         DATA_TYPE * d_B)
{
    const unsigned long long d_pow2 = pow(2, ceil(log2(d)));

    if (n > d) {
        const uint32_t tpb = 1024;
        const uint32_t wpb = tpb / 32;
        const uint64_t blocks = ceil((unsigned long long )(n*n) / (double)wpb);

        compute_kernel_matrix_naive<<<blocks, tpb>>>(d_B, d_points, n, d, d_pow2);
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    } else {
        const uint32_t tpb = 128;
        const uint64_t blocks = ceil((unsigned long long )(n*n) );

        compute_kernel_matrix_naive_blockreduce<<<blocks, tpb>>>(d_B, d_points, n, d, d_pow2);
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    }

    Kernel::function(n, d, d_B);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}


template <typename Kernel>
void init_kernel_mtx_gemm(cublasHandle_t& cublasHandle,
                         cudaDeviceProp * deviceProps,
                         const unsigned long long n,
                         const uint32_t k,
                         const uint32_t d,
                         const DATA_TYPE * d_points,
                         DATA_TYPE * d_B)
{
    DATA_TYPE b_beta = 0.0;
    DATA_TYPE b_alpha = 1.0;

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
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}

template <typename Kernel>
void init_kernel_mtx_syrk(cublasHandle_t& cublasHandle,
                         cudaDeviceProp * deviceProps,
                         const unsigned long long n,
                         const uint32_t k,
                         const uint32_t d,
                         const DATA_TYPE * d_points,
                         DATA_TYPE * d_B)
{
    DATA_TYPE b_beta = 0.0;
    DATA_TYPE b_alpha = 1.0;


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


    b_alpha = 1.0;
    b_beta = 1.0;
    CHECK_CUBLAS_ERROR(cublasSgeam(cublasHandle,
                                   CUBLAS_OP_T,
                                   CUBLAS_OP_N,
                                   n, n,
                                   &b_alpha,
                                   d_B_tmp, n,
                                   &b_beta,
                                   d_B_tmp, n,
                                   d_B, n));


    const uint32_t scale_diag_b_block_dim = std::min((unsigned long long )deviceProps->maxThreadsPerBlock, n);
    const uint32_t scale_diag_b_grid_dim = std::ceil(static_cast<double>(n)/static_cast<double>(scale_diag_b_block_dim));

    scale_diag<<<scale_diag_b_grid_dim, scale_diag_b_block_dim>>>(d_B, n, 0.5);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    //TODO: Make triangular version of these
    Kernel::function(n, d, d_B);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    CHECK_CUDA_ERROR(cudaFree(d_B_tmp));
    
}


template <typename Kernel>
void init_kernel_mtx(cublasHandle_t& cublasHandle,
                     cudaDeviceProp * deviceProps,
                     const unsigned long long n,
                     const uint32_t k,
                     const uint32_t d,
                     const DATA_TYPE * d_points,
                     DATA_TYPE * d_B,
                     int level)
{
    switch(level)
    {
        case NAIVE_GPU:
            init_kernel_mtx_gemm<Kernel>(cublasHandle, deviceProps,
                                          n, k, d,
                                          d_points, d_B);
            break;

        case NAIVE_MTX:
            init_kernel_mtx_gemm<Kernel>(cublasHandle, deviceProps,
                                          n, k, d,
                                          d_points, d_B);
            break;

        case OPT_MTX:
            init_kernel_mtx_syrk<Kernel>(cublasHandle, deviceProps,
                                          n, k, d,
                                          d_points, d_B);
            break;

        case REORDER:
            init_kernel_mtx_syrk<Kernel>(cublasHandle, deviceProps,
                                          n, k, d,
                                          d_points, d_B);
            break;

        case FINAL:
        {
            float ratio = static_cast<double>(n) / static_cast<double>(d);
            if (ratio > GEMM_THRESHOLD)
                init_kernel_mtx_gemm<Kernel>(cublasHandle, deviceProps,
                                              n, k, d,
                                              d_points, d_B);
            else
                init_kernel_mtx_syrk<Kernel>(cublasHandle, deviceProps,
                                              n, k, d,
                                              d_points, d_B);
        }

            
    }
}



__global__ void sum_points(const DATA_TYPE * d_K,
                            int32_t * d_clusters,
                            const uint32_t * d_clusters_len,
                            DATA_TYPE * d_distances,
                            const uint32_t n, const uint32_t k,
                            const uint32_t n_thread_ceil);

__global__ void sum_points_largek(const DATA_TYPE * d_K,
                                    int32_t * d_clusters,
                                    const uint32_t * d_clusters_len,
                                    DATA_TYPE * d_distances,
                                    const uint32_t n, const uint32_t k,
                                    const uint32_t n_thread_ceil);

__global__ void sum_centroids(const DATA_TYPE * d_K,
                            const int32_t * d_clusters,
                            const uint32_t * d_clusters_len, DATA_TYPE * d_centroids,
                            const uint32_t n, const uint32_t k);

__global__ void sum_centroids_largek(const DATA_TYPE * d_K,
                                        int32_t * d_clusters,
                                        const uint32_t * d_clusters_len,
                                        DATA_TYPE * d_centroids,
                                        const uint32_t n, const uint32_t k);

__global__ void compute_distances_naive(const DATA_TYPE * d_K,
                                        const DATA_TYPE * d_centroids,
                                        const DATA_TYPE * d_tmp,
                                        DATA_TYPE * d_distances,
                                        const uint32_t n, const uint32_t k);

__global__ void make_kvpairs(const uint32_t * d_perm_vec,
                             const uint32_t * d_perm_vec_prev,
                             const uint32_t * d_nonzero_inds,
                             Kvpair * d_perm_pairs,
                             const uint32_t n, const uint32_t nnz);

__global__ void check_convergence( const DATA_TYPE * d_centroids,
                                    const DATA_TYPE * d_last_centroids,
                                    const uint32_t d,
                                    const uint32_t k,
                                    const uint32_t next_pow_of2,
                                    const DATA_TYPE tol,
                                    bool is_row_maj,
                                    int * result);

#endif

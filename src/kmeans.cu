#include <stdio.h>
#include <vector>
#include <iomanip>
#include <cmath>
#include <ctime>
#include <limits>
#include <map>
#include <unordered_set>
#include <cublas_v2.h>

#include <raft/core/device_resources.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/resource/thrust_policy.hpp>
#include <raft/matrix/copy.cuh>
#include <raft/util/cudart_utils.hpp>

#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/sequence.h>
#include <thrust/gather.h>
#include <thrust/count.h>

#include <raft/core/device_mdarray.hpp>
#include <raft/cluster/kmeans.cuh>
#include <raft/cluster/kmeans_types.hpp>
#include <raft/core/resources.hpp>
#include <raft/linalg/map.cuh>
#include <raft/linalg/transpose.cuh>
#include <raft/linalg/coalesced_reduction.cuh>
#include <raft/linalg/reduce_cols_by_key.cuh>
#include <raft/linalg/map_then_reduce.cuh>
#include <raft/linalg/norm.cuh>
#include <raft/linalg/subtract.cuh>
#include <raft/linalg/matrix_vector.cuh>

#include <rmm/device_scalar.hpp>




#include <cstdint>
#include <optional>
#include <system_error>


#include "include/common.h"
#include "include/colors.h"

#include "cuda_utils.cuh"
#include "kmeans.cuh"

//#include "kernels/kernels.cuh"

//#define LOG_KERNEL
#define LOG 0
#define LOG_PERM 0
//#define LOG_LABELS

//std::ofstream permute_out;

using namespace std;

const DATA_TYPE INFNTY = numeric_limits<DATA_TYPE>::infinity();


Kmeans::Kmeans (const size_t _n, const uint32_t _d, const uint32_t _k, 
                const float _tol, const int* seed, 
                Point<DATA_TYPE>** _points, cudaDeviceProp* _deviceProps,
                const InitMethod _initMethod,
                const DistanceMethod _distMethod,
                const Kernel _kernel,
                const int _level)
                : n(_n), d(_d), k(_k), tol(_tol),
                POINTS_BYTES(_n * _d * sizeof(DATA_TYPE)),
                CENTROIDS_BYTES(_k * _d * sizeof(DATA_TYPE)),
                h_points_clusters(_n),
                points(_points),
                deviceProps(_deviceProps),
                initMethod(_initMethod),
                dist_method(_distMethod),
                level(_level)
{
#if LOG_PERM
    permute_out.open("permute_out.out");
#endif

    CHECK_CUBLAS_ERROR(cublasCreate(&cublasHandle));
    CHECK_CUSPARSE_ERROR(cusparseCreate(&cusparseHandle));


	CHECK_CUDA_ERROR(cudaHostAlloc(&h_points, POINTS_BYTES, cudaHostAllocDefault));
	for (size_t i = 0; i < n; ++i) {
		for (size_t j = 0; j < d; ++j) {
			h_points[i * d + j] = _points[i]->get(j);
		}
	}

    do_reorder = (k > 10) && (level>=REORDER);

#if LOG
    std::ofstream points_out;
    points_out.open("points-ours.out");
    for (int i=0; i<n; i++) {
      for (int j=0; j<d; j++) {
          points_out<<h_points[j + i*d]<<",";
      }
      points_out<<std::endl;
    }
    points_out.close();
#endif



#ifdef PERFORMANCES_MEMCPY
    cudaEvent_t e_perf_memcpy_start, e_perf_memcpy_stop;

    cudaEventCreate(&e_perf_memcpy_start);
    cudaEventCreate(&e_perf_memcpy_stop);
    cudaEventRecord(e_perf_memcpy_start);
#endif

	CHECK_CUDA_ERROR(cudaMalloc(&d_points, POINTS_BYTES));
	CHECK_CUDA_ERROR(cudaMemcpy(d_points, h_points, POINTS_BYTES, cudaMemcpyHostToDevice));


#ifdef PERFORMANCES_MEMCPY

    cudaEventRecord(e_perf_memcpy_stop);
    cudaEventSynchronize(e_perf_memcpy_stop);

    float e_perf_memcpy_ms = 0;
    cudaEventElapsedTime(&e_perf_memcpy_ms, e_perf_memcpy_start, e_perf_memcpy_stop);
    printf(CYAN "[PERFORMANCE]" RESET " memcpy time: %.8f\n", e_perf_memcpy_ms / 1000);

    cudaEventDestroy(e_perf_memcpy_start);
    cudaEventDestroy(e_perf_memcpy_stop);
#endif


    CHECK_CUDA_ERROR(cudaMalloc(&d_B, sizeof(DATA_TYPE)*n*n)); //TODO: Make this symmetric
    CHECK_CUDA_ERROR(cudaMalloc(&d_perm_vec, sizeof(uint32_t)*n));
    CHECK_CUDA_ERROR(cudaMalloc(&d_perm_vec_prev, sizeof(uint32_t)*n));
    CHECK_CUDA_ERROR(cudaMalloc(&d_clusters, n * sizeof(int32_t)));

#if PERFORMANCES_BMULT
    cudaEvent_t e_perf_bmult_start, e_perf_bmult_stop;

    cudaEventCreate(&e_perf_bmult_start);
    cudaEventCreate(&e_perf_bmult_stop);
    cudaEventRecord(e_perf_bmult_start);
#endif

    /* Init B */

    switch(_kernel)
    {
        case Kernel::linear:
            init_kernel_mtx<LinearKernel>(cublasHandle, deviceProps, n, k, d, d_points, d_B, level);
            break;

        case Kernel::polynomial:
            init_kernel_mtx<PolynomialKernel>(cublasHandle, deviceProps, n, k, d, d_points, d_B, level);
            break;

        case Kernel::sigmoid:
            init_kernel_mtx<SigmoidKernel>(cublasHandle, deviceProps, n, k, d, d_points, d_B, level);
            break;

        case Kernel::rbf:
            init_kernel_mtx<RBFKernel>(cublasHandle, deviceProps, n, k, d, d_points, d_B, level);
            break;
    }

#ifdef LOG_KERNEL
    std::ofstream kernel_out;
    kernel_out.open("kernel.out");
    DATA_TYPE * h_B = new DATA_TYPE[n*n];
    cudaMemcpy(h_B, d_B, sizeof(DATA_TYPE)*n*n, cudaMemcpyDeviceToHost);
    for (int i=0; i<10; i++) {
        for (int j=0; j<n; j++) {
            kernel_out<<h_B[j + i*n]<<",";
        }
        kernel_out<<std::endl;
    }
    kernel_out<<"...."<<std::endl;
        
    for (int i=n-10; i<n; i++) {
        for (int j=0; j<n; j++) {
            kernel_out<<h_B[j + i*n]<<",";
        }
        kernel_out<<std::endl;
    }
    kernel_out.close();
    delete[] h_B;
#endif



#if PERFORMANCES_BMULT

    cudaEventRecord(e_perf_bmult_stop);
    cudaEventSynchronize(e_perf_bmult_stop);

    float e_perf_bmult_ms = 0;
    cudaEventElapsedTime(&e_perf_bmult_ms, e_perf_bmult_start, e_perf_bmult_stop);
    printf(CYAN "[PERFORMANCE]" RESET " b-mult time: %.8f\n", e_perf_bmult_ms / 1000);

    cudaEventDestroy(e_perf_bmult_start);
    cudaEventDestroy(e_perf_bmult_stop);
#endif

	CHECK_CUDA_ERROR(cudaFree(d_points));

    /* Init matrix buffers */
    CHECK_CUDA_ERROR(cudaMalloc(&d_B_new, sizeof(DATA_TYPE)*n*n)); //TODO: Make this symmetric

    CHECK_CUDA_ERROR(cudaMalloc(&d_V_vals, sizeof(DATA_TYPE)*n));
    CHECK_CUDA_ERROR(cudaMalloc(&d_V_colinds, sizeof(int32_t)*n));
    CHECK_CUDA_ERROR(cudaMalloc(&d_V_rowptrs, sizeof(int32_t)*(k+1)));


    h_centroids_matrix = NULL;

    /* Init matrix descriptors */

    CHECK_CUSPARSE_ERROR(cusparseCreateCsr(&V_descr,
                                            k, n, n,
                                            d_V_rowptrs,
                                            d_V_colinds,
                                            d_V_vals,
                                            CUSPARSE_INDEX_32I,
                                            CUSPARSE_INDEX_32I,
                                            CUSPARSE_INDEX_BASE_ZERO,
                                            CUDA_R_32F));


    CHECK_CUSPARSE_ERROR(cusparseCreateDnMat(&B_descr,
                                            n, n, n,
                                            d_B,
                                            CUDA_R_32F,
                                            CUSPARSE_ORDER_ROW));

    if (do_reorder) {
        CHECK_CUSPARSE_ERROR(cusparseDnMatSetValues(B_descr, d_B_new));
    }

    if (level <= NAIVE_MTX) {
        CHECK_CUDA_ERROR(cudaMalloc(&d_C, sizeof(DATA_TYPE)*k*k));
    }

    CHECK_CUSPARSE_ERROR(cusparseCreateDnMat(&C_descr,
                                            k, k, k,
                                            d_C,
                                            CUDA_R_32F,
                                            CUSPARSE_ORDER_ROW));


    CHECK_CUDA_ERROR(cudaMalloc(&d_centroids_row_norms, sizeof(DATA_TYPE) * k));
    CHECK_CUDA_ERROR(cudaMalloc(&d_z_vals, sizeof(DATA_TYPE) * n));

    CHECK_CUSPARSE_ERROR(cusparseCreateDnVec(&c_tilde_descr,
                                             k, d_centroids_row_norms,
                                             CUDA_R_32F));

    CHECK_CUSPARSE_ERROR(cusparseCreateDnVec(&z_descr,
                                             n, d_z_vals,
                                             CUDA_R_32F));



#if PERFORMANCES_CENTROIDS_INIT

    cudaEvent_t e_centroid_init_start, e_centroid_init_stop;

    cudaEventCreate(&e_centroid_init_start);
    cudaEventCreate(&e_centroid_init_stop);
    cudaEventRecord(e_centroid_init_start);

#endif

    switch(initMethod)
    {
        case Kmeans::InitMethod::random:
            init_centroids_rand();
            break;
        case Kmeans::InitMethod::plus_plus:
            init_centroids_plus_plus();
            break;
        default:
            std::cerr<<"Invalid centroid initialization method"<<std::endl;
            exit(1);
    }

#if PERFORMANCES_CENTROIDS_INIT

    cudaEventRecord(e_centroid_init_stop);
    cudaEventSynchronize(e_centroid_init_stop);

    float e_centroid_init_ms = 0;
    cudaEventElapsedTime(&e_centroid_init_ms, e_centroid_init_start, e_centroid_init_stop);
    printf(CYAN "[PERFORMANCE]" RESET " init_centroids time: %.8f\n", e_centroid_init_ms / 1000);

    cudaEventDestroy(e_centroid_init_start);
    cudaEventDestroy(e_centroid_init_stop);

#endif


}

Kmeans::~Kmeans ()
{

#if LOG_PERM
    permute_out.close();
#endif

	delete generator;

	CHECK_CUDA_ERROR(cudaFreeHost(h_points));

    if (d_B != NULL) {
        CHECK_CUDA_ERROR(cudaFree(d_B));
        CHECK_CUDA_ERROR(cudaFree(d_B_new));
    }

	if (h_centroids_matrix != NULL) {
		CHECK_CUDA_ERROR(cudaFreeHost(h_centroids_matrix));
	}

    if (level <= NAIVE_MTX) {
        CHECK_CUDA_ERROR(cudaFree(d_C));
    }

    CHECK_CUDA_ERROR(cudaFree(d_V_vals));
    CHECK_CUDA_ERROR(cudaFree(d_V_colinds));
    CHECK_CUDA_ERROR(cudaFree(d_V_rowptrs));


    CHECK_CUDA_ERROR(cudaFree(d_centroids_row_norms));
    CHECK_CUDA_ERROR(cudaFree(d_z_vals));
    CHECK_CUDA_ERROR(cudaFree(d_perm_vec));
    CHECK_CUDA_ERROR(cudaFree(d_perm_vec_prev));
    CHECK_CUDA_ERROR(cudaFree(d_clusters));
    

    CHECK_CUSPARSE_ERROR(cusparseDestroyDnMat(B_descr));
    CHECK_CUSPARSE_ERROR(cusparseDestroyDnMat(C_descr));
    CHECK_CUSPARSE_ERROR(cusparseDestroyDnMat(D_descr));

    CHECK_CUSPARSE_ERROR(cusparseDestroyDnVec(c_tilde_descr));
    CHECK_CUSPARSE_ERROR(cusparseDestroyDnVec(z_descr));

    CHECK_CUSPARSE_ERROR(cusparseDestroySpMat(V_descr));

    CHECK_CUSPARSE_ERROR(cusparseDestroy(cusparseHandle));
    CHECK_CUBLAS_ERROR(cublasDestroy(cublasHandle));

	compute_gemm_distances_free();
}




/* Randomly give each point a cluster label */
void Kmeans::init_centroids_rand() 
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint32_t> distr(0, k-1);

    std::vector<int32_t> h_clusters(n);
    for (int i=0; i<n; i++) {
        h_clusters[i] = i % k;
    }
    //std::generate(h_clusters.begin(), h_clusters.end(), [&](){return distr(gen);});
    std::vector<uint32_t> h_clusters_len(k);
    std::for_each(h_clusters.begin(), h_clusters.end(), [&](auto const& cluster)mutable {h_clusters_len[cluster] += 1;});

#if LOG
    std::cout<<"CLUSTER LENS"<<std::endl;
    std::for_each(h_clusters_len.begin(), h_clusters_len.end(), [](auto elem){std::cout<<(DATA_TYPE)1/(DATA_TYPE)elem<<",";});
    std::cout<<endl;
#endif


    CHECK_CUDA_ERROR(cudaMemcpy(d_clusters, h_clusters.data(), sizeof(int32_t)*n, cudaMemcpyHostToDevice));


    CHECK_CUDA_ERROR(cudaMalloc(&d_clusters_len, k * sizeof(uint32_t)));
    CHECK_CUDA_ERROR(cudaMemcpy(d_clusters_len, h_clusters_len.data(), sizeof(uint32_t)*k, cudaMemcpyHostToDevice));


    thrust::device_vector<uint32_t> d_cluster_offsets(k);
    thrust::device_ptr<uint32_t> d_clusters_len_ptr(d_clusters_len);


    const uint32_t v_mat_block_dim = min(n, (size_t)deviceProps->maxThreadsPerBlock);
    const uint32_t v_mat_grid_dim = ceil((float)n / (float)v_mat_block_dim);
    thrust::exclusive_scan(d_clusters_len_ptr, d_clusters_len_ptr+k, d_cluster_offsets.begin());

    if (do_reorder) {

        cudaMemcpy(d_V_rowptrs,
                   thrust::raw_pointer_cast(d_cluster_offsets.data()),
                   sizeof(uint32_t)*k,
                   cudaMemcpyDeviceToDevice);

        set_perm_vec(d_clusters, thrust::raw_pointer_cast(d_cluster_offsets.data()));

        compute_v_sparse_csr_permuted<<<v_mat_grid_dim, v_mat_block_dim>>>
        (
          d_V_vals,
          d_V_colinds,
          d_V_rowptrs,
          d_clusters,
          d_clusters_len,
          thrust::raw_pointer_cast(d_cluster_offsets.data()),
          d_perm_vec,
          n, k
        );
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        //thrust::exclusive_scan(d_clusters_len_ptr, d_clusters_len_ptr+k, d_cluster_offsets.begin());

        permute_kernel_mat();

    } else if (level>0) {

        cudaMemcpy(d_V_rowptrs,
                   thrust::raw_pointer_cast(d_cluster_offsets.data()),
                   sizeof(uint32_t)*k,
                   cudaMemcpyDeviceToDevice);

        compute_v_sparse_csr<<<v_mat_grid_dim, v_mat_block_dim>>>
        (
          d_V_vals,
          d_V_colinds,
          d_V_rowptrs,
          d_clusters, d_clusters_len,
          thrust::raw_pointer_cast(d_cluster_offsets.data()),
          n, k
        );
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    }

}


void Kmeans::init_centroids_plus_plus()
{
#ifdef PLUSPLUS //This is here because otherwise this takes literally 10 MINUTES to compile
#if LOG
    std::ofstream centroids_out;
    centroids_out.open("our-centroids.out");
#endif

    /* Number of currently chosen clusters */
    uint32_t n_clusters = 0;

    /* Raft setup things */
    const raft::resources raft_handle;
    const cudaStream_t stream = raft::resource::get_cuda_stream(raft_handle);
    rmm::device_uvector<DATA_TYPE> tmp_datatype(0, stream);
    rmm::device_uvector<char> workspace(0, stream);
    std::random_device rd;
    std::mt19937 gen(rd());
    raft::random::RngState rng(rd());


    /* Number of centroids to sample each iteration */
    const uint32_t s = 2 + static_cast<uint32_t>(std::ceil(log(k)));

    /* Randomly sample initial centroid */
    std::uniform_int_distribution<> distr(0, n-1);

    int32_t first_centroid_idx = distr(gen);
    DATA_TYPE * d_first_centroid;
    CHECK_CUDA_ERROR(cudaMalloc(&d_first_centroid, sizeof(DATA_TYPE)*d));
    CHECK_CUDA_ERROR(cudaMemcpy(d_first_centroid, d_points+(first_centroid_idx*d), sizeof(DATA_TYPE)*d, cudaMemcpyDeviceToDevice));

    n_clusters++;

    /* Compute distances from each point to the first centroid,
     * technically this should probably use the Kmeans-HD distances method,
     * but this is so much easier to program and it likely doesn't matter THAT
     * much for performance since it's a single distances computation
     * with a single centroid
     */
    auto first_centroid_view = raft::make_device_matrix_view<DATA_TYPE, uint32_t>(d_first_centroid, 1, d);
    CHECK_CUDA_ERROR(cudaMemcpy(d_F_colinds, &first_centroid_idx, sizeof(int32_t), cudaMemcpyHostToDevice));

    auto points_view = raft::make_device_matrix_view<DATA_TYPE, uint32_t>(d_points, n, d);

    thrust::device_vector<DATA_TYPE> d_points_row_norms(n);
    raft::linalg::rowNorm(thrust::raw_pointer_cast(d_points_row_norms.data()),
                            d_points, d, (uint32_t)n, raft::linalg::L2Norm, true, stream);

    auto points_row_norms_view = raft::make_device_vector_view<DATA_TYPE, uint32_t>(thrust::raw_pointer_cast(d_points_row_norms.data()), n);

    auto min_distances = raft::make_device_vector<DATA_TYPE, uint32_t>(raft_handle, n);

    raft::cluster::kmeans::min_cluster_distance<DATA_TYPE, uint32_t>(raft_handle,
                                                points_view,
                                                first_centroid_view,
                                                min_distances.view(),
                                                points_row_norms_view,
                                                tmp_datatype,
                                                raft::distance::DistanceType::L2Expanded,
                                                n, 1,
                                                workspace);


    /* Setup descriptor and buffers for K */
    cusparseSpMatDescr_t K_descr;

    thrust::device_vector<DATA_TYPE> d_K_vals(s);
    thrust::fill(d_K_vals.begin(), d_K_vals.end(), 1);

    thrust::device_vector<int32_t> centroid_indices(s);
    auto centroid_indices_view = raft::make_device_vector_view<int32_t, uint32_t>(thrust::raw_pointer_cast(centroid_indices.data()),
                                                                                  s);

    thrust::device_vector<int32_t>  d_K_rowptrs(s+1);
    thrust::sequence(d_K_rowptrs.begin(), d_K_rowptrs.end(), 0);

    CHECK_CUSPARSE_ERROR(cusparseCreateCsr(&K_descr,
                                            s, n, s,
                                            thrust::raw_pointer_cast(d_K_rowptrs.data()),
                                            thrust::raw_pointer_cast(centroid_indices.data()),
                                            thrust::raw_pointer_cast(d_K_vals.data()),
                                            CUSPARSE_INDEX_32I,
                                            CUSPARSE_INDEX_32I,
                                            CUSPARSE_INDEX_BASE_ZERO,
                                            CUDA_R_32F));


    /* Setup descriptor and buffers for D_pp */
    cusparseDnMatDescr_t D_pp_descr;
    auto d_distances_span = raft::make_device_matrix<DATA_TYPE, uint32_t>(raft_handle, n, s);
    auto d_distances_transpose_span = raft::make_device_matrix<DATA_TYPE, uint32_t>(raft_handle, s, n);

    DATA_TYPE * d_distances = d_distances_span.data_handle();
    DATA_TYPE * d_distances_transpose = d_distances_transpose_span.data_handle();

    CHECK_CUSPARSE_ERROR(cusparseCreateDnMat(&D_pp_descr,
                                             s, n, s,
                                             d_distances,
                                             CUDA_R_32F,
                                             CUSPARSE_ORDER_COL));


    /* Norms of centroids, which are just norms of points here */
    thrust::device_vector<DATA_TYPE> d_centroids_row_norms(s);


    /* Stores cluster costs induced by each sampled centroid */
    DATA_TYPE * d_costs;
    CHECK_CUDA_ERROR(cudaMalloc(&d_costs, sizeof(DATA_TYPE)*s));


    /* Stores index of centroid to be added from each sample */
    rmm::device_scalar<cub::KeyValuePair<int32_t, DATA_TYPE>> min_cluster(stream);



    /* Main loop:
     * sample s centroids according to min_distances
     * init K using sampled centroid indices
     * compute next round of distances using spmm
     * choose centroid from candidates with lowest cluster cost
     * repeat until k centroids chosen
     */
    while (n_clusters < k)
    {

#if LOG

        std::vector<DATA_TYPE> h_min_distances(n);
        cudaMemcpy(h_min_distances.data(), min_distances.data_handle(),
                    sizeof(DATA_TYPE)*n, cudaMemcpyDeviceToHost);

        centroids_out<<"BEGIN MIN DISTANCES ITER "<<n_clusters-1<<std::endl;
        std::for_each(h_min_distances.begin(), h_min_distances.end(), [&](auto const& elem){centroids_out<<elem<<",";});
        centroids_out<<std::endl<<"END MIN DISTANCES ITER "<<n_clusters-1<<std::endl;
#endif

        /* Randomly sample s centroids.
         * Store the indices of the sampled centroids in centroid_indices,
         * which doubles as the colinds array of K.
         */
        raft::random::discrete<int32_t, DATA_TYPE, uint32_t>(raft_handle,
                                                              rng,
                                                              centroid_indices_view,
                                                              min_distances.view());

#if LOG
        cusparseCsrSetPointers(K_descr, thrust::raw_pointer_cast(d_K_rowptrs.data()),
                                        thrust::raw_pointer_cast(centroid_indices.data()),
                                        thrust::raw_pointer_cast(d_K_vals.data()));


        thrust::host_vector<int32_t> h_centroid_inds = centroid_indices;
        centroids_out<<"BEGIN CENTROID INDS "<<n_clusters-1<<std::endl;
        std::for_each(h_centroid_inds.begin(), h_centroid_inds.end(), [&](auto const& elem){centroids_out<<elem<<",";});
        centroids_out<<std::endl<<"END CENTROID INDS"<<n_clusters-1<<std::endl;
#endif


        /* Compute centroid norms by sampling from the previously computed row norms */
        thrust::gather(centroid_indices.begin(), centroid_indices.end(),
                        d_points_row_norms.begin(),
                        d_centroids_row_norms.begin());
#if LOG
        thrust::host_vector<DATA_TYPE> h_points_norms = d_points_row_norms;
        centroids_out<<"BEGIN POINTS NORMS  "<<n_clusters-1<<std::endl;
        std::for_each(h_points_norms.begin(), h_points_norms.end(), [&](auto const& elem){centroids_out<<elem<<",";});
        centroids_out<<std::endl<<"END POINTS NORMS"<<n_clusters-1<<std::endl;

        thrust::host_vector<DATA_TYPE> h_centroid_norms = d_centroids_row_norms;
        centroids_out<<"BEGIN CENTROID NORMS  "<<n_clusters-1<<std::endl;
        std::for_each(h_centroid_norms.begin(), h_centroid_norms.end(), [&](auto const& elem){centroids_out<<elem<<",";});
        centroids_out<<std::endl<<"END CENTROID NORMS"<<n_clusters-1<<std::endl;
#endif

        /* Compute distances from points to sampled centroids */
        compute_distances_spmm(cusparseHandle,
                                d, n, s,
                                thrust::raw_pointer_cast(d_points_row_norms.data()),
                                thrust::raw_pointer_cast(d_centroids_row_norms.data()),
                                B_descr,
                                K_descr,
                                D_pp_descr,
                                d_distances);

#if LOG
        std::vector<DATA_TYPE> h_distances(n*s);
        cudaMemcpy(h_distances.data(), d_distances,
                    sizeof(DATA_TYPE)*n*s, cudaMemcpyDeviceToHost);

        centroids_out<<"BEGIN DISTANCES ITER "<<n_clusters-1<<std::endl;
        for (int i=0; i<n; i++) {
            for (int j=0; j<s; j++) {
                centroids_out<<h_distances[j + i*s]<<",";
            }
            centroids_out<<std::endl;
        }
        centroids_out<<std::endl<<"END DISTANCES ITER "<<n_clusters-1<<std::endl;
#endif

        //TODO: This definitely isn't necessary, but for now it's fine
        // Now d_distances is [s x n]
        raft::linalg::transpose(raft_handle, d_distances_span.view(), d_distances_transpose_span.view());

        /* From here on out, this is identical to what RAFT does */

        /* Compute min distance to a sampled centroid for each point */
        raft::linalg::matrixVectorOp(d_distances_transpose,
                                     d_distances_transpose,
                                     min_distances.data_handle(),
                                     (uint32_t)n, s,
                                     true, true,
                                     raft::min_op{},
                                     stream);


        /* Compute cost induced by each centroid sample */
        raft::linalg::reduce(d_costs,
                             d_distances_transpose,
                             (uint32_t)n, s,
                             DATA_TYPE(0),
                             true, true,
                             stream);

        {

            /* Identify sampled centroid with smallest cluster cost */
            size_t tmp_bytes = 0;
            cub::DeviceReduce::ArgMin(nullptr,
                                        tmp_bytes,
                                        d_costs,
                                        min_cluster.data(),
                                        s,
                                        stream);

            workspace.resize(tmp_bytes, stream);

            cub::DeviceReduce::ArgMin(workspace.data(),
                                        tmp_bytes,
                                        d_costs,
                                        min_cluster.data(),
                                        s,
                                        stream);

            int32_t min_cluster_idx = (s+1);
            raft::copy(&min_cluster_idx, &min_cluster.data()->key, 1, stream);
            raft::resource::sync_stream(raft_handle);
#if LOG
            centroids_out<<"CENTROID IDX: "<<centroid_indices[min_cluster_idx]<<std::endl;
#endif


            /* Update min_distances */
            raft::copy(min_distances.data_handle(),
                        d_distances_transpose + (n * min_cluster_idx),
                        n,
                        stream);


            /* Add index of chosen point to colinds of F */
            //d_F_colinds_vec[n_clusters] = centroid_indices[min_cluster_idx];
            //std::for_each(d_F_colinds.begin(), d_F_colinds.end(), [&](auto const& elem) {centroids_out<<elem<<",";});
            CHECK_CUDA_ERROR(cudaMemcpy(d_F_colinds+n_clusters,
                                        thrust::raw_pointer_cast(centroid_indices.data())+min_cluster_idx,
                                        sizeof(int32_t),
                                        cudaMemcpyDeviceToDevice));


            /* Increment cluster counter */
            n_clusters++;

        }


    }

    /* Build F */
    //TODO: Do this on the device
    std::vector<int32_t> rowptrs(k+1);
    std::iota(rowptrs.begin(), rowptrs.end(), 0);

    std::vector<DATA_TYPE> vals(k);
    std::fill(vals.begin(), vals.end(), 1);


    (cudaMemcpy(d_F_vals, vals.data(), sizeof(DATA_TYPE)*k, cudaMemcpyHostToDevice));
    (cudaMemcpy(d_F_row_offsets, rowptrs.data(),
                sizeof(int32_t)*(k+1), cudaMemcpyHostToDevice));

    (cusparseCreateCsr(&F_descr,
                        k, n, k,
                        d_F_row_offsets,
                        d_F_colinds,
                        d_F_vals,
                        CUSPARSE_INDEX_32I,
                        CUSPARSE_INDEX_32I,
                        CUSPARSE_INDEX_BASE_ZERO,
                        CUDA_R_32F));

#if LOG
    centroids_out<<"CENTROIDS"<<std::endl;

    for (int i=0; i<k; i++) {
        for (int j=0; j<d; j++) {
            centroids_out<<h_centroids[d*i + j]<<",";
        }
        centroids_out<<std::endl;
    }

    centroids_out.close();
#endif


    /* Cleanup */
    CHECK_CUDA_ERROR(cudaFree(d_costs));
    CHECK_CUSPARSE_ERROR(cusparseDestroySpMat(K_descr));
    CHECK_CUSPARSE_ERROR(cusparseDestroyDnMat(D_pp_descr));

    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
#endif

}


uint64_t Kmeans::run (uint64_t maxiter, bool check_converged)
{

    thrust::device_vector<uint32_t> d_indices(n);

    uint64_t converged = maxiter;
    uint64_t iter = 0;

    const raft::resources raft_handle;
    const cudaStream_t stream = raft::resource::get_cuda_stream(raft_handle);

    rmm::device_uvector<char> workspace(0, stream);
    auto one_vec = raft::make_device_vector<uint32_t>(raft_handle, n);
    thrust::fill(raft::resource::get_thrust_policy(raft_handle),
                    one_vec.data_handle(),
                    one_vec.data_handle() + n,
                    1);


    KeyValueIndexOp<uint32_t, DATA_TYPE> conversion_op ;
    auto min_cluster_and_distance = raft::make_device_vector<raft::KeyValuePair<uint32_t, DATA_TYPE>, uint32_t>(raft_handle, n);

    //extract cluster labels from kvpair into d_points_clusters
    cub::TransformInputIterator<uint32_t,
                                KeyValueIndexOp<uint32_t, DATA_TYPE>,
                                raft::KeyValuePair<uint32_t, DATA_TYPE>*>
    clusters(min_cluster_and_distance.data_handle(), conversion_op);

    raft::KeyValuePair<uint32_t, DATA_TYPE> initial_value(0, std::numeric_limits<DATA_TYPE>::max());
    thrust::fill(raft::resource::get_thrust_policy(raft_handle),
               min_cluster_and_distance.data_handle(),
               min_cluster_and_distance.data_handle() + min_cluster_and_distance.size(),
               initial_value);
    

    DATA_TYPE* d_distances;
    CHECK_CUDA_ERROR(cudaMalloc(&d_distances, n * k * sizeof(double)));
    CHECK_CUSPARSE_ERROR(cusparseCreateDnMat(&D_descr,
                                             k, n, k,
                                             d_distances,
                                             CUDA_R_32F,
                                             CUSPARSE_ORDER_COL));

    //CHECK_CUDA_ERROR(cudaMalloc(&d_clusters_len, k * sizeof(uint32_t)));
    thrust::device_ptr<uint32_t> d_clusters_len_ptr(d_clusters_len);
    thrust::device_vector<uint32_t> d_cluster_offsets(k);

    DATA_TYPE * d_points_row_norms;
    CHECK_CUDA_ERROR(cudaMalloc(&d_points_row_norms, sizeof(DATA_TYPE)*n));

    const uint32_t diag_threads = std::min((size_t)deviceProps->maxThreadsPerBlock, n);
    const uint32_t diag_blocks = std::ceil(static_cast<float>(n) / static_cast<float>(diag_threads));
    copy_diag_scal<<<diag_blocks, diag_threads>>>(d_B, d_points_row_norms, n, n, -2.0);

    CHECK_CUDA_ERROR(cudaDeviceSynchronize());


#if LOG
    std::ofstream centroids_out;
    centroids_out.open("centroids-ours.out");
#endif

    /* MAIN LOOP */
    while (iter++ < maxiter) {
    /* COMPUTE DISTANCES */

#if PERFORMANCES_KERNEL_DISTANCES

        cudaEvent_t e_perf_dist_start, e_perf_dist_stop;

        cudaEventCreate(&e_perf_dist_start);
        cudaEventCreate(&e_perf_dist_stop);
        cudaEventRecord(e_perf_dist_start);

#endif
        if (level >= OPT_MTX) {
            compute_distances_popcorn_spmv(cusparseHandle,
                                                d, n, k,
                                                d_points_row_norms,
                                                B_descr, V_descr,
                                                D_descr, 
                                                c_tilde_descr,
                                                z_descr,
                                                d_perm_vec,
                                                d_clusters,
                                                d_distances,
                                                do_reorder);
        } else if (level == NAIVE_MTX) {
            compute_distances_popcorn_spmm(cusparseHandle,
                                                d, n, k,
                                                d_points_row_norms,
                                                B_descr, V_descr,
                                                D_descr, 
                                                C_descr,
                                                d_clusters,
                                                d_distances,
                                                level);
        } else if (level == NAIVE_GPU) {
            compute_distances_popcorn_naive(d, n, k,
                                            d_B,
                                            d_clusters,
                                            d_clusters_len,
                                            d_centroids_row_norms,
                                            d_distances);
        }


        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

#if LOG
        std::vector<DATA_TYPE> h_distances(n*k);
        cudaMemcpy(h_distances.data(), d_distances,
                    sizeof(DATA_TYPE)*n*k, cudaMemcpyDeviceToHost);

        centroids_out<<"BEGIN DISTANCES ITER "<<iter-1<<std::endl;
        for (int i=0; i<n; i++) {
            for (int j=0; j<k; j++) {
                centroids_out<<h_distances[j + i*k]<<",";
            }
            centroids_out<<std::endl;
        }
        centroids_out<<std::endl<<"END DISTANCES ITER "<<iter-1<<std::endl;
#endif

        auto pw_dist_view = raft::make_device_matrix_view<DATA_TYPE, uint32_t>(d_distances, n, k);

#if PERFORMANCES_KERNEL_DISTANCES

        cudaEventRecord(e_perf_dist_stop);
        cudaEventSynchronize(e_perf_dist_stop);

        float e_perf_dist_ms = 0;
        cudaEventElapsedTime(&e_perf_dist_ms, e_perf_dist_start, e_perf_dist_stop);
        printf(CYAN "[PERFORMANCE]" RESET " compute_distances time: %.8f\n", e_perf_dist_ms / 1000);

        cudaEventDestroy(e_perf_dist_start);
        cudaEventDestroy(e_perf_dist_stop);

#endif



		////////////////////////////////////////* ASSIGN POINTS TO NEW CLUSTERS */////////////////////////////////////////

#if PERFORMANCES_KERNEL_ARGMIN

        cudaEvent_t e_perf_argmin_start, e_perf_argmin_stop;

        cudaEventCreate(&e_perf_argmin_start);
        cudaEventCreate(&e_perf_argmin_stop);
        cudaEventRecord(e_perf_argmin_start);

#endif

        raft::linalg::coalescedReduction(
                min_cluster_and_distance.data_handle(),
                pw_dist_view.data_handle(), (uint32_t)k, (uint32_t)n,
                initial_value,
                stream,
                true,
                [= ] __device__(const DATA_TYPE val, const uint32_t i) {
                    raft::KeyValuePair<uint32_t, DATA_TYPE> pair;
                    pair.key   = i ;
                    pair.value = val;
                    return pair;
                },
                raft::argmin_op{},
                raft::identity_op{});

        CHECK_CUDA_ERROR(cudaDeviceSynchronize());


        raft::linalg::reduce_cols_by_key(one_vec.data_handle(),
                                            clusters,
                                            d_clusters_len,
                                            (uint32_t)1,
                                            (uint32_t)n,
                                            k,
                                            stream);
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

#if PERFORMANCES_KERNEL_ARGMIN

        cudaEventRecord(e_perf_argmin_stop);
        cudaEventSynchronize(e_perf_argmin_stop);

        float e_perf_argmin_ms = 0;
        cudaEventElapsedTime(&e_perf_argmin_ms, e_perf_argmin_start, e_perf_argmin_stop);

        printf(CYAN "[PERFORMANCE]" RESET " clusters_argmin_shfl time: %.8f\n", e_perf_argmin_ms / 1000);

        cudaEventDestroy(e_perf_argmin_stop);
        cudaEventDestroy(e_perf_argmin_start);

#endif

        if (level==NAIVE_GPU || true) {
            thrust::device_ptr<int32_t> d_clusters_ptr(d_clusters);
            thrust::copy(clusters, clusters+n, d_clusters_ptr);
        }

#if LOG
        /*
        thrust::device_vector<uint32_t> d_clusters(n);
        thrust::copy(clusters, clusters+n, d_clusters.begin());
        uint32_t * h_clusters = new uint32_t[n];
        cudaMemcpy(h_clusters, thrust::raw_pointer_cast(d_clusters.data()), sizeof(uint32_t)*n, cudaMemcpyDeviceToHost);
        centroids_out<<"CLUSTERS"<<std::endl;
        for (int i=0; i<n; i++) {
            centroids_out<<h_clusters[i]<<",";
        }
        centroids_out<<std::endl;
        delete[] h_clusters;
        */
#endif


		///////////////////////////////////////////* COMPUTE NEW CENTROIDS *///////////////////////////////////////////

#if PERFORMANCES_KERNEL_CENTROIDS

        cudaEvent_t e_perf_cent_start, e_perf_cent_stop;

        cudaEventCreate(&e_perf_cent_start);
        cudaEventCreate(&e_perf_cent_stop);
        cudaEventRecord(e_perf_cent_start);

#endif

        const uint32_t v_mat_block_dim = min(n, (size_t)deviceProps->maxThreadsPerBlock);
        const uint32_t v_mat_grid_dim = ceil((float)n / (float)v_mat_block_dim);

        /* Store current permutation vector before creating a new one */
        CHECK_CUDA_ERROR(cudaMemcpy(d_perm_vec_prev, d_perm_vec, sizeof(uint32_t)*n,
                                    cudaMemcpyDeviceToDevice));

        thrust::exclusive_scan(d_clusters_len_ptr, d_clusters_len_ptr+k,
                                d_cluster_offsets.begin());


        if (do_reorder) {

            cudaMemcpy(d_V_rowptrs,
                       thrust::raw_pointer_cast(d_cluster_offsets.data()),
                       sizeof(uint32_t)*k,
                       cudaMemcpyDeviceToDevice);

            set_perm_vec(clusters, thrust::raw_pointer_cast(d_cluster_offsets.data()));

            compute_v_sparse_csr_permuted<<<v_mat_grid_dim, v_mat_block_dim>>>
            (
              d_V_vals,
              d_V_colinds,
              d_V_rowptrs,
              clusters,
              d_clusters_len,
              thrust::raw_pointer_cast(d_cluster_offsets.data()),
              d_perm_vec,
              n, k
            );
            CHECK_CUDA_ERROR(cudaDeviceSynchronize());

            //thrust::exclusive_scan(d_clusters_len_ptr, d_clusters_len_ptr+k,
            //                        d_cluster_offsets.begin());


            /* After a few iterations, most points are stationary */
            if (iter > 3 && false) {
                permute_kernel_mat_swap(d_indices);
            } else {
                permute_kernel_mat();
            }

        } else if (level > 0) {

            cudaMemcpy(d_V_rowptrs,
                       thrust::raw_pointer_cast(d_cluster_offsets.data()),
                       sizeof(uint32_t)*k,
                       cudaMemcpyDeviceToDevice);

            compute_v_sparse_csr<<<v_mat_grid_dim, v_mat_block_dim>>>
            (
              d_V_vals,
              d_V_colinds,
              d_V_rowptrs,
              clusters, d_clusters_len,
              thrust::raw_pointer_cast(d_cluster_offsets.data()),
              n, k
            );

            /*
            std::vector<DATA_TYPE> h_vals(n);
            std::vector<int32_t> h_colinds(n);
            std::vector<int32_t> h_rowptrs(k+1);
            std::vector<uint32_t> h_cluster_lens(k);
            std::vector<uint32_t> h_clusters(n);

            cudaMemcpy(h_cluster_lens.data(),
                       d_clusters_len,
                       sizeof(uint32_t)*k,
                       cudaMemcpyDeviceToHost);

            thrust::copy(clusters, clusters+n, d_clusters.begin());

            cudaMemcpy(h_clusters.data(),
                       d_clusters,
                       sizeof(uint32_t)*n,
                       cudaMemcpyDeviceToHost);

            for (int i=0; i<n; i++) {
            */



            CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        }


#if PERFORMANCES_KERNEL_CENTROIDS

        cudaEventRecord(e_perf_cent_stop);
        cudaEventSynchronize(e_perf_cent_stop);

        float e_perf_cent_ms = 0;
        cudaEventElapsedTime(&e_perf_cent_ms, e_perf_cent_start, e_perf_cent_stop);

        printf(CYAN "[PERFORMANCE]" RESET " compute_centroids time: %.8f\n", e_perf_cent_ms / 1000);

        cudaEventDestroy(e_perf_cent_start);
        cudaEventDestroy(e_perf_cent_stop);
#endif


		/////////////////////////////////////////////* CHECK IF CONVERGED */////////////////////////////////////////////

        rmm::device_scalar<DATA_TYPE> d_score(stream);
        raft::cluster::detail::computeClusterCost(
                                                raft_handle,
                                                min_cluster_and_distance.view(),
                                                workspace,
                                                raft::make_device_scalar_view(d_score.data()),
                                                raft::value_op{},
                                                raft::add_op{});
        score = d_score.value(stream);
        std::cout<<std::fixed<<score<<std::endl;
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        if (iter==maxiter) {
            /*
            thrust::copy(clusters, clusters+n, d_clusters.begin());
            thrust::copy(d_clusters.begin(), d_clusters.end(), h_points_clusters.begin());
            */
            break;
        }

        if (check_converged &&
            (iter > 1) &&
            (std::abs(score-last_score) < tol)) {
            converged = iter;
            /*
            thrust::copy(clusters, clusters+n, d_clusters.begin());
            thrust::copy(d_clusters.begin(), d_clusters.end(), h_points_clusters.begin());
            */
            break;
        }

        last_score = score;

#if LOG
        centroids_out<<"END ITERATION "<<(iter-1)<<std::endl;
#endif
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

	}

#if LOG
    centroids_out.close();
#endif


#if PROFILE_MEMORY
    size_t total_mem, free_mem;
    CHECK_CUDA_ERROR(cudaMemGetInfo(&free_mem, &total_mem));
    size_t usage = (total_mem - free_mem)/1e6;
    cout<<"MEMORY FOOTPRINT: "<<usage<<" MB"<<endl;
#endif

	for (size_t i = 0; i < n; i++) {
		points[i]->setCluster(h_points_clusters[i]);
	}

#ifdef LOG_LABELS
    std::ofstream labels;
    labels.open("labels.out");

	for (size_t i = 0; i < n; i++) {
        labels<<h_points_clusters[i]<<std::endl;
    }
    
    labels.close();
#endif

	/* FREE MEMORY */
	CHECK_CUDA_ERROR(cudaFree(d_distances));
	CHECK_CUDA_ERROR(cudaFree(d_clusters_len));
    CHECK_CUDA_ERROR(cudaFree(d_points_row_norms));

	return converged;
}


template <typename ClusterIter>
void Kmeans::set_perm_vec(ClusterIter clusters,
                          uint32_t * d_cluster_offsets)
{

    /* Compute permutation vector */
    const uint32_t tpb = std::min((size_t)deviceProps->maxThreadsPerBlock, n);
    const uint32_t blocks = std::ceil(static_cast<float>(n) / static_cast<float>(tpb));
    compute_perm_vec<<<blocks, tpb>>>(d_perm_vec, clusters, d_cluster_offsets, (uint32_t)n);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

#if LOG_PERM
    std::vector<uint32_t> h_perm(n);
    cudaMemcpy(h_perm.data(), d_perm_vec, sizeof(uint32_t)*n, cudaMemcpyDeviceToHost);
    permute_out<<"PERM VEC"<<std::endl;;
    std::for_each(h_perm.begin(), h_perm.end(), [&](auto const& elem) {permute_out<<elem<<",";});
    permute_out<<std::endl;
#endif

}


void Kmeans::permute_kernel_mat()
{
    unsigned long long n2 = (unsigned long long )n*(unsigned long long )n;

    /* Use thrust fancy iterators and copying to write permuted kernel matrix to d_B */
    thrust::device_ptr<DATA_TYPE> d_B_ptr(d_B);
    thrust::device_ptr<DATA_TYPE> d_B_new_ptr(d_B_new);
    thrust::copy_n (
        thrust::make_permutation_iterator(
            d_B_ptr, 
            thrust::make_transform_iterator(
                thrust::counting_iterator<unsigned long long>(0),
                PermuteRowOp((unsigned long long)n, (unsigned long long)n, d_perm_vec)
            )
        ),
        n2,
        d_B_new_ptr
    );


    /* Set B pointer 
    CHECK_CUSPARSE_ERROR(cusparseDnMatSetValues(B_descr, d_B_new));
    std::swap(d_B, d_B_new);
    */
}


void Kmeans::permute_kernel_mat_swap(thrust::device_vector<uint32_t> d_indices)
{

    thrust::device_ptr<DATA_TYPE> d_B_ptr(d_B);
    thrust::device_ptr<DATA_TYPE> d_B_new_ptr(d_B_new);

    thrust::device_vector<uint32_t> d_moved(n);
    thrust::device_ptr<uint32_t> d_perm_ptr(d_perm_vec);
    thrust::device_ptr<uint32_t> d_perm_ptr_prev(d_perm_vec_prev);

    /* Step 1: 
     * Use d_perm_vec and d_perm_vec_prev to identify points that have 
     * changed clusters.
     * p = d_perm_vec - d_perm_vec_prev. If p[i] != 0, then point i changed clusters.
     */
    thrust::transform(d_perm_ptr, d_perm_ptr + n, 
                      d_perm_ptr_prev, d_moved.begin(),
                      check_not_equals());

    /* Step 2:
     * Compute key value pairs indicating which rows of K should be swapped.
     * This can be done using thrust::copy_if
     */
    uint32_t * h_moved  = new uint32_t[n];
    cudaMemcpy(h_moved, thrust::raw_pointer_cast(d_moved.data()), sizeof(uint32_t)*n,
                    cudaMemcpyDeviceToHost);

    for (int i=0; i<n; i++) 
        std::cout<<h_moved[i]<<std::endl;

    delete[] h_moved;

    /* Count the number of nonzero indices */
    int nnz = thrust::count_if(d_moved.begin(), d_moved.end(), is_nonzero());

    std::cout<<"NNZ: "<<nnz<<std::endl;

    /* If nnz==0, no points have moved, and we don't have to do anything */
    if (nnz==0)
        return;

    thrust::device_vector<uint32_t> d_nonzero_inds(nnz);

    /* Populate d_nonzero_inds with all the nonzero indices */
    thrust::copy_if(d_indices.begin(), d_indices.end(),
                    d_moved.begin(),
                    d_nonzero_inds.begin(),
                    is_nonzero());

    /* Create key value pairs */
    thrust::device_vector<Kvpair> d_perm_pairs(nnz);

    const uint32_t kvpair_threads = min(1024, nnz);
    const uint32_t kvpair_blocks = ceil((double)n / (double)kvpair_threads);
    make_kvpairs<<<kvpair_blocks, kvpair_threads>>>(d_perm_vec, d_perm_vec_prev,
                                                    thrust::raw_pointer_cast(d_nonzero_inds.data()),
                                                    thrust::raw_pointer_cast(d_perm_pairs.data()),
                                                    n, nnz);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    std::vector<Kvpair> h_perm_pairs(nnz);
    thrust::copy(d_perm_pairs.begin(),
                 d_perm_pairs.end(),
                 h_perm_pairs.begin());
    
    std::vector<bool> h_bitmap(n, false);


    /* Step 3:
     * Doing the swap.
     * The idea is to use the d_B buffer as a temporary storage to "cache"
     * rows that were overwritten by a swap, then, when we need the overwritten row,
     * we can copy it from d_B.
     * We store a bitmap to indicate whether or not rows are cached in the temporary buffer.
     * This is done on the host for now because presumably nnz should be quite small.
     */

    for (auto const& pair : h_perm_pairs) {

        const uint32_t i = pair.key;
        const uint32_t l = pair.value;

        /* Cache the row */
        thrust::copy_n(d_B_new_ptr+(i*n), n, d_B_ptr+(i*n));

        if (h_bitmap[l]==1) {

            /* Update bitmap */
            h_bitmap[l] = 0;

            /* Perform the swap */
            thrust::copy_n(d_B_ptr+(l*n), n, d_B_new_ptr+(i*n));

        } else {

            /* Perform the swap, but with the row of K */
            thrust::copy_n(d_B_new_ptr+(l*n), n, d_B_new_ptr+(i*n));
        }

        /* Update bitmap, since we cached row i */
        h_bitmap[i] = 1;
    }

}










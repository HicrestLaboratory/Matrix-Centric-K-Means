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
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/sequence.h>
#include <thrust/gather.h>

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

#include "kernels/kernels.cuh"


using namespace std;

const DATA_TYPE INFNTY = numeric_limits<DATA_TYPE>::infinity();

Kmeans::Kmeans (const size_t _n, const uint32_t _d, const uint32_t _k, const float _tol, const int* seed, Point<DATA_TYPE>** _points, cudaDeviceProp* _deviceProps,
const InitMethod _initMethod,
const DistanceMethod _distMethod)
		: n(_n), d(_d), k(_k), tol(_tol),
		POINTS_BYTES(_n * _d * sizeof(DATA_TYPE)),
		CENTROIDS_BYTES(_k * _d * sizeof(DATA_TYPE)),
        h_points_clusters(_n),
		points(_points),
		deviceProps(_deviceProps),
        initMethod(_initMethod),
        dist_method(_distMethod)
{

    CHECK_CUBLAS_ERROR(cublasCreate(&cublasHandle));
    CHECK_CUSPARSE_ERROR(cusparseCreate(&cusparseHandle));


	CHECK_CUDA_ERROR(cudaHostAlloc(&h_points, POINTS_BYTES, cudaHostAllocDefault));
	for (size_t i = 0; i < n; ++i) {
		for (size_t j = 0; j < d; ++j) {
			h_points[i * d + j] = _points[i]->get(j);
		}
	}

    /*
    std::ofstream points_out;
    points_out.open("points-ours.out");
    for (int i=0; i<n; i++) {
      for (int j=0; j<d; j++) {
          points_out<<h_points[j + i*d]<<",";
      }
      points_out<<std::endl;
    }
    points_out.close();
    */



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


#if PERFORMANCES_BMULT
    cudaEvent_t e_perf_bmult_start, e_perf_bmult_stop;

    cudaEventCreate(&e_perf_bmult_start);
    cudaEventCreate(&e_perf_bmult_stop);
    cudaEventRecord(e_perf_bmult_start);
#endif
    if (dist_method==Kmeans::DistanceMethod::spmm) {

        /* Init B */
        CHECK_CUDA_ERROR(cudaMalloc(&d_B, sizeof(DATA_TYPE)*n*n)); //TODO: Make this symmetric
        float b_alpha = -2.0;
        float b_beta = 0.0;


        if (n<=1000) {
            CHECK_CUBLAS_ERROR(cublasSgemm(cublasHandle, 
                                            CUBLAS_OP_T,
                                            CUBLAS_OP_N,
                                            n, n, d,
                                            &b_alpha,
                                            d_points, d,
                                            d_points, d,
                                            &b_beta,
                                            d_B, n));
        } else {

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
            const uint32_t scale_diag_b_block_dim = std::min((size_t)deviceProps->maxThreadsPerBlock, n);
            const uint32_t scale_diag_b_grid_dim = static_cast<float>(n)/static_cast<float>(scale_diag_b_block_dim);

            scale_diag<<<scale_diag_b_grid_dim, scale_diag_b_block_dim>>>(d_B, n, 0.5);

            CHECK_CUDA_ERROR(cudaDeviceSynchronize());


            CHECK_CUDA_ERROR(cudaFree(d_B_tmp));
        }

    } else {
        d_B = nullptr;
    }
#if PERFORMANCES_BMULT

    cudaEventRecord(e_perf_bmult_stop);
    cudaEventSynchronize(e_perf_bmult_stop);

    float e_perf_bmult_ms = 0;
    cudaEventElapsedTime(&e_perf_bmult_ms, e_perf_bmult_start, e_perf_bmult_stop);
    printf(CYAN "[PERFORMANCE]" RESET " b-mult time: %.8f\n", e_perf_bmult_ms / 1000);

    cudaEventDestroy(e_perf_bmult_start);
    cudaEventDestroy(e_perf_bmult_stop);
#endif


    /* Init matrix buffers */

    CHECK_CUDA_ERROR(cudaMalloc(&d_V_vals, sizeof(DATA_TYPE)*n));
    CHECK_CUDA_ERROR(cudaMalloc(&d_V_rowinds, sizeof(int32_t)*n));
    CHECK_CUDA_ERROR(cudaMalloc(&d_V_col_offsets, sizeof(int32_t)*(n+1)));

    CHECK_CUDA_ERROR(cudaMalloc(&d_F_vals, sizeof(DATA_TYPE)*k));
    CHECK_CUDA_ERROR(cudaMalloc(&d_F_colinds, sizeof(int32_t)*k));
    CHECK_CUDA_ERROR(cudaMalloc(&d_F_row_offsets, sizeof(int32_t)*(k+1)));

    h_centroids_matrix = NULL;
    CHECK_CUDA_ERROR(cudaHostAlloc(&h_centroids, CENTROIDS_BYTES, cudaHostAllocDefault));

    if (dist_method==Kmeans::DistanceMethod::gemm) {
        CHECK_CUDA_ERROR(cudaMalloc(&d_centroids, CENTROIDS_BYTES));
        CHECK_CUDA_ERROR(cudaMalloc(&d_new_centroids, CENTROIDS_BYTES));
    } else if (dist_method==Kmeans::DistanceMethod::spmm) {
        CHECK_CUDA_ERROR(cudaMalloc(&d_CC_t, k*k*sizeof(DATA_TYPE)));
    }



    /* Init matrix descriptors */

    CHECK_CUSPARSE_ERROR(cusparseCreateCsc(&V_descr,
                                            k, n, n,
                                            d_V_col_offsets,
                                            d_V_rowinds,
                                            d_V_vals,
                                            CUSPARSE_INDEX_32I,
                                            CUSPARSE_INDEX_32I,
                                            CUSPARSE_INDEX_BASE_ZERO,
                                            CUDA_R_32F));

    CHECK_CUSPARSE_ERROR(cusparseCreateDnMat(&P_descr,
                                              n, d, d,
                                              d_points,
                                              CUDA_R_32F,
                                              CUSPARSE_ORDER_ROW));

    if (dist_method==Kmeans::DistanceMethod::spmm) {

        CHECK_CUSPARSE_ERROR(cusparseCreateDnMat(&B_descr,
                                                n, n, n,
                                                d_B,
                                                CUDA_R_32F,
                                                CUSPARSE_ORDER_ROW));

        CHECK_CUSPARSE_ERROR(cusparseCreateDnMat(&C_descr,
                                                  k, k, k,
                                                  d_CC_t,
                                                  CUDA_R_32F,
                                                  CUSPARSE_ORDER_ROW));

    } else if (dist_method==Kmeans::DistanceMethod::gemm) {
        CHECK_CUSPARSE_ERROR(cusparseCreateDnMat(&C_descr,
                                                  k, d, d,
                                                  d_new_centroids,
                                                  CUDA_R_32F,
                                                  CUSPARSE_ORDER_ROW));
    }

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

	delete generator;

	CHECK_CUDA_ERROR(cudaFreeHost(h_points));
	CHECK_CUDA_ERROR(cudaFreeHost(h_centroids));

	CHECK_CUDA_ERROR(cudaFree(d_points));

    if (d_B != NULL) {
        CHECK_CUDA_ERROR(cudaFree(d_B));
    }

	if (h_centroids_matrix != NULL) {
		CHECK_CUDA_ERROR(cudaFreeHost(h_centroids_matrix));
	}

    CHECK_CUDA_ERROR(cudaFree(d_V_vals));
    CHECK_CUDA_ERROR(cudaFree(d_V_rowinds));
    CHECK_CUDA_ERROR(cudaFree(d_V_col_offsets));

    CHECK_CUDA_ERROR(cudaFree(d_F_vals));
    CHECK_CUDA_ERROR(cudaFree(d_F_colinds));
    CHECK_CUDA_ERROR(cudaFree(d_F_row_offsets));

    CHECK_CUSPARSE_ERROR(cusparseDestroyDnMat(P_descr)); 
    CHECK_CUSPARSE_ERROR(cusparseDestroyDnMat(C_descr)); 

    if (dist_method==Kmeans::DistanceMethod::spmm) {
        CHECK_CUSPARSE_ERROR(cusparseDestroyDnMat(B_descr)); 
        CHECK_CUDA_ERROR(cudaFree(d_CC_t));
    } else if (dist_method==Kmeans::DistanceMethod::gemm) {
        CHECK_CUDA_ERROR(cudaFree(d_centroids));
        CHECK_CUDA_ERROR(cudaFree(d_new_centroids));
    }

    CHECK_CUSPARSE_ERROR(cusparseDestroyDnMat(D_descr)); 
    CHECK_CUSPARSE_ERROR(cusparseDestroySpMat(V_descr)); 
    CHECK_CUSPARSE_ERROR(cusparseDestroySpMat(F_descr)); 

    CHECK_CUSPARSE_ERROR(cusparseDestroy(cusparseHandle));
    CHECK_CUBLAS_ERROR(cublasDestroy(cublasHandle));

	compute_gemm_distances_free();
}




void Kmeans::init_centroids_rand () {
    std::uniform_int_distribution<> distr(0, n-1);
    init_centroid_selector(k, n, d, distr, d_F_vals, d_F_colinds, d_F_row_offsets, &F_descr);
}


void Kmeans::init_centroids_plus_plus()
{
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

    /*
    centroids_out<<"BEGIN CENTROID INDICES"<<std::endl;
    thrust::host_vector<int32_t> h_F_colinds(k);
    CHECK_CUDA_ERROR(cudaMemcpy(thrust::raw_pointer_cast(h_F_colinds.data()), d_F_colinds, sizeof(int32_t)*k,
                                cudaMemcpyDeviceToHost));
    std::for_each(h_F_colinds.begin(), h_F_colinds.end(), [&](auto const& elem) {centroids_out<<elem<<",";});
    centroids_out<<std::endl<<"END CENTROID INDICES"<<std::endl;
    */

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

    /* One last spmm to actually compute the new centroids 
    compute_centroids_spmm(cusparseHandle,
                            d, n, k,
                            d_new_centroids,
                            F_descr,
                            P_descr,
                            C_descr);


    CHECK_CUDA_ERROR(cudaMemcpy(d_centroids, d_new_centroids, d * k * sizeof(DATA_TYPE), cudaMemcpyDeviceToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(h_centroids, d_centroids, d * k * sizeof(DATA_TYPE), cudaMemcpyDeviceToHost));
    */

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

}


uint64_t Kmeans::run (uint64_t maxiter, bool check_converged)
{

    uint64_t converged = maxiter;
    uint64_t iter = 0;

    const raft::resources raft_handle;

    const cudaStream_t stream = raft::resource::get_cuda_stream(raft_handle);
    rmm::device_uvector<char> workspace(0, stream);


    thrust::device_vector<uint32_t> d_clusters(n);


    auto one_vec = raft::make_device_vector<uint32_t>(raft_handle, n);
    thrust::fill(raft::resource::get_thrust_policy(raft_handle),
                    one_vec.data_handle(),
                    one_vec.data_handle() + n,
                    1);

    /* Number of centroids after pruning stationary centroids */
    auto d_k_pruned = raft::make_device_scalar(raft_handle, k);
    uint32_t k_pruned = k;

    KeyValueIndexOp<uint32_t, DATA_TYPE> conversion_op ;
    auto min_cluster_and_distance = raft::make_device_vector<raft::KeyValuePair<uint32_t, DATA_TYPE>, uint32_t>(raft_handle, n);

    raft::KeyValuePair<uint32_t, DATA_TYPE> initial_value(0, std::numeric_limits<DATA_TYPE>::max());

    thrust::fill(raft::resource::get_thrust_policy(raft_handle),
               min_cluster_and_distance.data_handle(),
               min_cluster_and_distance.data_handle() + min_cluster_and_distance.size(),
               initial_value);

    DATA_TYPE* d_distances;
    CHECK_CUDA_ERROR(cudaMalloc(&d_distances, n * k * sizeof(DATA_TYPE)));
    CHECK_CUSPARSE_ERROR(cusparseCreateDnMat(&D_descr,
                                             k, n, k,
                                             d_distances,
                                             CUDA_R_32F,
                                             CUSPARSE_ORDER_COL));

    uint32_t* d_clusters_len;
    CHECK_CUDA_ERROR(cudaMalloc(&d_clusters_len, k * sizeof(uint32_t)));


    DATA_TYPE * d_points_row_norms;
    DATA_TYPE * d_centroids_row_norms;

    CHECK_CUDA_ERROR(cudaMalloc(&d_points_row_norms, sizeof(DATA_TYPE)*n));
    CHECK_CUDA_ERROR(cudaMalloc(&d_centroids_row_norms, sizeof(DATA_TYPE) * k));

    if (dist_method==Kmeans::DistanceMethod::spmm) {
        const uint32_t diag_threads = std::min((size_t)deviceProps->maxThreadsPerBlock, n);
        const uint32_t diag_blocks = std::ceil(static_cast<float>(n) / static_cast<float>(diag_threads));
        copy_diag_scal<<<diag_blocks, diag_threads>>>(d_B, d_points_row_norms, n, n, -2.0);
    } else if (dist_method==Kmeans::DistanceMethod::gemm) {
        raft::linalg::rowNorm(d_points_row_norms, d_points, d, (uint32_t)n, raft::linalg::L2Norm, true, stream);
    } else {
        exit(1);
    }

    CHECK_CUDA_ERROR(cudaDeviceSynchronize());


    /* MAIN LOOP */
#if LOG
    std::ofstream centroids_out;
    centroids_out.open("centroids-ours.out");
#endif
    while (iter++ < maxiter) {
    /* COMPUTE DISTANCES */

#if LOG
        CHECK_CUDA_ERROR(cudaMemcpy(h_centroids, d_new_centroids, 
                                    d * k * sizeof(DATA_TYPE), 
                                    cudaMemcpyDeviceToHost));
        centroids_out<<"CENTROIDS"<<std::endl;

        for (int i=0; i<k; i++) {
            for (int j=0; j<d; j++) {
                centroids_out<<h_centroids[d*i + j]<<",";
            }
            centroids_out<<std::endl;
        }
#endif

#if PERFORMANCES_KERNEL_DISTANCES

        cudaEvent_t e_perf_dist_start, e_perf_dist_stop;

        cudaEventCreate(&e_perf_dist_start);
        cudaEventCreate(&e_perf_dist_stop);
        cudaEventRecord(e_perf_dist_start);

#endif


        switch(dist_method)
        {
            case Kmeans::DistanceMethod::gemm:
            {

                raft::linalg::rowNorm(d_centroids_row_norms, d_centroids, 
                                        d, k_pruned, raft::linalg::L2Norm, true, 
                                        stream);

                compute_gemm_distances_arizona(cublasHandle,
                                                d, n, k,
                                                d_points, d_points_row_norms,
                                                d_centroids, d_centroids_row_norms,
                                                d_distances);
                break;
            }

            case Kmeans::DistanceMethod::spmm:
            {
                if (iter==1) {
                    compute_distances_spmm_no_centroids(cusparseHandle, 
                                                        d, n, k_pruned,
                                                        d_points_row_norms,
                                                        d_centroids_row_norms,
                                                        B_descr, F_descr,
                                                        D_descr, C_descr,
                                                        d_distances);
                } else {
                    compute_distances_spmm_no_centroids(cusparseHandle, 
                                                        d, n, k_pruned,
                                                        d_points_row_norms,
                                                        d_centroids_row_norms,
                                                        B_descr, V_descr,
                                                        D_descr, C_descr,
                                                        d_distances);
                }
                break;
            }
        }

        CHECK_CUDA_ERROR(cudaDeviceSynchronize());


        auto pw_dist_view = raft::make_device_matrix_view<DATA_TYPE, uint32_t>(d_distances, n, k_pruned);

#if PERFORMANCES_KERNEL_DISTANCES

        cudaEventRecord(e_perf_dist_stop);
        cudaEventSynchronize(e_perf_dist_stop);

        float e_perf_dist_ms = 0;
        cudaEventElapsedTime(&e_perf_dist_ms, e_perf_dist_start, e_perf_dist_stop);
        printf(CYAN "[PERFORMANCE]" RESET " compute_distances time: %.8f\n", e_perf_dist_ms / 1000);

        cudaEventDestroy(e_perf_dist_start);
        cudaEventDestroy(e_perf_dist_stop);

#endif


#if DEBUG_KERNEL_DISTANCES

        printf(GREEN "[DEBUG_KERNEL_DISTANCES]\n");


        DATA_TYPE* cpu_distances = new DATA_TYPE[n * k];

        for (uint32_t ni = 0; ni < n; ++ni) {
            for (uint32_t ki = 0; ki < k; ++ki) {
                DATA_TYPE dist = 0, tmp;
                for (uint32_t di = 0; di < d; ++di) {
                    tmp = h_points[ni * d + di] - h_centroids[ki + di*k];
                    dist += tmp * tmp;
                }
                cpu_distances[ni * k + ki] = dist;
            }
        }

        DATA_TYPE* tmp_dist = new DATA_TYPE[n * k];

        int anyError = 0;

        CHECK_CUDA_ERROR(cudaMemcpy(tmp_dist, d_distances, 
                                    n * k * sizeof(DATA_TYPE), 
                                    cudaMemcpyDeviceToHost));

        for (uint32_t i = 0; i < n; ++i)
            for (uint32_t j = 0; j < k; ++j)
                if (fabs(tmp_dist[i * k + j] - cpu_distances[i * k + j]) > 0.001) {
                    printf("N=%-2u K=%-2u -> GPU=%.4f CPU=%.4f diff: %.8f\n", 
                            i, j, tmp_dist[i * k + j], cpu_distances[i * k + j], 
                            fabs(tmp_dist[i * k + j] - cpu_distances[i * k + j]));
                    anyError = 1;
                }
        cout << (anyError ? "Something wrong" : "Everything alright") << RESET << endl;
        delete[] cpu_distances;
        delete[] tmp_dist;
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
                pw_dist_view.data_handle(), (uint32_t)k_pruned, (uint32_t)n,
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

#if PERFORMANCES_KERNEL_ARGMIN

        cudaEventRecord(e_perf_argmin_stop);
        cudaEventSynchronize(e_perf_argmin_stop);

        float e_perf_argmin_ms = 0;
        cudaEventElapsedTime(&e_perf_argmin_ms, e_perf_argmin_start, e_perf_argmin_stop);

        printf(CYAN "[PERFORMANCE]" RESET " clusters_argmin_shfl time: %.8f\n", e_perf_argmin_ms / 1000);

        cudaEventDestroy(e_perf_argmin_stop);
        cudaEventDestroy(e_perf_argmin_start);

#endif

#if PRUNE_CENTROIDS
        cudaEvent_t e_perf_prune_start, e_perf_prune_stop;

        cudaEventCreate(&e_perf_prune_start);
        cudaEventCreate(&e_perf_prune_stop);
        cudaEventRecord(e_perf_prune_start);

        if (iter>1) {
            /* If it's past the first iteration, add offsets to min_clusters_and_distance vector so
             * the indices are correct, and do elementwise argmin with previous min_cluster_and_distance vector
             */
            const uint32_t scale_clusters_threads = min((uint32_t)n, (uint32_t)deviceProps->maxThreadsPerBlock);
            const uint32_t scale_clusters_blocks = ceil((float)n / scale_clusters_threads);
            scale_clusters_and_argmin<<<scale_clusters_blocks, scale_clusters_threads>>>
                            (min_cluster_and_distance.data_handle(), min_cluster_and_distance_prev.data_handle(),
                             d_offsets_ptr, n);

        }
#endif



        //extract cluster labels from kvpair into d_points_clusters
        cub::TransformInputIterator<uint32_t,
                                    KeyValueIndexOp<uint32_t, DATA_TYPE>,
                                    raft::KeyValuePair<uint32_t, DATA_TYPE>*>
        clusters(min_cluster_and_distance.data_handle(), conversion_op);

#if PRUNE_CENTROIDS

        thrust::copy(clusters, clusters+n, d_clusters.begin());


        // Begin centroid pruning 
        if (iter > 1) {


            thrust::fill(raft::resource::get_thrust_policy(raft_handle), 
                            d_stationary_clusters.begin(), d_stationary_clusters.end(), 1);

            // TODO: Replace this with thrust::transform
            raft::linalg::subtract(d_clusters_mask_ptr,
                                    d_clusters_ptr,
                                    d_clusters_prev_ptr,
                                    n,
                                    stream);

            find_stationary_clusters<<<(ceil( (float)n / 1024.0f)), 1024>>>
                                                                    (n,k,
                                                                    d_clusters_mask_ptr,
                                                                    d_clusters_ptr,
                                                                    d_clusters_prev_ptr,
                                                                    d_stationary_clusters_ptr);
#if DEBUG_PRUNING

            std::vector<uint32_t> h_clusters(n);
            std::vector<uint32_t> h_clusters_prev(n);
            std::vector<uint32_t> h_stationary_clusters(k);
            std::vector<int32_t> h_mask(n);

            thrust::copy(d_clusters.begin(), d_clusters.end(), h_clusters.begin());
            thrust::copy(d_clusters_prev.begin(), d_clusters_prev.end(), h_clusters_prev.begin());
            thrust::copy(d_stationary_clusters.begin(), d_stationary_clusters.end(), h_stationary_clusters.begin());
            thrust::copy(d_clusters_mask.begin(), d_clusters_mask.end(), h_mask.begin());

            std::cout<<"CLUSTERS"<<std::endl;
            for (int i=0; i<n; i++) {
                std::cout<<h_clusters[i]<<":"<<h_clusters_prev[i]<<std::endl;
            }

            std::cout<<"LIST OF STATIONARY CLUSTERS"<<std::endl;
            std::for_each(h_stationary_clusters.begin(), h_stationary_clusters.end(), [](auto& elem)
                    {std::cout<<elem<<std::endl;});

            //std::cout<<"MASK"<<std::endl;
            //std::for_each(h_mask.begin(), h_mask.end(), [](auto& elem)
             //       {std::cout<<elem<<std::endl;});


#endif

            void * d_tmp_storage = nullptr;
            size_t tmp_bytes = 0;
            cub::DeviceReduce::Sum(d_tmp_storage, tmp_bytes, d_stationary_clusters_ptr,
                                    d_k_pruned.data_handle(), k);

            CHECK_CUDA_ERROR(cudaMalloc(&d_tmp_storage, tmp_bytes));

            cub::DeviceReduce::Sum(d_tmp_storage, tmp_bytes, d_stationary_clusters_ptr,
                                    d_k_pruned.data_handle(), k);

            raft::copy(&k_pruned, d_k_pruned.data_handle(), 1, stream);

#if COUNT_STATIONARY_CLUSTERS
            std::cout<<"NUMBER OF STATIONARY CENTROIDS: "<<k_pruned<<std::endl;
#endif
            k_pruned = k - k_pruned;

            CHECK_CUDA_ERROR(cudaFree(d_tmp_storage));

            /* d_stationary_clusters[i]==1 if cluster i is stationary */
            std::vector<uint32_t> h_offsets;
            h_offsets.reserve(k);

            std::vector<uint32_t> h_stationary_clusters_vec(k);
            CHECK_CUDA_ERROR(cudaMemcpy(h_stationary_clusters_vec.data(),
                                        d_stationary_clusters_ptr,
                                        sizeof(uint32_t)*k,
                                        cudaMemcpyDeviceToHost));

            uint32_t curr_offset = 0;
            for (int i=0; i<k; i++) {
                if (h_stationary_clusters_vec[i] != 0) {
                    /* Cluster i is stationary */
                    curr_offset++;
                } else {
                    h_offsets.push_back(curr_offset);
                }
            }

#if DEBUG_PRUNING
            std::cout<<"OFFSETS"<<std::endl;
            std::for_each(h_offsets.begin(), h_offsets.end(),
                        [](auto& elem) {std::cout<<elem<<std::endl;});
#endif

            d_offsets = h_offsets; //Copy to device

            /* d_offsets will be added to the clusters array after it's computed with argmin during the 
             * next iteration
             */

        }

        raft::copy(min_cluster_and_distance_prev.data_handle(),
                   min_cluster_and_distance.data_handle(),
                   n, stream);

        thrust::copy(d_clusters.begin(), d_clusters.end(), d_clusters_prev.begin());

        cudaEventRecord(e_perf_prune_stop);
        cudaEventSynchronize(e_perf_prune_stop);;
        float e_perf_prune_ms = 0;
        cudaEventElapsedTime(&e_perf_prune_ms, e_perf_prune_start, e_perf_prune_stop);

        printf(CYAN "[PERFORMANCE]" RESET " prune_centroids time: %.8f\n", e_perf_prune_ms / 1000);

        cudaEventDestroy(e_perf_prune_start);
        cudaEventDestroy(e_perf_prune_stop);

#endif


        //reduce_cols_by_key to compute cluster d_clusters_len 

        raft::linalg::reduce_cols_by_key(one_vec.data_handle(),
                                            clusters,
                                            d_clusters_len,
                                            (uint32_t)1,
                                            (uint32_t)n,
                                            k,
                                            stream);
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
#if LOG


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
#endif

#if DEBUG_KERNEL_ARGMIN

        printf(GREEN "[DEBUG_KERNEL_ARGMIN]\n" RESET);

        thrust::device_vector<uint32_t> d_points_clusters_vec(n);

        thrust::copy(clusters, clusters+n, std::begin(d_points_clusters_vec));

        std::vector<uint32_t> tmp1(n);
        thrust::copy(d_points_clusters_vec.begin(), 
                d_points_clusters_vec.end(), 
                    std::begin(tmp1));

        printf(GREEN "p  -> c\n");
        for (uint32_t i = 0; i < n; ++i)
                printf("%-2u -> %-2u\n", i, tmp1[i]);
        cout << RESET << endl;

#endif

		///////////////////////////////////////////* COMPUTE NEW CENTROIDS *///////////////////////////////////////////

		//CHECK_CUDA_ERROR(cudaMemset(d_new_centroids, 0, k * d * sizeof(DATA_TYPE)));

#if PERFORMANCES_KERNEL_CENTROIDS

        cudaEvent_t e_perf_cent_start, e_perf_cent_stop;

        cudaEventCreate(&e_perf_cent_start);
        cudaEventCreate(&e_perf_cent_stop);
        cudaEventRecord(e_perf_cent_start);

#endif
        const uint32_t v_mat_block_dim = min(n, (size_t)deviceProps->maxThreadsPerBlock);
        const uint32_t v_mat_grid_dim = ceil((float)n / (float)v_mat_block_dim);

        compute_v_sparse<<<v_mat_grid_dim, v_mat_block_dim>>>(d_V_vals, 
                                                              d_V_rowinds, 
                                                              d_V_col_offsets, 
                                                              clusters, d_clusters_len,
                                                              n);
        if (dist_method==Kmeans::DistanceMethod::gemm) {


            compute_centroids_spmm(cusparseHandle,
                                    d, n, k,
                                    d_new_centroids,
                                    V_descr,
                                    P_descr,
                                    C_descr);

        }
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

#if PERFORMANCES_KERNEL_CENTROIDS

        cudaEventRecord(e_perf_cent_stop);
        cudaEventSynchronize(e_perf_cent_stop);

        float e_perf_cent_ms = 0;
        cudaEventElapsedTime(&e_perf_cent_ms, e_perf_cent_start, e_perf_cent_stop);

        printf(CYAN "[PERFORMANCE]" RESET " compute_centroids time: %.8f\n", e_perf_cent_ms / 1000);

        cudaEventDestroy(e_perf_cent_start);
        cudaEventDestroy(e_perf_cent_stop);
#endif

#if DEBUG_KERNEL_CENTROIDS

        CHECK_CUDA_ERROR(cudaMemset(h_centroids, 0, k * d * sizeof(DATA_TYPE)));

        thrust::copy(clusters, clusters + n, d_clusters.begin());
        thrust::copy(d_clusters.begin(), d_clusters.end(), h_points_clusters.begin());

        uint32_t* h_clusters_len;
        CHECK_CUDA_ERROR(cudaMallocHost(&h_clusters_len, k * sizeof(uint32_t)));
        CHECK_CUDA_ERROR(cudaMemcpy(h_clusters_len, 
                                    d_clusters_len,	
                                    k * sizeof(uint32_t), 
                                    cudaMemcpyDeviceToHost));

        for (uint32_t i = 0; i < n; ++i) {
            for (uint32_t j = 0; j < d; ++j) {
                h_centroids[h_points_clusters[i] * d + j] += h_points[i * d + j];
            }
        }

        for (uint32_t i = 0; i < k; ++i) {
            for (uint32_t j = 0; j < d; ++j) {
                uint64_t count = h_clusters_len[i] > 1 ? h_clusters_len[i] : 1;
                DATA_TYPE scale = 1.0 / ((double) count);
                h_centroids[i * d + j] *= scale;
            
            }
        }

        cout << GREEN "[DEBUG_KERNEL_CENTROIDS]" << endl;
        cout << endl << "CENTROIDS (CPU)" << endl;
        for (uint32_t i = 0; i < k; ++i) {
            for (uint32_t j = 0; j < d; ++j)
                printf("%.3f, ", h_centroids[i * d + j]);
            cout << endl;
        }

        CHECK_CUDA_ERROR(cudaMemset(h_centroids, 0, d * k * sizeof(DATA_TYPE)));
        CHECK_CUDA_ERROR(cudaMemcpy(h_centroids, d_new_centroids, 
                                    d * k * sizeof(DATA_TYPE), 
                                    cudaMemcpyDeviceToHost));

        cout << endl << "CENTROIDS (GPU)" << endl;
        for (uint32_t i = 0; i < k; ++i) {
            for (uint32_t j = 0; j < d; ++j)
#if COMPUTE_CENTROIDS_KERNEL>=1
                printf("%.3f, ", h_centroids[i + j*k]);
#else
                printf("%.3f, ", h_centroids[i*d + j]);
#endif
            cout << endl;
        }

        cout << RESET << endl;
        CHECK_CUDA_ERROR(cudaFreeHost(h_clusters_len));

#endif

		/////////////////////////////////////////////* CHECK IF CONVERGED */////////////////////////////////////////////

		// Check exit
        /*
        auto sqrd_norm = raft::make_device_scalar(raft_handle, DATA_TYPE(0));
        raft::linalg::mapThenSumReduce(sqrd_norm.data_handle(),
                                         d*k,
                                         raft::sqdiff_op{},
                                         stream,
                                         d_centroids,
                                         d_new_centroids);

        DATA_TYPE sqrd_norm_err = 0;
        raft::copy(&sqrd_norm_err, sqrd_norm.data_handle(), sqrd_norm.size(), stream);
        */

#if PRUNE_CENTROIDS
        if (iter > 1) {
            const uint32_t prune_threads = min((uint32_t)deviceProps->maxThreadsPerBlock,
                                               (uint32_t)(k_pruned * d));
            const uint32_t prune_blocks = ceil((float)(k_pruned * d) / prune_threads);
            prune_centroids<<<prune_blocks, prune_threads>>>(d_new_centroids,
                                                             d_centroids,
                                                             d_stationary_clusters_ptr,
                                                             d_offsets_ptr,
                                                             d, k, k_pruned);
        } else {
            CHECK_CUDA_ERROR(cudaMemcpy(d_centroids, d_new_centroids, CENTROIDS_BYTES, cudaMemcpyDeviceToDevice));
        }
#else
        //CHECK_CUDA_ERROR(cudaMemcpy(d_centroids, d_new_centroids, CENTROIDS_BYTES, cudaMemcpyDeviceToDevice));
#endif

        rmm::device_scalar<DATA_TYPE> d_score(stream);
        raft::cluster::detail::computeClusterCost(
                                                raft_handle, 
                                                min_cluster_and_distance.view(),
                                                workspace,
                                                raft::make_device_scalar_view(d_score.data()),
                                                raft::value_op{},
                                                raft::add_op{});
        score = d_score.value(stream);

        if (iter==maxiter) {
            thrust::copy(clusters, clusters+n, d_clusters.begin());
            thrust::copy(d_clusters.begin(), d_clusters.end(), h_points_clusters.begin());
            break;
        }

        if (check_converged && 
            (iter > 1) && 
            (std::abs(score-last_score) < tol)) {
            converged = iter;
            thrust::copy(clusters, clusters+n, d_clusters.begin());
            thrust::copy(d_clusters.begin(), d_clusters.end(), h_points_clusters.begin());
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
	/* MAIN LOOP END */
    /*
    CHECK_CUDA_ERROR(cudaMemcpy(h_centroids, 
                                d_centroids, 
                                d * k * sizeof(DATA_TYPE), 
                                cudaMemcpyDeviceToHost));
                                */
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());


#if DEBUG_INIT_CENTROIDS
    cout << endl << "Centroids" << endl;
    printMatrixRowMaj(h_centroids, k, d);
#endif

#if PROFILE_MEMORY
    size_t total_mem, free_mem;
    CHECK_CUDA_ERROR(cudaMemGetInfo(&free_mem, &total_mem));
    size_t usage = (total_mem - free_mem)/1e6;
    cout<<"MEMORY FOOTPRINT: "<<usage<<" MB"<<endl;
#endif

	/* COPY BACK RESULTS*/

	for (size_t i = 0; i < n; i++) {
		points[i]->setCluster(h_points_clusters[i]);
	}

	/* FREE MEMORY */
	CHECK_CUDA_ERROR(cudaFree(d_distances));
	CHECK_CUDA_ERROR(cudaFree(d_clusters_len));

    CHECK_CUDA_ERROR(cudaFree(d_centroids_row_norms));
    CHECK_CUDA_ERROR(cudaFree(d_points_row_norms));


	return converged;
}






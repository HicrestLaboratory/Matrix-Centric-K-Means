#include <stdio.h>
#include <vector>
#include <iomanip>
#include <cmath>
#include <ctime>
#include <limits>
#include <map>
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

#include <raft/core/device_mdarray.hpp>
#include <raft/cluster/kmeans.cuh>
#include <raft/cluster/kmeans_types.hpp>
#include <raft/core/resources.hpp>
#include <raft/linalg/map.cuh>
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
                const InitMethod _initMethod)
		: n(_n), d(_d), k(_k), tol(_tol),
		POINTS_BYTES(_n * _d * sizeof(DATA_TYPE)),
		CENTROIDS_BYTES(_k * _d * sizeof(DATA_TYPE)),
        h_points_clusters(_n),
		points(_points),
		deviceProps(_deviceProps),
        initMethod(_initMethod)
{

    CHECK_CUBLAS_ERROR(cublasCreate(&cublasHandle));

    random_device rd;
    generator = new mt19937(std::time(0));


	CHECK_CUDA_ERROR(cudaHostAlloc(&h_points, POINTS_BYTES, cudaHostAllocDefault));
	for (size_t i = 0; i < n; ++i) {
		for (size_t j = 0; j < d; ++j) {
			h_points[i * d + j] = _points[i]->get(j);
		}
	}


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

    /* Init B */
    CHECK_CUDA_ERROR(cudaMalloc(&d_B, sizeof(DATA_TYPE)*n*n)); //TODO: Make this symmetric

#if PERFORMANCES_BMULT
    cudaEvent_t e_perf_bmult_start, e_perf_bmult_stop;

    cudaEventCreate(&e_perf_bmult_start);
    cudaEventCreate(&e_perf_bmult_stop);
    cudaEventRecord(e_perf_bmult_start);
#endif
    //TODO: move this to main loop
    float b_alpha = -2.0;
    float b_beta = 0.0;
    CHECK_CUBLAS_ERROR(cublasSgemm(cublasHandle, 
                                    CUBLAS_OP_T,
                                    CUBLAS_OP_N,
                                    n, n, d,
                                    &b_alpha,
                                    d_points, d,
                                    d_points, d,
                                    &b_beta,
                                    d_B, n));
#if PERFORMANCES_BMULT

    cudaEventRecord(e_perf_bmult_stop);
    cudaEventSynchronize(e_perf_bmult_stop);

    float e_perf_bmult_ms = 0;
    cudaEventElapsedTime(&e_perf_bmult_ms, e_perf_bmult_start, e_perf_bmult_stop);
    printf(CYAN "[PERFORMANCE]" RESET " b-mult time: %.8f\n", e_perf_bmult_ms / 1000);

    cudaEventDestroy(e_perf_bmult_start);
    cudaEventDestroy(e_perf_bmult_stop);
#endif

#if PERFORMANCES_CENTROIDS_INIT 

    cudaEvent_t e_centroid_init_start, e_centroid_init_stop;

    cudaEventCreate(&e_centroid_init_start);
    cudaEventCreate(&e_centroid_init_stop);
    cudaEventRecord(e_centroid_init_start);

#endif

	CHECK_CUDA_ERROR(cudaMalloc(&d_centroids, CENTROIDS_BYTES));

    h_centroids_matrix = NULL;
    CHECK_CUDA_ERROR(cudaHostAlloc(&h_centroids, CENTROIDS_BYTES, cudaHostAllocDefault));
    CHECK_CUDA_ERROR(cudaMalloc(&d_new_centroids, CENTROIDS_BYTES));
    raft::resources handle;
    auto points_view = raft::make_device_matrix_view<DATA_TYPE, uint32_t>(d_points, n, d);
    auto centroids_view = raft::make_device_matrix_view<DATA_TYPE, uint32_t>(d_centroids, k, d);
    raft::cluster::detail::shuffleAndGather<DATA_TYPE, uint32_t>(handle, points_view, centroids_view, k, 11);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    CHECK_CUDA_ERROR(cudaMemcpy(h_centroids, d_centroids, d * k * sizeof(DATA_TYPE), cudaMemcpyDeviceToHost));


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

Kmeans::~Kmeans () {
	delete generator;
	CHECK_CUDA_ERROR(cudaFreeHost(h_points));
	CHECK_CUDA_ERROR(cudaFreeHost(h_centroids));
	CHECK_CUDA_ERROR(cudaFree(d_centroids));
    CHECK_CUDA_ERROR(cudaFree(d_new_centroids));
	CHECK_CUDA_ERROR(cudaFree(d_points));
    CHECK_CUDA_ERROR(cudaFree(d_B));
	if (h_centroids_matrix != NULL) {
		CHECK_CUDA_ERROR(cudaFreeHost(h_centroids_matrix));
	}
	compute_gemm_distances_free();
}

void Kmeans::init_centroids_rand (Point<DATA_TYPE>** points) {
	uniform_int_distribution<int> random_int(0, n - 1);

    h_centroids_matrix = NULL;

	CHECK_CUDA_ERROR(cudaHostAlloc(&h_centroids, CENTROIDS_BYTES, cudaHostAllocDefault));
    CHECK_CUDA_ERROR(cudaMalloc(&d_new_centroids, CENTROIDS_BYTES));


	unsigned int i = 0;
    map<int, bool> usedPoints;
	Point<DATA_TYPE>* centroids[k];
	while (i < k) {
        int point_idx = random_int(*generator);
		Point<DATA_TYPE>* p = points[point_idx];
		bool found = usedPoints.find(point_idx)!=usedPoints.end();
		if (!found) {
			centroids[i] = new Point<DATA_TYPE>(p);
			usedPoints.emplace(point_idx, true);
			++i;
		}
	}

#if DEBUG_INIT_CENTROIDS
    cout << endl << "Centroids" << endl;
    for (i = 0; i < k; ++i)
        cout << *(centroids[i]) << endl;
#endif

	for (size_t i = 0; i < k; ++i) {
		for (size_t j = 0; j < d; ++j) {
			h_centroids[i * d + j] = centroids[i]->get(j); // Row major
		}
	}

}


void Kmeans::init_centroids_plusplus(DATA_TYPE * d_points)
{
    h_centroids_matrix = NULL;

    uint32_t n_clusters = 0;
    
    /* Allocate host centroid buffers */
	CHECK_CUDA_ERROR(cudaHostAlloc(&h_centroids, CENTROIDS_BYTES, cudaHostAllocDefault));


    /* Choose first centroid uniformly at random */
    uniform_int_distribution<int> rand_uniform(0, n-1);
    int first_centroid_idx = rand_uniform(*generator);
    DATA_TYPE * d_first_centroid = d_points + (d * first_centroid_idx);

    /* Copy it to centroids matrix */
    CHECK_CUDA_ERROR(cudaMemcpy(d_centroids, d_first_centroid, 
                                sizeof(DATA_TYPE)*d, cudaMemcpyDeviceToDevice));
    n_clusters++;


    /* Malloc points matrix and centroid vector */
    DATA_TYPE * d_P;
    DATA_TYPE * d_c_vec;
    CHECK_CUDA_ERROR(cudaMalloc(&(d_P), n * 3 * d * sizeof(DATA_TYPE)));
    CHECK_CUDA_ERROR(cudaMalloc(&(d_c_vec), 3 * d * sizeof(DATA_TYPE)));

    /* Malloc tmp_distances and distances, init distances */
    DATA_TYPE * tmp_dist;
    DATA_TYPE * dist;
    CHECK_CUDA_ERROR(cudaMalloc(&tmp_dist, sizeof(DATA_TYPE)*n));
    CHECK_CUDA_ERROR(cudaMalloc(&dist, sizeof(DATA_TYPE)*n));


    /* Initialize points matrix */
    uint32_t p_mat_block_dim(min((size_t)deviceProps->maxThreadsPerBlock, n));
    uint32_t p_mat_grid_dim(3*d);
    uint32_t p_rounds = ceil((float)n / (float)p_mat_block_dim);

    compute_p_matrix<<<p_mat_grid_dim, p_mat_block_dim>>>(d_points, d_P, d, n, k, p_rounds);


    /* Initialize first c_vec using first chosen centroid */
    uint32_t c_vec_block_dim = min((uint32_t)deviceProps->maxThreadsPerBlock, d);
    uint32_t c_vec_grid_dim = ceil((float)d / (float)c_vec_block_dim) ;  

    compute_c_vec<<<c_vec_grid_dim, c_vec_block_dim>>>(d_first_centroid, d_c_vec, d);

    /* Call gemv to compute distances between all points and most recent centroid */
    const DATA_TYPE alpha = 1.0;
    const DATA_TYPE beta = 1.0;
    CHECK_CUBLAS_ERROR(cublasSgemv(cublasHandle, CUBLAS_OP_N, 
                                   n, 3*d,
                                   &alpha,
                                   d_P, n,
                                   d_c_vec, 1,
                                   &beta,
                                   dist, 1));

    /* Set up argmax */
    //cub::KeyValuePair<int32_t, DATA_TYPE> * d_argmax = nullptr;
    uint32_t argmax_idx;
    //CHECK_CUDA_ERROR(cudaMalloc(&d_argmax, sizeof(uint32_t)*2));

    void * d_tmp_storage = nullptr;
    size_t tmp_storage_bytes = 0;
    //cub::DeviceReduce::ArgMax(d_tmp_storage, tmp_storage_bytes,
     //                           dist, d_argmax, n);
    CHECK_CUDA_ERROR(cudaMalloc(&d_tmp_storage, tmp_storage_bytes));

                                

    while (n_clusters < k) {

        /* Argmax to get next centroid */
        //cub::DeviceReduce::ArgMax(d_tmp_storage, tmp_storage_bytes,
         //                       dist, d_argmax, n);
        //CHECK_CUDA_ERROR(cudaMemcpy(&argmax_idx, &(d_argmax->key), sizeof(uint32_t),
         //                           cudaMemcpyDeviceToHost));

        DATA_TYPE * d_farthest_point = d_points + (d * argmax_idx);
        DATA_TYPE * d_curr_centroid = d_centroids + (d * n_clusters);

        CHECK_CUDA_ERROR(cudaMemcpy(d_curr_centroid, 
                                    d_farthest_point,
                                    sizeof(DATA_TYPE)*d,
                                    cudaMemcpyDeviceToDevice));

        /* Compute distances between points and previously chosen centroid */
        compute_c_vec<<<c_vec_grid_dim, c_vec_block_dim>>>(d_curr_centroid, d_c_vec, d);
        CHECK_CUBLAS_ERROR(cublasSgemv(cublasHandle, CUBLAS_OP_N, 
                                       n, 3*d,
                                       &alpha,
                                       d_P, n,
                                       d_c_vec, 1,
                                       &beta,
                                       tmp_dist, 1));

        /* Update dist vector with new smallest distances */
        const uint32_t ewise_block_dim = min(d, deviceProps->maxThreadsPerBlock);
        const uint32_t ewise_grid_dim = ceil((float)d / (float)ewise_block_dim);
        ewise_min<<<ewise_grid_dim, ewise_block_dim>>>(tmp_dist, dist, d);

        n_clusters++;
    }

    // Copy centroids to host
    CHECK_CUDA_ERROR(cudaMemcpy(h_centroids, d_centroids, sizeof(DATA_TYPE)*d*k,
                                cudaMemcpyDeviceToHost));

    // Free  
    CHECK_CUDA_ERROR(cudaFree(d_P));
    CHECK_CUDA_ERROR(cudaFree(d_c_vec));
    CHECK_CUDA_ERROR(cudaFree(tmp_dist));
    CHECK_CUDA_ERROR(cudaFree(dist));
    CHECK_CUDA_ERROR(cudaFree(d_tmp_storage));

}

uint64_t Kmeans::run (uint64_t maxiter, bool check_converged) {
    uint64_t converged = maxiter;

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


    uint32_t* d_clusters_len;
    CHECK_CUDA_ERROR(cudaMalloc(&d_clusters_len, k * sizeof(uint32_t)));


    uint64_t iter = 0;

    cusparseHandle_t cusparseHandle;
    CHECK_CUSPARSE_ERROR(cusparseCreate(&cusparseHandle));

    DATA_TYPE * d_points_row_norms;
    DATA_TYPE * d_centroids_row_norms;

    CHECK_CUDA_ERROR(cudaMalloc(&d_points_row_norms, sizeof(DATA_TYPE)*n));
    CHECK_CUDA_ERROR(cudaMalloc(&d_centroids_row_norms, sizeof(DATA_TYPE) * k));

    raft::linalg::rowNorm(d_points_row_norms, d_points, d, (uint32_t)n, raft::linalg::L2Norm, true, stream);


    /* Malloc V for later */
    const uint32_t v_rows = k;
    const uint32_t v_cols = n;
    const uint32_t v_size = v_rows*v_cols;

    DATA_TYPE * d_V_vals;
    int32_t * d_V_rowinds;
    int32_t * d_V_col_offsets;

    CHECK_CUDA_ERROR(cudaMalloc(&d_V_vals, sizeof(DATA_TYPE)*n));
    CHECK_CUDA_ERROR(cudaMalloc(&d_V_rowinds, sizeof(int32_t)*n));
    CHECK_CUDA_ERROR(cudaMalloc(&d_V_col_offsets, sizeof(int32_t)*(n+1)));
	
    cusparseSpMatDescr_t V_descr;
    cusparseDnMatDescr_t P_descr;
    cusparseDnMatDescr_t B_descr;
    cusparseDnMatDescr_t D_descr;
    cusparseDnMatDescr_t C_descr;

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

    CHECK_CUSPARSE_ERROR(cusparseCreateDnMat(&B_descr,
                                            n, n, n,
                                            d_B,
                                            CUDA_R_32F,
                                            CUSPARSE_ORDER_ROW));

    CHECK_CUSPARSE_ERROR(cusparseCreateDnMat(&D_descr,
                                             k, n, k,
                                             d_distances,
                                             CUDA_R_32F,
                                             CUSPARSE_ORDER_COL));

    CHECK_CUSPARSE_ERROR(cusparseCreateDnMat(&C_descr,
                                              k, d, d,
                                              d_new_centroids,
                                              CUDA_R_32F,
                                              CUSPARSE_ORDER_ROW));


    /* MAIN LOOP */
    while (iter++ < maxiter) {
    /* COMPUTE DISTANCES */

#if PERFORMANCES_KERNEL_DISTANCES

        cudaEvent_t e_perf_dist_start, e_perf_dist_stop;

        cudaEventCreate(&e_perf_dist_start);
        cudaEventCreate(&e_perf_dist_stop);
        cudaEventRecord(e_perf_dist_start);

#endif

        raft::linalg::rowNorm(d_centroids_row_norms, d_centroids, 
                                d, k_pruned, raft::linalg::L2Norm, true, 
                                stream);

        if (iter==1) {
            compute_gemm_distances_arizona(cublasHandle, 
                                          d, n, k_pruned,
                                          d_points, d_points_row_norms,
                                          d_centroids, d_centroids_row_norms,
                                          d_distances);
        } else {
            /* Bellavita K-Means */
            compute_gemm_distances_bellavita(cusparseHandle, 
                                            d, n, k_pruned,
                                            d_points_row_norms,
                                            d_centroids_row_norms,
                                            B_descr, V_descr,
                                            D_descr, d_distances);
        }

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

		CHECK_CUDA_ERROR(cudaMemset(d_new_centroids, 0, k * d * sizeof(DATA_TYPE)));

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

        compute_centroids_spmm(cusparseHandle,
                                d, n, k,
                                d_new_centroids,
								V_descr,
								P_descr,
                                C_descr);

        

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
        auto sqrd_norm = raft::make_device_scalar(raft_handle, DATA_TYPE(0));
        raft::linalg::mapThenSumReduce(sqrd_norm.data_handle(),
                                         d*k,
                                         raft::sqdiff_op{},
                                         stream,
                                         d_centroids,
                                         d_new_centroids);

        DATA_TYPE sqrd_norm_err = 0;
        raft::copy(&sqrd_norm_err, sqrd_norm.data_handle(), sqrd_norm.size(), stream);

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
        CHECK_CUDA_ERROR(cudaMemcpy(d_centroids, d_new_centroids, CENTROIDS_BYTES, cudaMemcpyDeviceToDevice));
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

        if (check_converged && (iter > 1) && (sqrd_norm_err < tol)) {
            converged = iter;
            thrust::copy(clusters, clusters+n, d_clusters.begin());
            thrust::copy(d_clusters.begin(), d_clusters.end(), h_points_clusters.begin());
            break;
        } 



	}
	/* MAIN LOOP END */
    CHECK_CUDA_ERROR(cudaMemcpy(h_centroids, 
                                d_centroids, 
                                d * k * sizeof(DATA_TYPE), 
                                cudaMemcpyDeviceToHost));
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

    CHECK_CUDA_ERROR(cudaFree(d_V_vals));
    CHECK_CUDA_ERROR(cudaFree(d_V_rowinds));
    CHECK_CUDA_ERROR(cudaFree(d_V_col_offsets));

    CHECK_CUSPARSE_ERROR(cusparseDestroyDnMat(P_descr)); 
    CHECK_CUSPARSE_ERROR(cusparseDestroyDnMat(C_descr)); 
    CHECK_CUSPARSE_ERROR(cusparseDestroyDnMat(B_descr)); 
    CHECK_CUSPARSE_ERROR(cusparseDestroyDnMat(D_descr)); 
    CHECK_CUSPARSE_ERROR(cusparseDestroySpMat(V_descr)); 

    CHECK_CUSPARSE_ERROR(cusparseDestroy(cusparseHandle));

    CHECK_CUBLAS_ERROR(cublasDestroy(cublasHandle));

	return converged;
}






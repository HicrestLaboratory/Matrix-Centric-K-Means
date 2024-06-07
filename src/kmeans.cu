#include <stdio.h>
#include <vector>
#include <iomanip>
#include <cmath>
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

#include <raft/core/device_mdarray.hpp>
#include <raft/cluster/kmeans.cuh>
#include <raft/cluster/kmeans_types.hpp>
#include <raft/core/resources.hpp>
#include <raft/linalg/map.cuh>




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

	if (seed) {
		seed_seq s{*seed};
		generator = new mt19937(s);
	}
	else {
		random_device rd;
		generator = new mt19937(rd());
	}


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
}

Kmeans::~Kmeans () {
	delete generator;
	CHECK_CUDA_ERROR(cudaFreeHost(h_points));
	CHECK_CUDA_ERROR(cudaFreeHost(h_centroids));
	CHECK_CUDA_ERROR(cudaFree(d_centroids));
    CHECK_CUDA_ERROR(cudaFree(d_new_centroids));
	CHECK_CUDA_ERROR(cudaFree(d_points));
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
    //CHECK_CUDA_ERROR(cudaFree(d_argmax));
    

}

uint64_t Kmeans::run (uint64_t maxiter) {
    uint64_t converged = maxiter;

    CHECK_CUBLAS_ERROR(cublasCreate(&cublasHandle));


#if PERFORMANCES_CENTROIDS_INIT 

    cudaEvent_t e_centroid_init_start, e_centroid_init_stop;

    cudaEventCreate(&e_centroid_init_start);
    cudaEventCreate(&e_centroid_init_stop);
    cudaEventRecord(e_centroid_init_start);

#endif

	CHECK_CUDA_ERROR(cudaMalloc(&d_centroids, CENTROIDS_BYTES));

    switch(initMethod)
    {
        case InitMethod::random:
            init_centroids_rand(points);
            CHECK_CUDA_ERROR(cudaMemcpy(d_centroids, h_centroids, d * k * sizeof(DATA_TYPE), cudaMemcpyHostToDevice));
            break;
        case InitMethod::kmeans_plus_plus:
            init_centroids_plusplus(d_points);
            CHECK_CUDA_ERROR(cudaMemcpy(h_centroids, d_centroids, d * k * sizeof(DATA_TYPE), cudaMemcpyHostToDevice));
            break;
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


    const raft::resources raft_handle;
    const cudaStream_t stream = raft::resource::get_cuda_stream(raft_handle);

    DATA_TYPE* d_distances;
    CHECK_CUDA_ERROR(cudaMalloc(&d_distances, n * k * sizeof(DATA_TYPE)));

    uint32_t* d_points_clusters;
    CHECK_CUDA_ERROR(cudaMalloc(&d_points_clusters, sizeof(uint32_t)*n));

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
    DATA_TYPE * d_V;
    CHECK_CUDA_ERROR(cudaMalloc(&d_V, v_size*sizeof(DATA_TYPE)));


    DATA_TYPE * d_V_vals;
    int32_t * d_V_rowinds;
    int32_t * d_V_col_offsets;

    CHECK_CUDA_ERROR(cudaMalloc(&d_V_vals, sizeof(DATA_TYPE)*n));
    CHECK_CUDA_ERROR(cudaMalloc(&d_V_rowinds, sizeof(int32_t)*n));
    CHECK_CUDA_ERROR(cudaMalloc(&d_V_col_offsets, sizeof(int32_t)*(n+1)));
	
    cusparseSpMatDescr_t V_descr;
    cusparseDnMatDescr_t P_descr;
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


        raft::linalg::rowNorm(d_centroids_row_norms, d_centroids, d, k, raft::linalg::L2Norm, true, 
                                stream);

        compute_gemm_distances_arizona(cublasHandle, 
                                      d, n, k,
                                      d_points, d_points_row_norms,
                                      d_centroids, d_centroids_row_norms,
                                      d_distances);


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


		CHECK_CUDA_ERROR(cudaMemset(d_clusters_len, 0, k * sizeof(uint32_t)));


        auto min_cluster_and_distance = raft::make_device_vector<raft::KeyValuePair<uint32_t, DATA_TYPE>, uint32_t>(raft_handle, n);
        
        raft::KeyValuePair<uint32_t, DATA_TYPE> initial_value(0, std::numeric_limits<DATA_TYPE>::max());

        thrust::fill(raft::resource::get_thrust_policy(raft_handle),
                   min_cluster_and_distance.data_handle(),
                   min_cluster_and_distance.data_handle() + min_cluster_and_distance.size(),
                   initial_value);

        auto pw_dist_const = raft::make_device_matrix_view<const DATA_TYPE, uint32_t>(d_distances, k, n);

        raft::linalg::coalescedReduction(
                min_cluster_and_distance.data_handle(),
                pw_dist_const.data_handle(), (uint32_t)k, (uint32_t)n,
                initial_value,
                stream,
                true,
                [=] __device__(const DATA_TYPE val, const uint32_t i) {
                    raft::KeyValuePair<uint32_t, DATA_TYPE> pair;
                    pair.key   = i;
                    pair.value = val;
                    return pair;
                },
                raft::argmin_op{},
                raft::identity_op{});

        //extract cluster labels from kvpair into d_points_clusters
        raft::cluster::kmeans::KeyValueIndexOp<uint32_t, DATA_TYPE> conversion_op;
        cub::TransformInputIterator<uint32_t,
                                    raft::cluster::KeyValueIndexOp<uint32_t, DATA_TYPE>,
                                    raft::KeyValuePair<uint32_t, DATA_TYPE>*>
        clusters(min_cluster_and_distance.data_handle(), conversion_op);

        cub::TransformInputIterator<DATA_TYPE,
                                    raft::cluster::KeyValueIndexOp<uint32_t, DATA_TYPE>,
                                    raft::KeyValuePair<uint32_t, DATA_TYPE>*>
        min_distances(min_cluster_and_distance.data_handle(), conversion_op);


        //TODO: prune centroids that don't move

        //reduce_cols_by_key to compute cluster d_clusters_len 
        auto one_vec = raft::make_device_vector<uint32_t>(raft_handle, n);
        thrust::fill(raft::resource::get_thrust_policy(raft_handle),
                        one_vec.data_handle(),
                        one_vec.data_handle() + n,
                        1);
        raft::linalg::reduce_cols_by_key(one_vec.data_handle(),
                                            clusters,
                                            d_clusters_len,
                                            (uint32_t)1,
                                            (uint32_t)n,
                                            k,
                                            stream);

#if PERFORMANCES_KERNEL_ARGMIN

        cudaEventRecord(e_perf_argmin_stop);
        cudaEventSynchronize(e_perf_argmin_stop);

        float e_perf_argmin_ms = 0;
        cudaEventElapsedTime(&e_perf_argmin_ms, e_perf_argmin_start, e_perf_argmin_stop);

        printf(CYAN "[PERFORMANCE]" RESET " clusters_argmin_shfl time: %.8f\n", e_perf_argmin_ms / 1000);

        cudaEventDestroy(e_perf_argmin_stop);
        cudaEventDestroy(e_perf_argmin_start);

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

		CHECK_CUDA_ERROR(cudaMemset(d_new_centroids, 0, k * d * sizeof(DATA_TYPE)));

#if PERFORMANCES_KERNEL_CENTROIDS

        cudaEvent_t e_perf_cent_start, e_perf_cent_stop;

        cudaEventCreate(&e_perf_cent_start);
        cudaEventCreate(&e_perf_cent_stop);
        cudaEventRecord(e_perf_cent_start);

#endif

        const uint32_t v_mat_block_dim = min(n, (size_t)deviceProps->maxThreadsPerBlock);
        const uint32_t v_mat_grid_dim = ceil((float)n / (float)v_mat_block_dim);

        compute_v_sparse<<<v_mat_grid_dim, v_mat_block_dim>>>(d_V_vals, d_V_rowinds, d_V_col_offsets, 
                                                              clusters, d_clusters_len,
                                                              n);

        compute_centroids_spmm(cusparseHandle,
                                d, n, k,
                                d_V_vals, 
                                d_V_rowinds,
                                d_V_col_offsets,
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

        thrust::device_vector<uint32_t> d_clusters_vec(n);
        thrust::copy(clusters, clusters + n, d_clusters_vec.begin());
        thrust::copy(d_clusters_vec.begin(), d_clusters_vec.end(), h_points_clusters.begin());

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


        CHECK_CUDA_ERROR(cudaMemcpy(d_centroids, d_new_centroids, CENTROIDS_BYTES, cudaMemcpyDeviceToDevice));
        rmm::device_scalar<DATA_TYPE> d_score(stream);
        rmm::device_uvector<char> workspace(0, stream);
        raft::cluster::detail::computeClusterCost(
                                                raft_handle, 
                                                min_cluster_and_distance.view(),
                                                workspace,
                                                raft::make_device_scalar_view(d_score.data()),
                                                raft::value_op{},
                                                raft::add_op{});
        score = d_score.value(stream);

        if (iter > 1 && (sqrd_norm_err < tol)) {
            converged = iter;
            thrust::device_vector<uint32_t> d_points_clusters_vec(n);
            thrust::copy(clusters, clusters + n, d_points_clusters_vec.begin());
            thrust::copy(d_points_clusters_vec.begin(), d_points_clusters_vec.end(), h_points_clusters.begin());
            break;
        } else if (iter==maxiter) {
            thrust::device_vector<uint32_t> d_points_clusters_vec(n);
            thrust::copy(clusters, clusters + n, d_points_clusters_vec.begin());
            thrust::copy(d_points_clusters_vec.begin(), d_points_clusters_vec.end(), h_points_clusters.begin());
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
	CHECK_CUDA_ERROR(cudaFree(d_points_clusters));
	CHECK_CUDA_ERROR(cudaFree(d_clusters_len));

    CHECK_CUDA_ERROR(cudaFree(d_centroids_row_norms));
    CHECK_CUDA_ERROR(cudaFree(d_points_row_norms));

    CHECK_CUDA_ERROR(cudaFree(d_V_vals));
    CHECK_CUDA_ERROR(cudaFree(d_V_rowinds));
    CHECK_CUDA_ERROR(cudaFree(d_V_col_offsets));
    CHECK_CUSPARSE_ERROR(cusparseDestroy(cusparseHandle));

    CHECK_CUBLAS_ERROR(cublasDestroy(cublasHandle));

	return converged;
}






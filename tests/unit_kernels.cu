#include <catch2/catch_test_macros.hpp>

#include <limits>
#include <iostream>
#include <math.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse.h>


#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/counting_iterator.h>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/resources.hpp>
#include <raft/linalg/map.cuh>
#include <raft/linalg/norm.cuh>
#include <raft/linalg/reduce_cols_by_key.cuh>
#include <raft/core/kvp.hpp>
#include <raft/core/resource/thrust_policy.hpp>

#include "../src/kernels/kernels.cuh"
#include "../src/cuda_utils.cuh"
#include "../src/include/common.h"

#define TEST_DEBUG 1
#define WARP_SIZE  32

template <typename IndexT, typename DataT>
struct KeyValueIndexOp {
 
  __host__ __device__ __forceinline__ IndexT
  operator()(const raft::KeyValuePair<IndexT, DataT>& a) const
  {
    return a.key ;
  }
};

const DATA_TYPE infty		= numeric_limits<DATA_TYPE>::infinity();
const DATA_TYPE EPSILON = 1e-1;

cudaDeviceProp deviceProps;

const raft::resources raft_handle;
const cudaStream_t stream = raft::resource::get_cuda_stream(raft_handle);

void initRandomMatrixColMaj (DATA_TYPE* M, uint32_t rows, uint32_t cols) {
	for (uint32_t i = 0; i < rows; ++i) {
		for (uint32_t j = 0; j < cols; ++j) {
			//std::rand() / 100000005.32;
			M[IDX2C(i, j, rows)] = ((int)trunc(std::rand() / 100000005.32)) % 6;
		}
	}
}

/**
 * @brief
 *
 * @param A a matrix of size (d+1)x(d+1) stored in column-major order
 * @param center a vector of size d
 * @param d number of dimensions
 * @param idx index of the point
 * @param ld number of rows of the matrix
 */
void computeCPUCentroidAssociatedMatrix (DATA_TYPE* A, DATA_TYPE* points, uint32_t d, uint32_t idx, uint32_t ld) {
	++d; // d = d + 1
	DATA_TYPE c;
	DATA_TYPE c_11 = 0;
	for (size_t i = 0; i < d - 1; ++i) { // Matrix borders
		c = points[IDX2C(idx, i, ld)];
		A[i + 1] = -c;
		//A[(i + 1) * d] = -c;
		c_11 += c * c;
	}
	A[0] = c_11;
	for (size_t i = 1; i < d; ++i) { // Matrix diagonal + fill with 0s
		for (size_t j = 1; j < d; ++j) {
			A[i * d + j] = i == j ? 1 : 0;
		}
	}
}

TEST_CASE("kernel_distances_matrix_arizona", "[kernel][distances]") { // FIXME does not work well with N >= 500
	const unsigned int TESTS_N = 9;
	const unsigned int N[TESTS_N] = {10, 10, 17, 30, 17,	 15,	300,	2000, 100000};
	const unsigned int D[TESTS_N] = { 1,	2,	3, 11, 42, 1500,	400,	 200, 64};
	const unsigned int K[TESTS_N] = { 2,	6,	3, 11, 20,		5,	 10,	 200, 1000};

	getDeviceProps(0, &deviceProps);

	for (int test_i = 0; test_i < TESTS_N ; ++test_i) {
		const unsigned int n = N[test_i];
		const unsigned int d = D[test_i];
		const unsigned int k = K[test_i];

		char test_name[100];
		sprintf(test_name, "kernel compute_distances_matrix_arizona n: %u	d: %u  k: %u", n, d, k);
		SECTION(test_name) {
			printf("Test: %s\n", test_name);

			DATA_TYPE *h_points = new DATA_TYPE[n * d];
			DATA_TYPE *h_points_row_maj = new DATA_TYPE[n * d];
			DATA_TYPE *h_centroids = new DATA_TYPE[k * d];
			DATA_TYPE *h_centroids_row_maj = new DATA_TYPE[k * d];
			DATA_TYPE *h_distances = new DATA_TYPE[n * k];

			// Constructing P and C
			initRandomMatrixColMaj(h_points, n, d);
			for (size_t i = 0; i < n; i++) {
				for (size_t j = 0; j < d; j++) {
					h_points_row_maj[i * d + j] = h_points[IDX2C(i, j, n)];
				}
			}

			initRandomMatrixColMaj(h_centroids, k, d);
            for (size_t i = 0; i<k; i++) {
                for (size_t j = 0; j<d; j++) {
                    h_centroids_row_maj[i*d + j] = h_centroids[IDX2C(i, j, k)];
                }
            }

			if (TEST_DEBUG) {
				printf("\nPOINTS %d:\n", n);
				printMatrixColMajLimited(h_points, n, d, 10, 10);
				printf("\nCENTERS %d:\n", k);
				printMatrixColMajLimited(h_centroids, k, d, 10, 10);
			}

			DATA_TYPE* d_points;
			cudaMalloc(&d_points, n * d * sizeof(DATA_TYPE));
			cudaMemcpy(d_points, h_points_row_maj, n * d * sizeof(DATA_TYPE), cudaMemcpyHostToDevice);

			DATA_TYPE* d_centroids;
			cudaMalloc(&d_centroids, k * d * sizeof(DATA_TYPE));
			cudaMemcpy(d_centroids, h_centroids_row_maj, k * d * sizeof(DATA_TYPE), cudaMemcpyHostToDevice);

            DATA_TYPE * d_points_row_norms;
            cudaMalloc(&d_points_row_norms, n*sizeof(DATA_TYPE));

            DATA_TYPE * d_centroids_row_norms;
            cudaMalloc(&d_centroids_row_norms, k*sizeof(DATA_TYPE));

			DATA_TYPE* d_distances;
			cudaMalloc(&d_distances, n * k * sizeof(DATA_TYPE));

			cublasHandle_t cublasHandle;
			cublasCreate(&cublasHandle);


            raft::linalg::rowNorm(d_centroids_row_norms, d_centroids, d, k, raft::linalg::L2Norm, true, stream);

            raft::linalg::rowNorm(d_points_row_norms, d_points, d, n, raft::linalg::L2Norm, true, stream);


            compute_gemm_distances_arizona(cublasHandle,
                                            d, n, k,
                                            d_points, d_points_row_norms,
                                            d_centroids, d_centroids_row_norms,
                                            d_distances);
            cudaDeviceSynchronize();
			cudaMemcpy(h_distances, d_distances, n * k * sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);
            
            if (TEST_DEBUG) {
                DATA_TYPE * h_norms_p = new DATA_TYPE[n];
                DATA_TYPE * h_norms_c = new DATA_TYPE[k];

                cudaMemcpy(h_norms_p, d_points_row_norms, sizeof(DATA_TYPE)*n, cudaMemcpyDeviceToHost);
                cudaMemcpy(h_norms_c, d_centroids_row_norms, sizeof(DATA_TYPE)*k, cudaMemcpyDeviceToHost);
                printf("NORM P\n");
                printMatrixColMajLimited(h_norms_p, n, 1, 10, 10);
                printf("NORM C\n");
                printMatrixColMajLimited(h_norms_c, k, 1, 10, 10);


                const DATA_TYPE alpha = -2.0;
                const DATA_TYPE beta = 0.0;
                
                /* -2.0*P*C */
                CHECK_CUBLAS_ERROR(cublasSgemm(cublasHandle,
                                                CUBLAS_OP_T, CUBLAS_OP_N,
                                                n, k, d,
                                                &alpha,
                                                d_points, d,
                                                d_centroids, d,
                                                &beta,
                                                d_distances, n));
                
                DATA_TYPE * h_distances_tmp = new DATA_TYPE[n*k];
                cudaMemcpy(h_distances_tmp, d_distances, sizeof(DATA_TYPE)*n*k, cudaMemcpyDeviceToHost);

                printMatrixColMajLimited(h_distances_tmp, n, k, 10, 10);

                /*
                DATA_TYPE * h_P_debug = new DATA_TYPE[p_size];
                cudaMemcpy(h_P_debug, d_P, p_size*sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);

                DATA_TYPE * h_C_debug = new DATA_TYPE[c_size];
                cudaMemcpy(h_C_debug, d_C, c_size*sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);

                std::cout<<"P MATRIX"<<std::endl;
                printMatrixColMajLimited(h_P_debug, p_rows, p_cols, 10, 10);

                std::cout<<"C MATRIX"<<std::endl;
                printMatrixColMajLimited(h_C_debug, c_rows, c_cols, 10, 10);

                delete[] h_P_debug;
                delete[] h_C_debug;
                */
            }
	

            if (TEST_DEBUG) {
                std::cout<<"Computed distances"<<std::endl;
                printMatrixColMajLimited(h_distances, n, k, 10, 10);
            }

			for (uint32_t ni = 0; ni < n; ++ni) {
				for (uint32_t ki = 0; ki < k; ++ki) {
					DATA_TYPE cpu_dist = 0, tmp;
					for (uint32_t di = 0; di < d; ++di) {
						tmp = h_points_row_maj[ni*d + di] - h_centroids_row_maj[ki*d + di];
						cpu_dist += tmp * tmp;
					}
					DATA_TYPE gpu_dist = h_distances[ni*k + ki];
					if (TEST_DEBUG && fabs(gpu_dist - cpu_dist) >= EPSILON) printf("point: %u center: %u gpu(%.6f) cpu(%.6f)\n", ni, ki, gpu_dist, cpu_dist);
					REQUIRE( fabs(gpu_dist - cpu_dist) < EPSILON );
				}
			}

			cublasDestroy(cublasHandle);
			delete[] h_points;
			delete[] h_points_row_maj;
			delete[] h_centroids;
            delete[] h_centroids_row_maj;
			delete[] h_distances;
			cudaFree(d_points);
            cudaFree(d_centroids);
			cudaFree(d_distances);
            cudaFree(d_points_row_norms);
            cudaFree(d_centroids_row_norms);
		}
	}

	compute_gemm_distances_free();
}



TEST_CASE("kernel_distances_matrix_bellavita", "[kernel][distances]") { // FIXME does not work well with N >= 500
	const unsigned int TESTS_N = 9;
	const unsigned int N[TESTS_N] = {10, 10, 17, 30, 17,	 15,	300,	2000, 1000};
	const unsigned int D[TESTS_N] = { 2,	2,	3, 11, 42, 1500,	400,	 200, 64};
	const unsigned int K[TESTS_N] = { 2,	6,	3, 11, 20,		5,	 10,	 200, 1000};

	getDeviceProps(0, &deviceProps);

	for (int test_i = 0; test_i < TESTS_N ; ++test_i) {
		const unsigned int n = N[test_i];
		const unsigned int d = D[test_i];
		const unsigned int k = K[test_i];

		char test_name[100];
		sprintf(test_name, "kernel compute_distances_matrix_bellavita n: %u	d: %u  k: %u", n, d, k);
		SECTION(test_name) {
			printf("Test: %s\n", test_name);

			DATA_TYPE *h_points = new DATA_TYPE[n * d];
			DATA_TYPE *h_points_row_maj = new DATA_TYPE[n * d];
			DATA_TYPE *h_centroids = new DATA_TYPE[k * d];
			DATA_TYPE *h_centroids_row_maj = new DATA_TYPE[k * d];
			DATA_TYPE *h_distances = new DATA_TYPE[n * k];

			// Constructing P and C
			initRandomMatrixColMaj(h_points, n, d);
			for (size_t i = 0; i < n; i++) {
				for (size_t j = 0; j < d; j++) {
					h_points_row_maj[i * d + j] = h_points[IDX2C(i, j, n)];
				}
			}


			DATA_TYPE* d_points;
			cudaMalloc(&d_points, n * d * sizeof(DATA_TYPE));
			cudaMemcpy(d_points, h_points_row_maj, n * d * sizeof(DATA_TYPE), cudaMemcpyHostToDevice);

			DATA_TYPE* d_centroids;
			cudaMalloc(&d_centroids, k * d * sizeof(DATA_TYPE));

            DATA_TYPE * d_points_row_norms;
            cudaMalloc(&d_points_row_norms, n*sizeof(DATA_TYPE));

            DATA_TYPE * d_centroids_row_norms;
            cudaMalloc(&d_centroids_row_norms, k*sizeof(DATA_TYPE));

			DATA_TYPE* d_distances;
			cudaMalloc(&d_distances, n * k * sizeof(DATA_TYPE));

            DATA_TYPE* d_B;
            cudaMalloc(&d_B, n*n*sizeof(DATA_TYPE));


			cublasHandle_t cublasHandle;
			cublasCreate(&cublasHandle);

            /* Init B */
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

            CHECK_CUDA_ERROR(cudaDeviceSynchronize());

            cusparseHandle_t cusparseHandle;
            cusparseCreate(&cusparseHandle);

            /* Init random V */
            DATA_TYPE * d_V_vals;
            int32_t * d_V_rowinds;
            int32_t * d_V_col_offsets;

            CHECK_CUDA_ERROR(cudaMalloc(&d_V_vals, sizeof(DATA_TYPE)*n));
            CHECK_CUDA_ERROR(cudaMalloc(&d_V_rowinds, sizeof(int32_t)*n));
            CHECK_CUDA_ERROR(cudaMalloc(&d_V_col_offsets, sizeof(int32_t)*(n+1)));

            std::vector<uint32_t> h_clusters_rand(n);
            std::generate(std::begin(h_clusters_rand), 
                          std::end(h_clusters_rand),
                          [k]()
                          { 
                            auto c = std::rand() % k;
                            return c;
                          });

            std::vector<uint32_t> h_clusters_len(k);
            std::for_each(h_clusters_rand.begin(),
                          h_clusters_rand.end(),
                          [&h_clusters_len](auto cluster)mutable
                          {
                              h_clusters_len[cluster]++;
                          });


            uint32_t * d_clusters;
            uint32_t * d_clusters_len;
            cudaMalloc(&d_clusters, sizeof(uint32_t)*n);
            cudaMalloc(&d_clusters_len, sizeof(uint32_t)*k);
            cudaMemcpy(d_clusters, h_clusters_rand.data(),
                        sizeof(uint32_t)*n,
                        cudaMemcpyHostToDevice);
            cudaMemcpy(d_clusters_len, h_clusters_len.data(),
                        sizeof(uint32_t)*k,
                        cudaMemcpyHostToDevice);


            const uint32_t v_mat_block_dim = min((size_t)n, (size_t)deviceProps.maxThreadsPerBlock);
            const uint32_t v_mat_grid_dim = ceil((float)n / (float)v_mat_block_dim);

            compute_v_sparse<<<v_mat_grid_dim, v_mat_block_dim>>>(d_V_vals, 
                                                                  d_V_rowinds, 
                                                                  d_V_col_offsets, 
                                                                  d_clusters, 
                                                                  d_clusters_len,
                                                                  n);


            /* Init cusparse descriptors */

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
                                                      d_centroids,
                                                      CUDA_R_32F,
                                                      CUSPARSE_ORDER_ROW));


            /* Construct C with random V and copy it to host centroids */
            compute_centroids_spmm(cusparseHandle,
                                    d, n, k,
                                    d_V_vals, 
                                    d_V_rowinds,
                                    d_V_col_offsets,
                                    d_centroids,
                                    V_descr,
                                    P_descr,
                                    C_descr);
            cudaMemcpy(h_centroids_row_maj, d_centroids, sizeof(DATA_TYPE)*k*d,
                        cudaMemcpyDeviceToHost);

			if (TEST_DEBUG) {
				printf("\nPOINTS %d:\n", n);
				printMatrixColMajLimited(h_points, n, d, 10, 10);

                for (int i=0; i<d; i++) {
                    for (int j=0; j<k; j++) {
                        h_centroids[j + i*k] = h_centroids_row_maj[i + j*d];
                    }
                }

				printf("\nCENTERS %d:\n", k);
				printMatrixColMajLimited(h_centroids, k, d, 10, 10);
			}


            raft::linalg::rowNorm(d_centroids_row_norms, d_centroids, d, k, raft::linalg::L2Norm, true, stream);

            raft::linalg::rowNorm(d_points_row_norms, d_points, d, n, raft::linalg::L2Norm, true, stream);


            compute_gemm_distances_bellavita(cusparseHandle,
                                            d, n, k,
                                            d_points_row_norms,
                                            d_centroids_row_norms,
                                            B_descr, V_descr,
                                            D_descr,
                                            d_distances);
            cudaDeviceSynchronize();

			cudaMemcpy(h_distances, d_distances, n * k * sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);
            
            if (TEST_DEBUG) {
                DATA_TYPE * h_norms_p = new DATA_TYPE[n];
                DATA_TYPE * h_norms_c = new DATA_TYPE[k];

                cudaMemcpy(h_norms_p, d_points_row_norms, sizeof(DATA_TYPE)*n, cudaMemcpyDeviceToHost);
                cudaMemcpy(h_norms_c, d_centroids_row_norms, sizeof(DATA_TYPE)*k, cudaMemcpyDeviceToHost);


                const DATA_TYPE alpha = -2.0;
                const DATA_TYPE beta = 0.0;
                
                /* -2.0*P*C */
                CHECK_CUBLAS_ERROR(cublasSgemm(cublasHandle,
                                                CUBLAS_OP_T, CUBLAS_OP_N,
                                                n, k, d,
                                                &alpha,
                                                d_points, d,
                                                d_centroids, d,
                                                &beta,
                                                d_distances, n));
                
                DATA_TYPE * h_distances_tmp = new DATA_TYPE[n*k];
                cudaMemcpy(h_distances_tmp, d_distances, sizeof(DATA_TYPE)*n*k, cudaMemcpyDeviceToHost);

                printMatrixColMajLimited(h_distances_tmp, n, k, 10, 10);

                /*
                DATA_TYPE * h_P_debug = new DATA_TYPE[p_size];
                cudaMemcpy(h_P_debug, d_P, p_size*sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);

                DATA_TYPE * h_C_debug = new DATA_TYPE[c_size];
                cudaMemcpy(h_C_debug, d_C, c_size*sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);

                std::cout<<"P MATRIX"<<std::endl;
                printMatrixColMajLimited(h_P_debug, p_rows, p_cols, 10, 10);

                std::cout<<"C MATRIX"<<std::endl;
                printMatrixColMajLimited(h_C_debug, c_rows, c_cols, 10, 10);

                delete[] h_P_debug;
                delete[] h_C_debug;
                */
            }
	

            if (TEST_DEBUG) {
                std::cout<<"Computed distances"<<std::endl;
                printMatrixColMajLimited(h_distances, n, k, 10, 10);
            }

			for (uint32_t ni = 0; ni < n; ++ni) {
				for (uint32_t ki = 0; ki < k; ++ki) {
					DATA_TYPE cpu_dist = 0, tmp;
					for (uint32_t di = 0; di < d; ++di) {
						tmp = h_points_row_maj[di + ni*d] - h_centroids_row_maj[di + ki*d];
						cpu_dist += tmp * tmp;
					}
					DATA_TYPE gpu_dist = h_distances[ki + ni*k];
					if (TEST_DEBUG && fabs(gpu_dist - cpu_dist) >= EPSILON) printf("point: %u center: %u gpu(%.6f) cpu(%.6f)\n", ni, ki, gpu_dist, cpu_dist);
					REQUIRE( fabs(gpu_dist - cpu_dist) < EPSILON );
				}
			}

			cublasDestroy(cublasHandle);

			delete[] h_points;
			delete[] h_points_row_maj;
			delete[] h_centroids;
            delete[] h_centroids_row_maj;
			delete[] h_distances;

			cudaFree(d_points);
            cudaFree(d_centroids);
			cudaFree(d_distances);
            cudaFree(d_points_row_norms);
            cudaFree(d_centroids_row_norms);
            cudaFree(d_clusters);
            cudaFree(d_clusters_len);
            cudaFree(d_V_vals);
            cudaFree(d_V_rowinds);
            cudaFree(d_V_col_offsets);
            cudaFree(d_B);

            cusparseDestroyDnMat(P_descr); 
            cusparseDestroyDnMat(C_descr); 
            cusparseDestroyDnMat(B_descr); 
            cusparseDestroyDnMat(D_descr); 
            cusparseDestroySpMat(V_descr); 

            cusparseDestroy(cusparseHandle);
		}
	}

	compute_gemm_distances_free();
}


TEST_CASE("kernel_argmin", "[kernel][argmin]") {
	const unsigned int TESTS_N = 9;
	const unsigned int N[TESTS_N] = {2, 10, 17, 51, 159, 1000, 3456, 10056, 100000};
	const unsigned int K[TESTS_N] = {1,  2,  7,  5, 129,	997, 1023, 1024, 10000};

	getDeviceProps(0, &deviceProps);

	for (int n_idx = 0; n_idx < TESTS_N; ++n_idx) {
		for (int k_idx = 0; k_idx < TESTS_N; ++k_idx) {
			const unsigned int n = N[n_idx];
			const unsigned int k = K[k_idx];
			const unsigned int SIZE = n * k;

			char test_name[50];
			sprintf(test_name, "kernel clusters_argmin n: %u  k: %u", n, k);
			SECTION(test_name) {
				printf("Test: %s\n", test_name);
				DATA_TYPE *h_distances = new DATA_TYPE[SIZE];
				for (uint32_t i = 0; i < n; ++i) {
					for (uint32_t j = 0; j < k; ++j) {
						h_distances[i * k + j] = static_cast <DATA_TYPE> (std::rand() / 105.456);
						if (TEST_DEBUG) { printf("%-2u %-2u -> %.0f\n", i, j, h_distances[i * k + j]); }
					}
				}

				DATA_TYPE *d_distances;
				cudaMalloc(&d_distances, sizeof(DATA_TYPE) * SIZE);
				cudaMemcpy(d_distances, h_distances, sizeof(DATA_TYPE) * SIZE,	cudaMemcpyHostToDevice);

                thrust::device_vector<uint32_t> d_clusters_vec(n);

				uint32_t* d_clusters_len;
				cudaMalloc(&d_clusters_len, k * sizeof(uint32_t));
				cudaMemset(d_clusters_len, 0, k * sizeof(uint32_t));

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
                KeyValueIndexOp<uint32_t, DATA_TYPE> conversion_op;
                cub::TransformInputIterator<uint32_t,
                                            KeyValueIndexOp<uint32_t, DATA_TYPE>,
                                            raft::KeyValuePair<uint32_t, DATA_TYPE>*>
                clusters(min_cluster_and_distance.data_handle(), conversion_op);

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

                thrust::copy(clusters, clusters+n, std::begin(d_clusters_vec));


                std::vector<uint32_t> h_points_clusters(n);
                thrust::copy(d_clusters_vec.begin(), d_clusters_vec.end(), h_points_clusters.begin());

                std::vector<uint32_t> h_clusters_len(k);
                std::vector<uint32_t> correct_clusters_len(k);
                cudaMemcpy(h_clusters_len.data(), d_clusters_len, sizeof(uint32_t)*k, 
                            cudaMemcpyDeviceToHost);

				for (uint32_t i = 0; i < n; i++) {
					DATA_TYPE min = infty;
					uint32_t idx = 0;
					for (uint32_t j = 0, ii = i * k; j < k; j++, ii++) {
						if (TEST_DEBUG) { printf("j: %u, ii: %u, v: %.0f\n", j, ii, h_distances[ii]); }
						if (h_distances[ii] < min) {
							min = h_distances[ii];
							idx = j;
						}
					}

                    correct_clusters_len[idx]++;

					REQUIRE( h_points_clusters[i] == idx );
					if (TEST_DEBUG) { printf("%-7u -> %5u (should be %-5u %.3f)\n", i, h_points_clusters[i], idx, min); }
				}

                for (int i=0; i<k; i++) {
                    REQUIRE(correct_clusters_len[i] == h_clusters_len[i]);
                }


				delete[] h_distances;
				cudaFree(d_distances);
				cudaFree(d_clusters_len);
			}
		}
	}
}



TEST_CASE("kernel_centroids_spmm", "[kernel][centroids]") {
	#define TESTS_N 8
	const unsigned int D[TESTS_N] = {2,  3,  10,	32,  50,	100, 1000, 1024};
	const unsigned int N[TESTS_N] = {2, 10, 100,	51, 159, 1000, 3456, 10056};
	const unsigned int K[TESTS_N] = {1,  4,		7,	10, 129,	997, 1023, 1024};

	getDeviceProps(0, &deviceProps);

	for (int d_idx = 0; d_idx < TESTS_N; ++d_idx) {
		for (int n_idx = 0; n_idx < TESTS_N; ++n_idx) {
			for (int k_idx = 0; k_idx < TESTS_N; ++k_idx) {
				const unsigned int d = D[d_idx];
				const unsigned int n = N[n_idx];
				const unsigned int k = K[k_idx];
				char test_name[50];

				snprintf(test_name, 49, "kernel centroids spmm d=%u n=%u k=%u", d, n, k);

				SECTION(test_name) {
					printf("Test: %s\n", test_name);
					DATA_TYPE *h_centroids = new DATA_TYPE[k * d];
					DATA_TYPE *h_points = new DATA_TYPE[n * d];
					uint32_t	*h_points_clusters = new uint32_t[n];
					uint32_t	*h_clusters_len = new uint32_t[k];

					memset(h_clusters_len, 0, k * sizeof(uint32_t));
					for (uint32_t i = 0; i < n; ++i) {
						h_points_clusters[i] = (static_cast <uint32_t> (std::rand() % k));
						h_clusters_len[h_points_clusters[i]]++;
						for (uint32_t j = 0; j < d; ++j) {
							h_points[i * d + j] = (static_cast <DATA_TYPE> (std::rand() / 1000.0)) / 1000.00;
						}
					}

					memset(h_centroids, 0, k * d * sizeof(DATA_TYPE));
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


					DATA_TYPE* d_centroids;
					CHECK_CUDA_ERROR(cudaMalloc(&d_centroids, k * d * sizeof(DATA_TYPE)));
					DATA_TYPE* d_points;
					CHECK_CUDA_ERROR(cudaMalloc(&d_points, n * d * sizeof(DATA_TYPE)));
					CHECK_CUDA_ERROR(cudaMemcpy(d_points, h_points, n * d * sizeof(DATA_TYPE), cudaMemcpyHostToDevice));
					uint32_t* d_points_clusters;
					CHECK_CUDA_ERROR(cudaMalloc(&d_points_clusters, n * sizeof(uint32_t)));
					CHECK_CUDA_ERROR(cudaMemcpy(d_points_clusters, h_points_clusters, n * sizeof(uint32_t), cudaMemcpyHostToDevice));
					uint32_t* d_clusters_len;
					CHECK_CUDA_ERROR(cudaMalloc(&d_clusters_len, k * sizeof(uint32_t)));
					CHECK_CUDA_ERROR(cudaMemcpy(d_clusters_len, h_clusters_len, k * sizeof(uint32_t), cudaMemcpyHostToDevice));


                    DATA_TYPE * d_V_vals;
                    int32_t * d_V_rowinds;
                    int32_t * d_V_col_offsets;

                    cudaMalloc(&d_V_vals, n * sizeof(DATA_TYPE));
                    cudaMalloc(&d_V_rowinds, n * sizeof(int32_t));
                    cudaMalloc(&d_V_col_offsets, (n+1) * sizeof(int32_t));

					
					cusparseSpMatDescr_t V_descr;
					cusparseDnMatDescr_t P_descr;                                                              cusparseDnMatDescr_t C_descr;

					CHECK_CUSPARSE_ERROR(cusparseCreateCsc(&V_descr,
															k, n, n,
															d_V_col_offsets,
															d_V_rowinds,
															d_V_vals,
															CUSPARSE_INDEX_32I,                                                                        CUSPARSE_INDEX_32I,
															CUSPARSE_INDEX_BASE_ZERO,
															CUDA_R_32F));
					
					
					CHECK_CUSPARSE_ERROR(cusparseCreateDnMat(&P_descr,
															  n, d, d,
															  d_points,
															  CUDA_R_32F,
															  CUSPARSE_ORDER_ROW));

					CHECK_CUSPARSE_ERROR(cusparseCreateDnMat(&C_descr,
															  k, d, d,
															  d_centroids,
															  CUDA_R_32F,
															  CUSPARSE_ORDER_ROW));



                    uint32_t v_mat_block_dim = min((uint32_t)n, (uint32_t)deviceProps.maxThreadsPerBlock);
                    uint32_t v_mat_grid_dim = ceil((float)n/(float)v_mat_block_dim);

                    compute_v_sparse<<<v_mat_grid_dim, v_mat_block_dim>>>
                                    (d_V_vals, d_V_rowinds, d_V_col_offsets,
                                     d_points_clusters,
                                     d_clusters_len,
                                     n);

                    cusparseHandle_t handle;
                    cusparseCreate(&handle);

                    compute_centroids_spmm(handle, 
                                            d, n, k,
                                            d_V_vals, 
                                            d_V_rowinds,
                                            d_V_col_offsets,
                                            d_centroids,
				                            V_descr,
                                            P_descr,
                                            C_descr);
                                                                            

					CHECK_CUDA_ERROR(cudaDeviceSynchronize());
					CHECK_LAST_CUDA_ERROR();
                    
					DATA_TYPE *h_centroids_cpy = new DATA_TYPE[k * d];
					CHECK_CUDA_ERROR(cudaMemcpy(h_centroids_cpy, d_centroids, k * d * sizeof(DATA_TYPE), cudaMemcpyDeviceToHost));
                    

					const DATA_TYPE EPSILON = numeric_limits<DATA_TYPE>::round_error();
					bool is_equal = true;
					for (uint32_t i = 0; i < k; ++i) {
						for (uint32_t j = 0; j < d; ++j) {
							is_equal &= fabs(h_centroids[i * d + j] - h_centroids_cpy[i * d + j]) < EPSILON;
						}
					}

					delete[] h_centroids;
					delete[] h_centroids_cpy;
					delete[] h_points;
					delete[] h_points_clusters;
					delete[] h_clusters_len;
					cudaFree(d_centroids);
					cudaFree(d_points);
					cudaFree(d_clusters_len);

					REQUIRE(is_equal);
				}
			}
		}
	}
}


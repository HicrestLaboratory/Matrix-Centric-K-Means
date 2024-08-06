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

#define TEST_DEBUG 0
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
const DATA_TYPE EPSILON = 1e-3;

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

void write_mat_file(std::ofstream& out,
                    DATA_TYPE * A,
                    const uint32_t n,
                    const uint32_t d,
                    const std::string prefix)
{
    out<<prefix<<std::endl;
    for (int i=0; i<n; i++) {
        for (int j=0; j<d; j++) {
            out<<A[j + i*d]<<",";
        }
        out<<std::endl;
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

TEST_CASE("kernel_kernel_mtx_naive", "[kernel][kernel]") { // FIXME does not work well with N >= 500
    /*
	const unsigned int TESTS_N = 9;
	const unsigned int N[TESTS_N] = {10, 10, 17, 30, 17,	 15,	300,	2000};
	const unsigned int D[TESTS_N] = { 1,	2,	3, 11, 42, 32,	64,	 200};
	const unsigned int K[TESTS_N] = { 2,	6,	3, 11, 20,		5,	 10,	 200};
    */
	const unsigned int TESTS_N = 6;
    const unsigned int N[TESTS_N] = {10, 30, 31, 107, 200, 10500};
    const unsigned int D[TESTS_N] = {32, 64, 17, 42, 256, 16};
    const unsigned int K[TESTS_N] = {2, 2, 2, 2, 2, 10};

	getDeviceProps(0, &deviceProps);

    cublasHandle_t cublasHandle;
    cublasCreate(&cublasHandle);


	for (int test_i = 0; test_i < TESTS_N ; ++test_i) {
		const unsigned int n = N[test_i];
		const unsigned int d = D[test_i];
		const unsigned int k = K[test_i];

        const unsigned int d_pow2 = pow(2, ceil(log2(d)));

		char test_name[100];
		sprintf(test_name, "kernel compute_kernel_mtx n: %u	d: %u  k: %u", n, d, k);
		SECTION(test_name) {
			printf("Test: %s\n", test_name);

#if TEST_DEBUG
            std::ofstream logfile;
            logfile.open("logfile_kernel_" +std::to_string(test_i)+".out");
            logfile<<"n:"<<n<<" d:"<<d<<" k:"<<k<<std::endl;
#endif

			DATA_TYPE *h_points = new DATA_TYPE[n * d];
			DATA_TYPE *h_points_row_maj = new DATA_TYPE[n * d];

			// Constructing P and C
			initRandomMatrixColMaj(h_points, n, d);
			for (size_t i = 0; i < n; i++) {
				for (size_t j = 0; j < d; j++) {
					h_points_row_maj[i * d + j] = h_points[IDX2C(i, j, n)];
				}
			}

#if LOG
            printf("\nPOINTS %d:\n", n);
            printMatrixColMajLimited(h_points, n, d, 10, 10);
#endif

			DATA_TYPE* d_points;
			cudaMalloc(&d_points, n * d * sizeof(DATA_TYPE));
			cudaMemcpy(d_points, h_points_row_maj, n * d * sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
			DATA_TYPE* d_kernel;
			cudaMalloc(&d_kernel, n * n * sizeof(DATA_TYPE));

			DATA_TYPE* d_kernel_correct;
			cudaMalloc(&d_kernel_correct, n * n * sizeof(DATA_TYPE));

            DATA_TYPE b_beta = 0.0;
            DATA_TYPE b_alpha = 1.0;
            auto stime_gemm = std::chrono::high_resolution_clock::now();
            CHECK_CUBLAS_ERROR(cublasSgemm(cublasHandle, 
                                            CUBLAS_OP_T,
                                            CUBLAS_OP_N,
                                            n, n, d,
                                            &b_alpha,
                                            d_points, d,
                                            d_points, d,
                                            &b_beta,
                                            d_kernel_correct, n));
            CHECK_CUDA_ERROR(cudaDeviceSynchronize());
            auto etime_gemm = std::chrono::high_resolution_clock::now();
            auto duration_gemm = std::chrono::duration_cast<std::chrono::duration<double>>(etime_gemm- stime_gemm).count();
            std::cout<<"GEMM: "<<duration_gemm<<"s"<<std::endl;

            const uint32_t tpb = 512;
            const uint32_t wpb = tpb / 32;
            const uint64_t blocks = ceil((double)(n*n) / (double)wpb);
            auto stime_naive = std::chrono::high_resolution_clock::now();
            compute_kernel_matrix_naive<<<blocks, tpb>>>(d_kernel, d_points, n, d, d_pow2);
            CHECK_CUDA_ERROR(cudaDeviceSynchronize());
            auto etime_naive = std::chrono::high_resolution_clock::now();
            auto duration_naive = std::chrono::duration_cast<std::chrono::duration<double>>(etime_naive - stime_naive).count();
            std::cout<<"NAIVE: "<<duration_naive<<"s"<<std::endl;


            DATA_TYPE * h_correct = new DATA_TYPE[n*n];
            DATA_TYPE * h_computed = new DATA_TYPE[n*n];
            cudaMemcpy(h_correct, d_kernel_correct, sizeof(DATA_TYPE)*n*n, cudaMemcpyDeviceToHost);
            cudaMemcpy(h_computed, d_kernel, sizeof(DATA_TYPE)*n*n, cudaMemcpyDeviceToHost);

#if TEST_DEBUG
            write_mat_file(logfile, h_correct, n, n, "CORRECT");
            write_mat_file(logfile, h_computed, n, n, "COMPUTED");
#endif

			for (uint32_t ni = 0; ni < n; ++ni) {
				for (uint32_t ni2 = 0; ni2 < n; ++ni2) {
					REQUIRE( fabs(h_correct[ni2 + ni*n] - h_computed[ni2 + ni*n] ) < EPSILON );
				}
			}

			cublasDestroy(cublasHandle);
			delete[] h_points;
			delete[] h_points_row_maj;
            delete[] h_correct;
            delete[] h_computed;
			cudaFree(d_points);
			cudaFree(d_kernel);
#if TEST_DEBUG
            logfile.close();
#endif
		}
	}

}

TEST_CASE("kernel_compute_distances_naive", "[kernel][kernel]") { // FIXME does not work well with N >= 500
    /*
	const unsigned int TESTS_N = 9;
	const unsigned int N[TESTS_N] = {10, 10, 17, 30, 17,	 15,	300,	2000};
	const unsigned int D[TESTS_N] = { 1,	2,	3, 11, 42, 32,	64,	 200};
	const unsigned int K[TESTS_N] = { 2,	6,	3, 11, 20,		5,	 10,	 200};
    */
	const unsigned int TESTS_N = 6;
    const unsigned int N[TESTS_N] = {10, 30, 31, 107, 200, 10500};
    const unsigned int D[TESTS_N] = {32, 64, 17, 42, 256, 16};
    const unsigned int K[TESTS_N] = {2, 4, 6, 10, 100, 10};

	getDeviceProps(0, &deviceProps);

    cublasHandle_t cublasHandle;
    cublasCreate(&cublasHandle);


	for (int test_i = 0; test_i < 6; ++test_i) {
		const unsigned int n = N[test_i];
		const unsigned int d = D[test_i];
		const unsigned int k = K[test_i];

        const unsigned int d_pow2 = pow(2, ceil(log2(d)));

		char test_name[100];
		sprintf(test_name, "kernel compute_distances_naive n: %u	d: %u  k: %u", n, d, k);
		SECTION(test_name) {
			printf("Test: %s\n", test_name);

#if TEST_DEBUG
            std::ofstream logfile;
            logfile.open("logfile_kernel_" +std::to_string(test_i)+".out");
            logfile<<"n:"<<n<<" d:"<<d<<" k:"<<k<<std::endl;
#endif

			DATA_TYPE * h_points = new DATA_TYPE[n * d];
			DATA_TYPE * h_points_row_maj = new DATA_TYPE[n * d];
			DATA_TYPE * h_kernel = new DATA_TYPE[n * n];
            DATA_TYPE * h_tmp = new DATA_TYPE[n*k];
            DATA_TYPE * h_distances_correct = new DATA_TYPE[n*k];
            DATA_TYPE * h_distances_computed = new DATA_TYPE[n*k];
            std::memset(h_tmp, 0, sizeof(DATA_TYPE)*n*k);
            std::memset(h_distances_correct, 0, sizeof(DATA_TYPE)*n*k);

			initRandomMatrixColMaj(h_points, n, d);
			for (size_t i = 0; i < n; i++) {
				for (size_t j = 0; j < d; j++) {
					h_points_row_maj[i * d + j] = h_points[IDX2C(i, j, n)];
				}
			}

            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<uint32_t> distr(0, k-1);

            std::vector<uint32_t> h_clusters(n);
            for (int i=0; i<n; i++) {
                h_clusters[i] = i % k;
            }

            std::vector<uint32_t> h_clusters_len(k);
            std::for_each(h_clusters.begin(), h_clusters.end(), 
                            [&](auto const& cluster)mutable {h_clusters_len[cluster] += 1;});

            uint32_t * d_clusters;
            cudaMalloc(&d_clusters, sizeof(uint32_t)*n);
            cudaMemcpy(d_clusters, h_clusters.data(), sizeof(uint32_t)*n,
                            cudaMemcpyHostToDevice);

            uint32_t * d_clusters_len;
            cudaMalloc(&d_clusters_len, sizeof(uint32_t)*k);
            cudaMemcpy(d_clusters_len, h_clusters_len.data(), sizeof(uint32_t)*k,
                            cudaMemcpyHostToDevice);

			DATA_TYPE * d_kernel;
			cudaMalloc(&d_kernel, n * n * sizeof(DATA_TYPE));

            DATA_TYPE * d_points;
            cudaMalloc(&d_points, n*d*sizeof(DATA_TYPE));
            cudaMemcpy(d_points, h_points_row_maj, sizeof(DATA_TYPE)*n*d,
                            cudaMemcpyHostToDevice);

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
                                            d_kernel, n));
            cudaMemcpy(h_kernel, d_kernel, sizeof(DATA_TYPE)*n*n,
                        cudaMemcpyDeviceToHost);

            DATA_TYPE * d_distances;
			cudaMalloc(&d_distances, n * k * sizeof(DATA_TYPE));

            DATA_TYPE * d_tmp;
			cudaMalloc(&d_tmp, n * k * sizeof(DATA_TYPE));

            auto stime = std::chrono::high_resolution_clock::now();
            // Reduce by keys to compute sums of inner products in same cluster
            const uint32_t reduce_tpb = 1024;
            const uint32_t reduce_blocks = n;
            const uint32_t n_thread_ceil = ceil((double)n / (double) reduce_tpb) * reduce_tpb;
            sum_points<<<reduce_blocks, reduce_tpb>>>(d_kernel,
                                                      d_clusters,
                                                      d_clusters_len,
                                                      d_tmp,
                                                      n, k, n_thread_ceil);
            CHECK_CUDA_ERROR(cudaDeviceSynchronize());

#if TEST_DEBUG
            cudaMemcpy(h_distances_computed, d_tmp, sizeof(DATA_TYPE)*n*k, cudaMemcpyDeviceToHost);
            write_mat_file(logfile, h_distances_computed, n, k, "TMP-COMPUTED");
#endif
            DATA_TYPE * d_centroids;
            cudaMalloc(&d_centroids, sizeof(DATA_TYPE)*k);

            const uint32_t centroid_tpb = 1024;
            const uint32_t centroid_blocks = ceil(((uint64_t)n * (uint64_t)n) / (double)centroid_tpb); 
            sum_centroids<<<centroid_blocks, centroid_tpb>>>(d_kernel,
                                                              d_clusters,
                                                              d_clusters_len,
                                                              d_centroids,
                                                              n, k);
            CHECK_CUDA_ERROR(cudaDeviceSynchronize());

            const uint32_t distances_tpb = 1024;
            const uint32_t distances_blocks = ceil( (double)(n*k) / (double)distances_tpb);
            compute_distances_naive<<<distances_blocks, distances_tpb>>>
                                    (d_kernel, d_centroids, d_tmp, d_distances, n, k);
            CHECK_CUDA_ERROR(cudaDeviceSynchronize());

            auto etime = std::chrono::high_resolution_clock::now();
            auto duration= std::chrono::duration_cast<std::chrono::duration<double>>(etime- stime).count();
            std::cout<<"Time: "<<duration<<"s"<<std::endl;


            cudaMemcpy(h_distances_computed, d_distances, sizeof(DATA_TYPE)*n*k, cudaMemcpyDeviceToHost);


            for (uint64_t ni = 0; ni < n; ++ni) {
                for (uint64_t ni2 = 0; ni2 < n; ++ni2) {
                    uint64_t cluster = h_clusters[ni2];
                    h_tmp[ni * k + cluster] += (h_kernel[ni*n + ni2]/h_clusters_len[cluster]);
                }

            }

#if TEST_DEBUG
            write_mat_file(logfile, h_tmp, n, k, "TMP-CORRECT");
#endif


            for (uint64_t ni3=0; ni3<n; ++ni3) {
                for (uint64_t ni = 0; ni < n; ++ni) {
                    uint64_t cluster = h_clusters[ni];
                    for (uint64_t ni2 = 0; ni2 < n; ++ni2) {
                        uint64_t cluster2 = h_clusters[ni2];
                        if (cluster==cluster2) {
                            h_distances_correct[ni3 * k + cluster] += h_kernel[ni * n + ni2]/(double)pow(h_clusters_len[cluster], 2);
                        }
                    }
                }
            }

            for (uint32_t ni = 0; ni < n; ++ni) {
                for (uint32_t ki = 0; ki < k; ++ki) {
                    h_distances_correct[ni * k + ki] += h_kernel[ni*n + ni];
                    h_distances_correct[ni * k + ki] += -2*h_tmp[ni * k + ki];
                }
            }


#if TEST_DEBUG
            write_mat_file(logfile, h_distances_correct, n, k, "CORRECT");
            write_mat_file(logfile, h_distances_computed, n, k, "COMPUTED");
            printf("\nKERNEL %d:\n", n);
            printMatrixColMajLimited(h_kernel, n, n, 10, 10);
#endif

			for (uint32_t ni = 0; ni < n; ++ni) {
				for (uint32_t ni2 = 0; ni2 < k; ++ni2) {

					REQUIRE( fabs(h_distances_correct[ni2 + ni*k] - h_distances_computed[ni2 + ni*k] ) < EPSILON );
				}
			}

			cublasDestroy(cublasHandle);
			delete[] h_kernel;
			delete[] h_points;
            delete[] h_tmp;
			delete[] h_points_row_maj;
            delete[] h_distances_correct;
            delete[] h_distances_computed;
			cudaFree(d_distances);
			cudaFree(d_kernel);
			cudaFree(d_points);
			cudaFree(d_tmp);
			cudaFree(d_clusters);
			cudaFree(d_clusters_len);
            cudaFree(d_centroids);
#if TEST_DEBUG
            logfile.close();
#endif
		}
	}

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



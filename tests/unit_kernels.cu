#include <catch2/catch_test_macros.hpp>

#include <limits>
#include <iostream>
#include <math.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "../src/kernels/kernels.cuh"
#include "../src/cuda_utils.cuh"
#include "../src/include/common.h"

#define TEST_DEBUG 0
#define WARP_SIZE  32

const DATA_TYPE infty		= numeric_limits<DATA_TYPE>::infinity();
const DATA_TYPE EPSILON = numeric_limits<DATA_TYPE>::epsilon();

cudaDeviceProp deviceProps;

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


TEST_CASE("kernel_distances_matrix_fast", "[kernel][distances]") { // FIXME does not work well with N >= 500
	const unsigned int TESTS_N = 8;
	const unsigned int N[TESTS_N] = {10, 10, 17, 30, 17,	 15,	300,	2000};
	const unsigned int D[TESTS_N] = { 1,	2,	3, 11, 42, 1500,	400,	 200};
	const unsigned int K[TESTS_N] = { 2,	6,	3, 11, 20,		5,	 10,	 200};

	getDeviceProps(0, &deviceProps);

	for (int test_i = 0; test_i < TESTS_N - 1; ++test_i) {
		const unsigned int n = N[test_i];
		const unsigned int d = D[test_i];
		const unsigned int k = K[test_i];

		char test_name[100];
		sprintf(test_name, "kernel compute_distances_matrix_fast n: %u	d: %u  k: %u", n, d, k);
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
				printMatrixColMajLimited(h_points, n, d, 10, 5);
				printf("\nCENTERS %d:\n", k);
				printMatrixColMajLimited(h_centroids, k, d, 10, 5);
			}

			size_t p_rows = n;
			size_t p_cols = 3*d;
			size_t p_size = p_rows*p_cols;

			size_t c_rows = 3*d;
			size_t c_cols = k;
			size_t c_size = c_rows*c_cols;

			DATA_TYPE* d_points;
			cudaMalloc(&d_points, n * d * sizeof(DATA_TYPE));
			cudaMemcpy(d_points, h_points_row_maj, n * d * sizeof(DATA_TYPE), cudaMemcpyHostToDevice);

			DATA_TYPE* d_centroids;
			cudaMalloc(&d_centroids, k * d * sizeof(DATA_TYPE));
			cudaMemcpy(d_centroids, h_centroids, k * d * sizeof(DATA_TYPE), cudaMemcpyHostToDevice);

			DATA_TYPE* d_P;
			cudaMalloc(&d_P, p_size * sizeof(DATA_TYPE));
			cudaMemset(d_P, 0, p_size * sizeof(DATA_TYPE));

			DATA_TYPE * d_C;
	        cudaMalloc(&d_C, c_size*sizeof(DATA_TYPE));
            cudaMemset(d_C, 0, c_size*sizeof(DATA_TYPE));

			DATA_TYPE* d_distances;
			cudaMalloc(&d_distances, n * k * sizeof(DATA_TYPE));

			uint32_t p_mat_block_dim(min((size_t)deviceProps.maxThreadsPerBlock, p_rows));
			uint32_t p_mat_grid_dim(p_cols);
			uint32_t p_rounds = ceil((float)p_rows / (float)p_mat_block_dim);
			compute_p_matrix<<<p_mat_grid_dim, p_mat_block_dim>>>(d_points, d_P, d, n, k, p_rounds);
			
			uint32_t c_mat_grid_dim = c_cols;
			uint32_t c_mat_block_dim = min((size_t)deviceProps.maxThreadsPerBlock, c_rows/3);
			uint32_t c_rounds = ceil((float)(c_rows/3) / (float)c_mat_block_dim);
			compute_c_matrix_col_major<<<c_mat_grid_dim, c_mat_block_dim>>>(d_centroids, d_C, d, n, k, c_rounds);
            
            if (TEST_DEBUG) {
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
            }
	
			// Test function compute_gemm_distances
			cublasHandle_t cublasHandle;
			cublasCreate(&cublasHandle);
			compute_gemm_distances_fast(cublasHandle, 
                                        d, n, k, 
                                        d_P, d_C, d_distances);
            cudaDeviceSynchronize();
			cudaMemcpy(h_distances, d_distances, n * k * sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);

            if (TEST_DEBUG) {
                std::cout<<"Computed distances"<<std::endl;
                printMatrixColMajLimited(h_distances, n, k, 10, 10);
            }

			for (uint32_t ni = 0; ni < n; ++ni) {
				for (uint32_t ki = 0; ki < k; ++ki) {
					DATA_TYPE cpu_dist = 0, tmp;
					for (uint32_t di = 0; di < d; ++di) {
						tmp = h_points[IDX2C(ni, di, n)] - h_centroids[IDX2C(ki, di, k)];
						cpu_dist += tmp * tmp;
					}
					DATA_TYPE gpu_dist = h_distances[ni + ki*n];
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
			cudaFree(d_C);
			cudaFree(d_P);
		}
	}

	compute_gemm_distances_free();
}


TEST_CASE("kernel_distances_matrix", "[kernel][distances]") { // FIXME does not work well with N >= 500
	const unsigned int TESTS_N = 8;
	const unsigned int N[TESTS_N] = {10, 10, 17, 30, 17,	 15,	300,	2000};
	const unsigned int D[TESTS_N] = { 1,	2,	3, 11, 42, 1500,	400,	 200};
	const unsigned int K[TESTS_N] = { 2,	6,	3, 11, 20,		5,	 10,	 200};

	getDeviceProps(0, &deviceProps);

	for (int test_i = 0; test_i < TESTS_N - 1; ++test_i) {
		const unsigned int n = N[test_i];
		const unsigned int d = D[test_i];
		const unsigned int d1 = d + 1;
		const unsigned int d1d1 = d1 * d1;
		const unsigned int nd1d1 = n * d1 * d1;
		const unsigned int k = K[test_i];
		const unsigned int rounds = ((d - 1) / WARP_SIZE) + 1;

		char test_name[50];
		sprintf(test_name, "kernel compute_distances_matrix n: %u  d: %u	k: %u", n, d, k);
		SECTION(test_name) {
			printf("Test: %s\n", test_name);

			DATA_TYPE *h_points = new DATA_TYPE[n * d];
			DATA_TYPE *h_points_row_maj = new DATA_TYPE[n * d];
			DATA_TYPE *h_centroids = new DATA_TYPE[k * d1];
			DATA_TYPE *h_distances = new DATA_TYPE[n * k];
			DATA_TYPE* h_P_CPU = new DATA_TYPE[d1 * d1];
			DATA_TYPE* h_Ps_GPU = new DATA_TYPE[d1 * d1 * n];

			// Constructing P and C
			initRandomMatrixColMaj(h_points, n, d);
			for (size_t i = 0; i < n; i++) {
				for (size_t j = 0; j < d; j++) {
					h_points_row_maj[i * d + j] = h_points[IDX2C(i, j, n)];
				}
			}

			initRandomMatrixColMaj(h_centroids, k, d1);
			for (size_t i = 0; i < k; ++i) {
				h_centroids[i] = 1;
			}
			if (TEST_DEBUG) {
				printf("\nPOINTS %d:\n", n);
				printMatrixColMajLimited(h_points, n, d, 10, 5);
				printf("\nCENTERS %d:\n", k);
				printMatrixColMajLimited(h_centroids, k, d1, 10, 5);
			}

			DATA_TYPE* d_points;
			cudaMalloc(&d_points, n * d * sizeof(DATA_TYPE));
			cudaMemcpy(d_points, h_points_row_maj, n * d * sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
			DATA_TYPE* d_P;
			cudaMalloc(&d_P, nd1d1 * sizeof(DATA_TYPE));
			cudaMemset(d_P, 0, nd1d1 * sizeof(DATA_TYPE));
			DATA_TYPE* d_C;
			cudaMalloc(&d_C, k * d1 * sizeof(DATA_TYPE));
			cudaMemcpy(d_C, h_centroids, k * d1 * sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
			DATA_TYPE* d_distances;
			cudaMalloc(&d_distances, n * k * sizeof(DATA_TYPE));

			// Test kernel compute_point_associated_matrices
			dim3 dist_assoc_matrices_grid_dim(n);
			dim3 dist_assoc_matrices_block_dim(min(next_pow_2(d), WARP_SIZE));
			for (uint32_t i = 0; i < rounds; i++) {
				compute_point_associated_matrices<<<dist_assoc_matrices_grid_dim, dist_assoc_matrices_block_dim>>>(d_points, d_P, d, i);
			}
			cudaMemcpy(h_Ps_GPU, d_P, nd1d1 * sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);

			// Test function compute_gemm_distances
			cublasHandle_t cublasHandle;
			cublasCreate(&cublasHandle);
			compute_gemm_distances(cublasHandle,&deviceProps, d1, n, k, d_P, d_C, d_distances);
			cudaMemcpy(h_distances, d_distances, n * k * sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);

			for (uint32_t ni = 0; ni < n; ++ni) {
				computeCPUCentroidAssociatedMatrix(h_P_CPU, h_points, d, ni, n);
				DATA_TYPE* h_P_GPU = h_Ps_GPU + (d1d1 * ni);

				// Check associated matrices
				for (size_t i = 0; i < d1; i++) {
					for (size_t j = 0; j < i; j++) {
						if (TEST_DEBUG && h_P_CPU[j * d1 + i] != h_P_GPU[j * d1 + i]) {
							printf("Associated matrix error at (%lu, %lu)", i, j);
							printf("\nPoint %u associated matrix:\n", ni);
							printMatrixColMajLimited(h_P_CPU, d1, d1, 15, 15);
							printf("\nGPU:\n");
							printMatrixColMajLimited(h_P_GPU, d1, d1, 15, 15);
						}
						REQUIRE( h_P_CPU[j * d1 + i] == h_P_GPU[j * d1 + i] );
					}
				}

				for (uint32_t ki = 0; ki < k; ++ki) {
					DATA_TYPE cpu_dist = 0, tmp;
					for (uint32_t di = 0; di < d; ++di) {
						tmp = h_points[IDX2C(ni, di, n)] - h_centroids[IDX2C(ki, di + 1, k)];
						cpu_dist += tmp * tmp;
					}
					DATA_TYPE gpu_dist = h_distances[ni * k + ki];
					if (TEST_DEBUG && fabs(gpu_dist - cpu_dist) >= EPSILON) printf("point: %u center: %u gpu(%.6f) cpu(%.6f)\n", ni, ki, gpu_dist, cpu_dist);
					REQUIRE( fabs(gpu_dist - cpu_dist) < EPSILON );
				}
			}

			cublasDestroy(cublasHandle);
			delete[] h_points;
			delete[] h_points_row_maj;
			delete[] h_P_CPU;
			delete[] h_Ps_GPU;
			delete[] h_centroids;
			delete[] h_distances;
			cudaFree(d_points);
			cudaFree(d_distances);
			cudaFree(d_C);
			cudaFree(d_P);
		}
	}

	compute_gemm_distances_free();
}

TEST_CASE("kernel_distances_warp", "[kernel][distances]") {
	const unsigned int TESTS_N = 9;
	const unsigned int N[TESTS_N] = {10, 10, 17, 51, 159, 3000, 1000, 3456, 10056};
	const unsigned int D[TESTS_N] = { 1, 47, 92,	5,	11,		 2,		12,		24,		 32};
	const unsigned int K[TESTS_N] = { 2,	6,	3, 28,	 7,		20,  500, 1763,  9056};

	for (int i = 0; i < TESTS_N; ++i) {
		const unsigned int n = N[i];
		const unsigned int d = D[i];
		const unsigned int k = K[i];

		char test_name[50];
		sprintf(test_name, "kernel compute_distances_shfl n: %u  d: %u	k: %u", n, d, k);

		SECTION(test_name) {
			printf("Test: %s\n", test_name);
			DATA_TYPE *h_points = new DATA_TYPE[n * d];
			DATA_TYPE *h_centroids = new DATA_TYPE[k * d];
			DATA_TYPE *h_distances = new DATA_TYPE[n * k];
			DATA_TYPE *h_distances_1 = new DATA_TYPE[n * k];
			for (uint32_t i = 0; i < n; ++i) {
				for (uint32_t j = 0; j < d; ++j) {
					h_points[i * d + j] = std::rand() / 100000002.32;
				}
			}
			if (TEST_DEBUG) {
				printf("Points:\n");
				printMatrixRowMajLimited(h_points, n, d, 10, 10);
			}
			for (uint32_t i = 0; i < k; ++i) {
				for (uint32_t j = 0; j < d; ++j) {
					h_centroids[i * d + j] = static_cast <DATA_TYPE> (std::rand() / 100000002.45);
				}
			}
			if (TEST_DEBUG) {
				printf("\nCentroids:\n");
				printMatrixRowMajLimited(h_centroids, k, d, 10, 10);
			}

			DATA_TYPE *d_distances;
			cudaMalloc(&d_distances, sizeof(DATA_TYPE) * n * k);
			DATA_TYPE *d_distances_1;
			cudaMalloc(&d_distances_1, sizeof(DATA_TYPE) * n * k);
			DATA_TYPE *d_points;
			cudaMalloc(&d_points, sizeof(DATA_TYPE) * n * d);
			cudaMemcpy(d_points, h_points, sizeof(DATA_TYPE) * n * d, cudaMemcpyHostToDevice);
			DATA_TYPE *d_centroids;
			cudaMalloc(&d_centroids, sizeof(DATA_TYPE) * k * d);
			cudaMemcpy(d_centroids, h_centroids, sizeof(DATA_TYPE) * k * d, cudaMemcpyHostToDevice);

			// Test kernel SHUFFLE
			if (d <= WARP_SIZE) {
				const uint32_t dist_max_points_per_warp = WARP_SIZE / next_pow_2(d);
				dim3 dist_grid_dim(ceil(((float) n) / dist_max_points_per_warp), k);
				dim3 dist_block_dim(dist_max_points_per_warp * next_pow_2(d));
				compute_distances_shfl<<<dist_grid_dim, dist_block_dim>>>(d_distances, d_centroids, d_points, n, dist_max_points_per_warp, d, log2(next_pow_2(d)) > 0 ? log2(next_pow_2(d)) : 1);
				cudaMemcpy(h_distances, d_distances, sizeof(DATA_TYPE) * n * k,  cudaMemcpyDeviceToHost);
			}

			// Test kernel ONE POINT PER WARP
			const uint32_t rounds = ((d - 1) / WARP_SIZE) + 1;
			dim3 dist_grid_dim_1(n, k);
			dim3 dist_block_dim_1(min(d, WARP_SIZE));
			for (uint32_t i = 0; i < rounds; i++) {
				compute_distances_one_point_per_warp<<<dist_grid_dim_1, dist_block_dim_1>>>(d_distances_1, d_centroids, d_points, d, next_pow_2(d), i);
			}
			cudaMemcpy(h_distances_1, d_distances_1, sizeof(DATA_TYPE) * n * k,  cudaMemcpyDeviceToHost);

			DATA_TYPE* cpu_distances = new DATA_TYPE[n * k];
			for (uint32_t ni = 0; ni < n; ++ni) {
				for (uint32_t ki = 0; ki < k; ++ki) {
					DATA_TYPE dist = 0, tmp;
					for (uint32_t di = 0; di < d; ++di) {
						tmp = h_points[ni * d + di] - h_centroids[ki * d + di];
						dist += tmp * tmp;
					}
					cpu_distances[ni * k + ki] = dist;
				}
			}

			cudaDeviceSynchronize();
			const DATA_TYPE epsilon = 0.0035;
			DATA_TYPE max_diff = 0;

			for (uint32_t i = 0; i < n * k; ++i) {
				if (d <= WARP_SIZE && fabs(h_distances[i] - cpu_distances[i]) >= epsilon) {
					printf("point: %u center: %u cmp: %.6f - %.6f = %.6f\n", i / k, i % k, h_distances[i], cpu_distances[i], h_distances[i] - cpu_distances[i]);
				}
				if (fabs(h_distances_1[i] - cpu_distances[i]) >= epsilon) {
					printf("(1PointperWarp) point: %u center: %u cmp: %.6f - %.6f = %.6f\n", i / k, i % k, h_distances_1[i], cpu_distances[i], h_distances_1[i] - cpu_distances[i]);
				}
				if (d <= WARP_SIZE) {
					DATA_TYPE diff = fabs(h_distances[i] - h_distances_1[i]);
					if (diff > max_diff) max_diff = diff;
				}
			}
			for (uint32_t i = 0; i < n * k; ++i) {
				if (d <= WARP_SIZE) REQUIRE( fabs(h_distances[i] - cpu_distances[i]) < epsilon );
				REQUIRE( fabs(h_distances_1[i] - cpu_distances[i]) < epsilon );
			}

			printf("Max diff for test: %s  =	%6.8f\n", test_name, max_diff);

			delete[] h_points;
			delete[] h_centroids;
			delete[] h_distances;
			delete[] h_distances_1;
			delete[] cpu_distances;
			cudaFree(d_distances);
			cudaFree(d_distances_1);
			cudaFree(d_points);
			cudaFree(d_centroids);
		}
	}
}

TEST_CASE("kernel_argmin", "[kernel][argmin]") {
	const unsigned int TESTS_N = 8;
	const unsigned int N[TESTS_N] = {2, 10, 17, 51, 159, 1000, 3456, 10056};
	const unsigned int K[TESTS_N] = {1,  2,  7,  5, 129,	997, 1023, 1024};

	getDeviceProps(0, &deviceProps);

	for (int n_idx = 0; n_idx < TESTS_N; ++n_idx) {
		for (int k_idx = 0; k_idx < TESTS_N; ++k_idx) {
			const unsigned int n = N[n_idx];
			const unsigned int k = K[k_idx];
			const unsigned int SIZE = n * k;

			char test_name[50];
			sprintf(test_name, "kernel clusters_argmin_shfl n: %u  k: %u", n, k);
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

				uint32_t* d_clusters_len;
				cudaMalloc(&d_clusters_len, k * sizeof(uint32_t));
				cudaMemset(d_clusters_len, 0, k * sizeof(uint32_t));

				uint32_t* d_points_clusters;
				cudaMalloc(&d_points_clusters, sizeof(uint32_t) * n);

				dim3 grid_dim, block_dim;
				uint32_t sh_mem, warps_per_block;
				schedule_argmin_kernel(&deviceProps, n, k, &grid_dim, &block_dim, &warps_per_block, &sh_mem);
				clusters_argmin_shfl<<<grid_dim, block_dim, sh_mem>>>(n, k, d_distances, d_points_clusters, d_clusters_len, warps_per_block, infty, true);
				cudaDeviceSynchronize();

				uint32_t *h_points_clusters = new uint32_t[n];
				cudaMemcpy(h_points_clusters, d_points_clusters, sizeof(uint32_t) * n,	cudaMemcpyDeviceToHost);

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

					REQUIRE( h_points_clusters[i] == idx );
					if (TEST_DEBUG) { printf("%-7u -> %5u (should be %-5u %.3f)\n", i, h_points_clusters[i], idx, min); }
				}

                // Test col major
                initRandomMatrixColMaj(h_distances, n, k); 

				cudaMemcpy(d_distances, h_distances, sizeof(DATA_TYPE) * SIZE,	cudaMemcpyHostToDevice);
				cudaMemset(d_clusters_len, 0, k * sizeof(uint32_t));
                cudaMemset(d_points_clusters, 0, sizeof(uint32_t)*n);

				clusters_argmin_shfl<<<grid_dim, block_dim, sh_mem>>>(n, k, d_distances, d_points_clusters, d_clusters_len, warps_per_block, infty, false);
				cudaMemcpy(h_points_clusters, d_points_clusters, sizeof(uint32_t) * n,	cudaMemcpyDeviceToHost);


				for (uint32_t i = 0; i < n; i++) {
					DATA_TYPE min = infty;
					uint32_t idx = 0;
					for (uint32_t j = 0; j < k; j++) {
						if (TEST_DEBUG) { printf("j: %u, i: %u, v: %.0f\n", j, i, h_distances[i + j * n]); }
						if (h_distances[i + j * n] < min) {
							min = h_distances[i + j * n];
							idx = j;
						}
					}

					REQUIRE( h_points_clusters[i] == idx );
					if (TEST_DEBUG) { printf("%-7u -> %5u (should be %-5u %.3f)\n", i, h_points_clusters[i], idx, min); }
				}

				delete[] h_distances;
				delete[] h_points_clusters;
				cudaFree(d_distances);
				cudaFree(d_clusters_len);
				cudaFree(d_points_clusters);
			}
		}
	}
}

TEST_CASE("kernel_centroids", "[kernel][centroids]") {
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

				snprintf(test_name, 49, "kernel centroids d=%u n=%u k=%u", d, n, k);

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

					int rounds = ((d - 1) / WARP_SIZE) + 1;
					dim3 grid_dim, block_dim;
					schedule_centroids_kernel(&deviceProps, n, d, k, &grid_dim, &block_dim);

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

					for (int i = 0; i < rounds; i++) {
						compute_centroids_shfl<<<grid_dim, block_dim>>>(d_centroids, d_points, d_points_clusters, d_clusters_len, n, d, k, i);
					}
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
					cudaFree(d_points_clusters);
					cudaFree(d_clusters_len);

					REQUIRE(is_equal);
				}
			}
		}
	}
}



TEST_CASE("kernel_centroids_matrix", "[kernel][centroids]") {
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

				snprintf(test_name, 49, "kernel centroids matrix d=%u n=%u k=%u", d, n, k);

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

                    DATA_TYPE * d_V;
                    cudaMalloc(&d_V, k*n*sizeof(DATA_TYPE));

                    uint32_t v_mat_grid_dim = k;
                    uint32_t v_mat_block_dim = min(n, (uint32_t)deviceProps.maxThreadsPerBlock);
                    uint32_t v_rounds = std::ceil((float)n / (float)v_mat_block_dim);

                    compute_v_matrix<<<v_mat_grid_dim, v_mat_block_dim>>>
                                    (d_V, d_points_clusters,
                                     d_clusters_len,
                                     n, k,
                                     v_rounds);

                    
                    cublasHandle_t handle;
                    cublasCreate(&handle);

                    compute_centroids_gemm(handle, d, n, k,
                                            d_V, d_points,
                                            d_centroids);
                                                                            

					CHECK_CUDA_ERROR(cudaDeviceSynchronize());
					CHECK_LAST_CUDA_ERROR();
                    
					DATA_TYPE *h_centroids_cpy = new DATA_TYPE[k * d];
					CHECK_CUDA_ERROR(cudaMemcpy(h_centroids_cpy, d_centroids, k * d * sizeof(DATA_TYPE), cudaMemcpyDeviceToHost));
                    

					const DATA_TYPE EPSILON = numeric_limits<DATA_TYPE>::round_error();
					bool is_equal = true;
					for (uint32_t i = 0; i < k; ++i) {
						for (uint32_t j = 0; j < d; ++j) {
							is_equal &= fabs(h_centroids[i * d + j] - h_centroids_cpy[i + k*j]) < EPSILON;
						}
					}

					delete[] h_centroids;
					delete[] h_centroids_cpy;
					delete[] h_points;
					delete[] h_points_clusters;
					delete[] h_clusters_len;
					cudaFree(d_centroids);
					cudaFree(d_points);
					cudaFree(d_points_clusters);
					cudaFree(d_clusters_len);
                    cudaFree(d_V);

					REQUIRE(is_equal);
				}
			}
		}
	}
}



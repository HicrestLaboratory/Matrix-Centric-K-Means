#include <stdio.h>
#include <vector>
#include <random>
#include <iomanip>
#include <cmath>
#include <limits>
#include <cublas_v2.h>

#include "include/common.h"
#include "include/colors.h"

#include "utils.cuh"
#include "kmeans.cuh"

#include "kernels/kernels.cuh"

using namespace std;

random_device rd;
seed_seq seed{0}; // FIXME use rd()
mt19937 rng(seed);

const DATA_TYPE INFNTY  = numeric_limits<DATA_TYPE>::infinity();

/* Kmeans class */
void Kmeans::init_centroids (Point<DATA_TYPE>** points) {
  uniform_int_distribution<int> random_int(0, n - 1);
  if (COMPUTE_DISTANCES_KERNEL >= 2) {
    CENTROIDS_BYTES += (k * sizeof(DATA_TYPE)); // Be aware
    CHECK_CUDA_ERROR(cudaHostAlloc(&h_centroids_matrix, CENTROIDS_BYTES, cudaHostAllocDefault));
  } else {
    h_centroids_matrix = NULL;
  }
  CHECK_CUDA_ERROR(cudaHostAlloc(&h_centroids, CENTROIDS_BYTES, cudaHostAllocDefault));
  CHECK_CUDA_ERROR(cudaHostAlloc(&h_last_centroids, CENTROIDS_BYTES, cudaHostAllocDefault));
  unsigned int i = 0;
  vector<Point<DATA_TYPE>*> usedPoints;
  Point<DATA_TYPE>* centroids[k];
  while (i < k) {
    Point<DATA_TYPE>* p = points[random_int(rng)];
    bool found = false;
    for (auto p1 : usedPoints) {
      if ((*p1) == (*p)) { // FIXME Is it better use some min distance??
        found = true;
        break;
      }
    }
    if (!found) {
      centroids[i] = new Point<DATA_TYPE>(p);
      usedPoints.push_back(p);
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
      h_centroids[i * d + j] = centroids[i]->get(j);                // Row major
      #if COMPUTE_DISTANCES_KERNEL >= 2
        h_centroids_matrix[(j + 1) * k + i] = centroids[i]->get(j); // Col major
      #endif
    }
  }
  #if COMPUTE_DISTANCES_KERNEL >= 2
    for (size_t i = 0; i < k; ++i)
      h_centroids_matrix[i] = 1; // Static prefix
  #endif
  CHECK_CUDA_ERROR(cudaMalloc(&d_centroids, CENTROIDS_BYTES));
}

Kmeans::Kmeans (size_t _n, unsigned int _d, unsigned int _k, float _tol, Point<DATA_TYPE>** _points, cudaDeviceProp* _deviceProps)
    : n(_n), d(_d), k(_k), tol(_tol),
    POINTS_BYTES(_n * _d * sizeof(DATA_TYPE)),
    CENTROIDS_BYTES(_k * _d * sizeof(DATA_TYPE)),
    points(_points),
    deviceProps(_deviceProps) {

  CHECK_CUDA_ERROR(cudaHostAlloc(&h_points, POINTS_BYTES, cudaHostAllocDefault));
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < d; ++j) {
      h_points[i * d + j] = _points[i]->get(j);
    }
  }
  CHECK_CUDA_ERROR(cudaMalloc(&d_points, POINTS_BYTES));
  CHECK_CUDA_ERROR(cudaMemcpy(d_points, h_points, POINTS_BYTES, cudaMemcpyHostToDevice));

  init_centroids(_points);
  CHECK_CUDA_ERROR(cudaMemcpy(d_centroids, h_centroids, d * k * sizeof(DATA_TYPE), cudaMemcpyHostToDevice));
}

Kmeans::~Kmeans () {
  CHECK_CUDA_ERROR(cudaFreeHost(h_points));
  CHECK_CUDA_ERROR(cudaFreeHost(h_centroids));
  CHECK_CUDA_ERROR(cudaFreeHost(h_last_centroids));
  CHECK_CUDA_ERROR(cudaFreeHost(h_points_clusters));
  CHECK_CUDA_ERROR(cudaFree(d_centroids));
  CHECK_CUDA_ERROR(cudaFree(d_points));
  if (h_centroids_matrix != NULL) {
    CHECK_CUDA_ERROR(cudaFreeHost(h_centroids_matrix));
  }
  compute_gemm_distances_free();
}

uint64_t Kmeans::run (uint64_t maxiter) {
  uint64_t converged = maxiter;

  /* INIT */
  DATA_TYPE* d_distances;
  CHECK_CUDA_ERROR(cudaMalloc(&d_distances, n * k * sizeof(DATA_TYPE)));
  uint32_t* d_points_clusters;
  CHECK_CUDA_ERROR(cudaMalloc(&d_points_clusters, n * sizeof(uint32_t)));
  CHECK_CUDA_ERROR(cudaMallocHost(&h_points_clusters, n * sizeof(uint32_t)));
  uint32_t* d_clusters_len;
  CHECK_CUDA_ERROR(cudaMalloc(&d_clusters_len, k * sizeof(uint32_t)));

  uint64_t iter = 0;
  const uint32_t rounds = ((d - 1) / deviceProps->warpSize) + 1;

  #if COMPUTE_DISTANCES_KERNEL <= 1
    dim3 dist_grid_dim, dist_block_dim;
    uint32_t dist_max_points_per_warp;
    schedule_distances_kernel(deviceProps, n, d, k, &dist_grid_dim, &dist_block_dim, &dist_max_points_per_warp);
  #else
    DATA_TYPE* d_points_assoc_matrices;
    DATA_TYPE* d_centroids_matrix;
    uint32_t d1 = d + 1;
    uint32_t nd1d1 = n * d1 * d1;
    // Associated to POINTS (centers change after every iteration)
    CHECK_CUDA_ERROR(cudaMalloc(&d_points_assoc_matrices, nd1d1 * sizeof(DATA_TYPE)));
    CHECK_CUDA_ERROR(cudaMemset(d_points_assoc_matrices, 0, nd1d1 * sizeof(DATA_TYPE)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_centroids_matrix, CENTROIDS_BYTES));
    dim3 dist_assoc_matrices_grid_dim(n);
    dim3 dist_assoc_matrices_block_dim(min(next_pow_2(d), deviceProps->warpSize));
    #if COMPUTE_DISTANCES_KERNEL >= 2 && DEBUG_KERNELS_INVOKATION
      printf(YELLOW "[KERNEL]" RESET " %-25s: Grid (%4u, %4u, %4u), Block (%4u, %4u, %4u), Sh.mem. %uB\n", "compute_point_associated_matrices", dist_assoc_matrices_grid_dim.x, dist_assoc_matrices_grid_dim.y, dist_assoc_matrices_grid_dim.z, dist_assoc_matrices_block_dim.x, dist_assoc_matrices_block_dim.y, dist_assoc_matrices_block_dim.z, 0);
    #endif
    for (uint32_t i = 0; i < rounds; i++) {
      compute_point_associated_matrices<<<dist_assoc_matrices_grid_dim, dist_assoc_matrices_block_dim>>>(d_points, d_points_assoc_matrices, d, i);
    }
    cublasHandle_t cublasHandle;
    CHECK_CUBLAS_ERROR(cublasCreate(&cublasHandle));
  #endif

  dim3 argmin_grid_dim, argmin_block_dim;
  uint32_t argmin_warps_per_block, argmin_sh_mem;
  schedule_argmin_kernel(deviceProps, n, k, &argmin_grid_dim, &argmin_block_dim, &argmin_warps_per_block, &argmin_sh_mem);

  dim3 cent_grid_dim, cent_block_dim;
  schedule_centroids_kernel(deviceProps, n, d, k, &cent_grid_dim, &cent_block_dim);

  /* MAIN LOOP */
  while (iter++ < maxiter) {
    /* COMPUTE DISTANCES */

    #if COMPUTE_DISTANCES_KERNEL >= 2
      if (DEBUG_KERNELS_INVOKATION) printf(YELLOW "[KERNEL]" RESET " Matmul\n");
    #else
      if (DEBUG_KERNELS_INVOKATION) printf(YELLOW "[KERNEL]" RESET " %-25s: Grid (%4u, %4u, %4u), Block (%4u, %4u, %4u), Sh.mem. %uB\n", "compute_distances", dist_grid_dim.x, dist_grid_dim.y, dist_grid_dim.z, dist_block_dim.x, dist_block_dim.y, dist_block_dim.z, 0);
    #endif
    #if PERFORMANCES_KERNEL_DISTANCES
      cudaEvent_t e_perf_dist_start, e_perf_dist_stop;
      cudaEventCreate(&e_perf_dist_start);
      cudaEventCreate(&e_perf_dist_stop);
      cudaEventRecord(e_perf_dist_start);
    #endif
    #if COMPUTE_DISTANCES_KERNEL <= 1
      if (static_cast<int>(d) <= deviceProps->warpSize && COMPUTE_DISTANCES_KERNEL == 1) {
        compute_distances_shfl<<<dist_grid_dim, dist_block_dim>>>(d_distances, d_centroids, d_points, n, dist_max_points_per_warp, d, log2(next_pow_2(d)) > 0 ? log2(next_pow_2(d)) : 1);
      } else {
        for (uint32_t i = 0; i < rounds; i++) {
          compute_distances_one_point_per_warp<<<dist_grid_dim, dist_block_dim>>>(d_distances, d_centroids, d_points, d, next_pow_2(d), i);
        }
      }
    #else
      CHECK_CUBLAS_ERROR(cublasSetMatrix(k, d1, sizeof(DATA_TYPE), h_centroids_matrix, k, d_centroids_matrix, k)); // same as CHECK_CUDA_ERROR(cudaMemcpy(d_centroids_matrix, h_centroids_matrix, CENTROIDS_BYTES, cudaMemcpyHostToDevice));
      compute_gemm_distances(cublasHandle, d1, n, k, d_points_assoc_matrices, d_centroids_matrix, d_distances);
    #endif
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
      #if COMPUTE_DISTANCES_KERNEL >= 2
        cout << "Centroids matrix" << endl;
        printMatrixColMaj(h_centroids_matrix, k, d1);
        cout << endl;
        DATA_TYPE tmp_assoc_mat[(d + 1) * (d + 1)];
        uint32_t d1d1 = d1 * d1;
        for (size_t i = 0; i < 1; i++) {
          cout << "Point " << i << " associated matrix" << endl;
          CHECK_CUDA_ERROR(cudaMemcpy(tmp_assoc_mat, d_points_assoc_matrices + (d1d1 * i), d1d1 * sizeof(DATA_TYPE), cudaMemcpyDeviceToHost));
          printMatrixColMaj(tmp_assoc_mat, d1, d1);
          cout << endl;
        }
      #endif
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
      DATA_TYPE* tmp_dist = new DATA_TYPE[n * k];
      int anyError = 0;
      CHECK_CUDA_ERROR(cudaMemcpy(tmp_dist, d_distances, n * k * sizeof(DATA_TYPE), cudaMemcpyDeviceToHost));
      for (uint32_t i = 0; i < n; ++i)
        for (uint32_t j = 0; j < k; ++j)
          if (fabs(tmp_dist[i * k + j] - cpu_distances[i * k + j]) > 0.001) {
            printf("N=%-2u K=%-2u -> GPU=%.4f CPU=%.4f diff: %.8f\n", i, j, tmp_dist[i * k + j], cpu_distances[i * k + j], fabs(tmp_dist[i * k + j] - cpu_distances[i * k + j]));
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

    if (DEBUG_KERNELS_INVOKATION) printf(YELLOW "[KERNEL]" RESET " %-25s: Grid (%4u, %4u, %4u), Block (%4u, %4u, %4u), Sh.mem. %uB\n", "clusters_argmin_shfl", argmin_grid_dim.x, argmin_grid_dim.y, argmin_grid_dim.z, argmin_block_dim.x, argmin_block_dim.y, argmin_block_dim.z, argmin_sh_mem);

    CHECK_CUDA_ERROR(cudaMemset(d_clusters_len, 0, k * sizeof(uint32_t)));
    clusters_argmin_shfl<<<argmin_grid_dim, argmin_block_dim, argmin_sh_mem>>>(n, k, d_distances, d_points_clusters, d_clusters_len, argmin_warps_per_block, INFNTY);
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

    #if DEBUG_KERNEL_ARGMIN
      printf(GREEN "[DEBUG_KERNEL_ARGMIN]\n" RESET);
      uint32_t tmp1[n];
      CHECK_CUDA_ERROR(cudaMemcpy(tmp1, d_points_clusters, n * sizeof(uint32_t), cudaMemcpyDeviceToHost));
      printf(GREEN "p  -> c\n");
      for (uint32_t i = 0; i < n; ++i)
          printf("%-2u -> %-2u\n", i, tmp1[i]);
      cout << RESET << endl;
    #endif

    ///////////////////////////////////////////* COMPUTE NEW CENTROIDS *///////////////////////////////////////////

    CHECK_CUDA_ERROR(cudaMemset(h_centroids, 0, k * d * sizeof(DATA_TYPE)));
    CHECK_CUDA_ERROR(cudaMemset(d_centroids, 0, k * d * sizeof(DATA_TYPE)));

    #if PERFORMANCES_KERNEL_CENTROIDS
      cudaEvent_t e_perf_cent_start, e_perf_cent_stop;
      cudaEventCreate(&e_perf_cent_start);
      cudaEventCreate(&e_perf_cent_stop);
      cudaEventRecord(e_perf_cent_start);
    #endif

    if (DEBUG_KERNELS_INVOKATION) printf(YELLOW "[KERNEL]" RESET " %-25s: Grid (%4u, %4u, %4u), Block (%4u, %4u, %4u)\n", "compute_centroids", cent_grid_dim.x, cent_grid_dim.y, cent_grid_dim.z, cent_block_dim.x, cent_block_dim.y, cent_block_dim.z);

    for (uint32_t i = 0; i < rounds; i++) {
      compute_centroids_shfl<<<cent_grid_dim, cent_block_dim>>>(d_centroids, d_points, d_points_clusters, d_clusters_len, n, d, k, i);
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
      uint32_t* h_clusters_len;
      CHECK_CUDA_ERROR(cudaMallocHost(&h_clusters_len, k * sizeof(uint32_t)));
      CHECK_CUDA_ERROR(cudaMemcpy(h_points_clusters, d_points_clusters, n * sizeof(uint32_t), cudaMemcpyDeviceToHost));
      CHECK_CUDA_ERROR(cudaMemcpy(h_clusters_len,    d_clusters_len,    k * sizeof(uint32_t), cudaMemcpyDeviceToHost));
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
      CHECK_CUDA_ERROR(cudaMemcpy(h_centroids, d_centroids, d * k * sizeof(DATA_TYPE), cudaMemcpyDeviceToHost));
      cout << endl << "CENTROIDS (GPU)" << endl;
      for (uint32_t i = 0; i < k; ++i) {
        for (uint32_t j = 0; j < d; ++j)
          printf("%.3f, ", h_centroids[i * d + j]);
        cout << endl;
      }
      cout << RESET << endl;
      CHECK_CUDA_ERROR(cudaFreeHost(h_clusters_len));
    #endif

    CHECK_CUDA_ERROR(cudaMemcpy(h_centroids, d_centroids, d * k * sizeof(DATA_TYPE), cudaMemcpyDeviceToHost));

    /////////////////////////////////////////////* CHECK IF CONVERGED */////////////////////////////////////////////

    // Check exit
    if (iter > 1 && cmp_centroids()) {
      converged = iter;
      break;
    }

    // Copy current centroids
    memcpy(h_last_centroids, h_centroids, CENTROIDS_BYTES);

    #if COMPUTE_DISTANCES_KERNEL >= 2
      /* UPDATE h_centroids_matrix */
      for (size_t i = 0; i < k; ++i) {
        h_centroids_matrix[i] = 1; // Static prefix
        for (size_t j = 0; j < d; ++j) {
          h_centroids_matrix[IDX2C(i, j + 1, k)] = h_centroids[i * d + j]; // Row maj to Col maj
        }
      }
    #endif

  }
  /* MAIN LOOP END */

  /* COPY BACK RESULTS*/
  CHECK_CUDA_ERROR(cudaMemcpy(h_points_clusters, d_points_clusters, n * sizeof(uint32_t), cudaMemcpyDeviceToHost));
  for (size_t i = 0; i < n; i++) {
    points[i]->setCluster(h_points_clusters[i]);
  }

  /* FREE MEMORY */
  CHECK_CUDA_ERROR(cudaFree(d_distances));
  CHECK_CUDA_ERROR(cudaFree(d_points_clusters));
  CHECK_CUDA_ERROR(cudaFree(d_clusters_len));
  #if COMPUTE_DISTANCES_KERNEL >= 2
    CHECK_CUDA_ERROR(cudaFree(d_points_assoc_matrices));
    CHECK_CUDA_ERROR(cudaFree(d_centroids_matrix));
    CHECK_CUBLAS_ERROR(cublasDestroy(cublasHandle));
  #endif

  return converged;
}

bool Kmeans::cmp_centroids () {
  for (size_t i = 0; i < k; ++i) {
    DATA_TYPE dist_sum = 0;
    for (size_t j = 0; j < d; ++j) {
      DATA_TYPE dist = h_centroids[i * d + j] - h_last_centroids[i * d + j];
      dist_sum += dist * dist;
    }
    if (sqrt(dist_sum) > tol) { return false; }
  }

  return true;
}

void Kmeans::to_csv(ostream& o, char separator) {
  o << "cluster" << separator;
  for (size_t i = 0; i < d; ++i) {
    o << "d" << i;
    if (i != (d - 1)) o << separator;
  }
  o << endl;
  for (size_t i = 0; i < n; ++i) {
    o << h_points_clusters[i] << separator;
    for (size_t j = 0; j < d; ++j) {
      o << setprecision(8) << h_points[i * d + j];
      if (j != (d - 1)) o << separator;
    }
    o << endl;
  }
}
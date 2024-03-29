#ifndef __KMEANS__
#define __KMEANS__

#include <random>
#include "include/common.h"
#include "include/point.hpp"

/**
 * @brief
 * 0: compute_distances_one_point_per_warp
 * 1: compute_distances_shfl
 * 2: matrix multiplication
 */
#define COMPUTE_DISTANCES_KERNEL 1

class Kmeans {
  private:
    const size_t n;
    const uint32_t d, k;
    const float tol;
    const uint64_t POINTS_BYTES;
    uint64_t CENTROIDS_BYTES;
    Point<DATA_TYPE>** points;
    mt19937* generator;

    DATA_TYPE* h_points;
    DATA_TYPE* h_centroids;
    DATA_TYPE* h_last_centroids;
    DATA_TYPE* h_centroids_matrix;
    uint32_t*  h_points_clusters;
    DATA_TYPE* d_points;
    DATA_TYPE* d_centroids;

    cudaDeviceProp* deviceProps;

    /**
     * @brief Select k random centroids sampled form points
     */
    void init_centroids(Point<DATA_TYPE>** points);
    bool cmp_centroids();

  public:
    Kmeans(const size_t n, const uint32_t d, const uint32_t k, const float tol, const int *seed, Point<DATA_TYPE>** points, cudaDeviceProp* deviceProps);
    ~Kmeans();

    /**
     * @brief
     * Notice: once finished will set clusters on each of Point<DATA_TYPE> of points passed in contructor
     * @param maxiter
     * @return iter at which k-means converged
     * @return maxiter if did not converge
     */
    uint64_t run(uint64_t maxiter);
};

#endif

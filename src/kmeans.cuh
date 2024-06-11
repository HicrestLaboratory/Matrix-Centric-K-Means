#ifndef __KMEANS__
#define __KMEANS__

#include <random>
#include "include/common.h"
#include "include/point.hpp"
#include <cublas_v2.h>
#include <cub/cub.cuh>

#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/counting_iterator.h>

#include <raft/core/kvp.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/device_mdarray.hpp>

#ifdef NVTX
#include "nvToolsExt.h"
#endif

const uint32_t colors[] = { 0xff00ff00, 0xff0000ff, 0xffffff00, 0xffff00ff, 0xff00ffff, 0xffff0000, 0xffffffff };
const int num_colors = sizeof(colors)/sizeof(uint32_t);

#ifdef NVTX
#define PUSH_RANGE(name,cid) { \
        int color_id = cid; \
        color_id = color_id%num_colors;\
        nvtxEventAttributes_t eventAttrib = {0}; \
        eventAttrib.version = NVTX_VERSION; \
        eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE; \
        eventAttrib.colorType = NVTX_COLOR_ARGB; \
        eventAttrib.color = colors[color_id]; \
        eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII; \
        eventAttrib.message.ascii = name; \
        nvtxRangePushEx(&eventAttrib); \
}
#define POP_RANGE nvtxRangePop();
#endif


class Kmeans {
  public:

    enum class InitMethod
    {
        random,
        kmeans_plus_plus
    };

	template <typename IndexT, typename DataT>
	struct KeyValueIndexOp {
	  __host__ __device__ __forceinline__ IndexT
	  operator()(const raft::KeyValuePair<IndexT, DataT>& a) const
	  {
		return a.key;
	  }
	};

    Kmeans(const size_t n, const uint32_t d, const uint32_t k, const float tol, const int *seed, Point<DATA_TYPE>** points, cudaDeviceProp* deviceProps,
            InitMethod _initMethod=InitMethod::random);
    ~Kmeans();

    /**
     * @brief
     * Notice: once finished will set clusters on each of Point<DATA_TYPE> of points passed in contructor
     * @param maxiter
     * @return iter at which k-means converged
     * @return maxiter if did not converge
     */
    uint64_t run(uint64_t maxiter);

    inline float get_score() const {return score;}

  private:
    const size_t n;
    const uint32_t d, k;
    const float tol;
    const uint64_t POINTS_BYTES;
    uint64_t CENTROIDS_BYTES;
    Point<DATA_TYPE>** points;
    InitMethod initMethod;
    mt19937* generator;

    DATA_TYPE* h_points;
    DATA_TYPE* h_centroids;
    DATA_TYPE* d_new_centroids;
    DATA_TYPE* h_centroids_matrix;
    std::vector<uint32_t>  h_points_clusters;
    DATA_TYPE* d_points;
    DATA_TYPE* d_centroids;



    DATA_TYPE score;

    cudaDeviceProp* deviceProps;

    cublasHandle_t cublasHandle;

    /**
     * @brief Select k random centroids sampled form points
     */
    void init_centroids_rand(Point<DATA_TYPE>** points);
    void init_centroids_plusplus(DATA_TYPE * d_points);
    bool cmp_centroids();
    bool cmp_centroids_col_maj();



};

#endif

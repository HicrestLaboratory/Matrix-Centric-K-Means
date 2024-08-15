#ifndef __KMEANS__
#define __KMEANS__

#include "include/common.h"
#include "include/point.hpp"
#include "kernels/kernels.cuh"

#include <random>

#include <cublas_v2.h>
#include <cub/cub.cuh>

#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/transform_iterator.h>

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
        plus_plus
    };

    enum class Kernel 
    {
        linear,
        polynomial,
        sigmoid,
    };

	template <typename IndexT, typename DataT>
	struct KeyValueIndexOp {
     
	  __host__ __device__ __forceinline__ IndexT
	  operator()(const raft::KeyValuePair<IndexT, DataT>& a) const
	  {
		return a.key ;
	  }
	};

    //TODO: Remove unnecessary long longs
    struct PermuteRowOp : public thrust::unary_function<unsigned long long, unsigned long long>
    {
        PermuteRowOp(const unsigned long long _rows,
                     const unsigned long long _cols,
                     uint32_t * _d_perm) :
            rows(_rows), cols(_cols), d_perm(_d_perm) 
        {}

        __host__ __device__
        unsigned long long operator()(unsigned long long idx)
        {
            unsigned long long i = idx / cols;
            unsigned long long new_i = (unsigned long long)d_perm[i];
            unsigned long long j = idx % cols;
            return (new_i*cols) + j;
        }
            

        unsigned long long rows;
        unsigned long long cols;
        uint32_t * d_perm;

    };

    struct is_nonzero
    {
        __host__ __device__
        bool operator()(uint32_t a) 
        {
            return a != 0;
        }
    };

    struct check_not_equals
    {
        __host__ __device__
        uint32_t operator()(uint32_t a, uint32_t b)
        {
            return static_cast<uint32_t>(a!=b);
        }
    };

    Kmeans(const size_t n, const uint32_t d, const uint32_t k, const float tol, const int *seed, Point<DATA_TYPE>** points, cudaDeviceProp* deviceProps,
            InitMethod _initMethod=InitMethod::random,
            Kernel _kernel=Kernel::linear,
            int _level=3);
    ~Kmeans();

    /**
     * @brief
     * Notice: once finished will set clusters on each of Point<DATA_TYPE> of points passed in contructor
     * @param maxiter
     * @return iter at which k-means converged
     * @return maxiter if did not converge
     */
    uint64_t run(uint64_t maxiter, bool check_converged);

    template <typename ClusterIter>
    void set_perm_vec(ClusterIter clusters,
                      uint32_t * d_cluster_offsets);

    void permute_kernel_mat();
    void permute_kernel_mat_swap(thrust::device_vector<uint32_t> d_indices);

    inline float get_score() const {return score;}

  private:
    const size_t n;
    const uint32_t d, k;
    const float tol;
    const uint64_t POINTS_BYTES;
    uint64_t CENTROIDS_BYTES;
    Point<DATA_TYPE>** points;
    InitMethod initMethod;
    int level;
    mt19937* generator;

    bool do_reorder;

    DATA_TYPE* h_points;
    DATA_TYPE* h_centroids;
    DATA_TYPE* d_new_centroids;
    DATA_TYPE* h_centroids_matrix;
    std::vector<uint32_t>  h_points_clusters;
    DATA_TYPE* d_points;
    DATA_TYPE* d_centroids;
    DATA_TYPE* d_centroids_row_norms;
    DATA_TYPE* d_z_vals;
    int32_t * d_clusters;
    uint32_t * d_clusters_len;

    uint32_t * d_perm_vec;
    uint32_t * d_perm_vec_prev;

    DATA_TYPE * d_B;
    DATA_TYPE * d_B_new;

    DATA_TYPE * d_C;

    DATA_TYPE * d_V_vals;
    int32_t * d_V_colinds;
    int32_t * d_V_rowptrs;

    DATA_TYPE * d_F_vals;
    int32_t * d_F_colinds;
    int32_t * d_F_row_offsets;

    cusparseDnMatDescr_t P_descr;
    cusparseDnMatDescr_t B_descr;
    cusparseDnMatDescr_t D_descr;
    cusparseDnMatDescr_t C_descr;
    cusparseDnVecDescr_t c_tilde_descr;
    cusparseDnVecDescr_t z_descr;
    cusparseSpMatDescr_t V_descr;
    cusparseSpMatDescr_t F_descr;


    DATA_TYPE score;
    DATA_TYPE last_score;

    cudaDeviceProp* deviceProps;

    cublasHandle_t cublasHandle;
    cusparseHandle_t cusparseHandle;

    /**
     * @brief Select k random centroids sampled form points
     */
    void init_centroids_rand();
    void init_centroids_plus_plus();
    bool cmp_centroids();
    bool cmp_centroids_col_maj();



};

#endif

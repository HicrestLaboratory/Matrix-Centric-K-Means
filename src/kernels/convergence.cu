
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>

#include "kernels.cuh"
#include "../cuda_utils.cuh"
#include "../kmeans.cuh"


/** 
 * One block per centroid
 * warp_size * ceil(d / warp_size) threads per block 
 * Reduce in each warp to shared memory array
 * then reduce in shared mem to get final result in thread 0 of each
 */
__global__ void check_convergence( const DATA_TYPE * d_centroids,
                                    const DATA_TYPE * d_last_centroids,
                                    const uint32_t d,
                                    const uint32_t k,
                                    const uint32_t next_pow_of2,
                                    const DATA_TYPE tol,
                                    bool is_row_maj,
                                    int * result)
{

    *result = 0;

    const uint32_t tid = threadIdx.x;

    uint32_t centroid_idx;

    if (is_row_maj)
        centroid_idx = blockIdx.x * d + tid;
    else
        centroid_idx = blockIdx.x + tid * k; 


    const uint32_t warp_size = 32;
    const uint32_t wid = threadIdx.x / warp_size; //TODO: warp size param
    const uint32_t idx_in_warp = threadIdx.x % warp_size;
    const uint32_t n_warps = next_pow_of2 / warp_size;

    extern __shared__ DATA_TYPE warp_results[];
    if (tid < d) {
        DATA_TYPE dist = ( d_centroids[centroid_idx] - 
                        d_last_centroids[centroid_idx] );

        dist *= dist;

		for (int i = warp_size >> 1; i > 0; i >>= 1) {
			dist += __shfl_down_sync(DISTANCES_SHFL_MASK, dist, i);
		}

        if (idx_in_warp==0) {
            warp_results[wid] = dist;
        }

    }



    __syncthreads();

    for (int i = n_warps >> 1; i > 0; i >>= 1) {
        if (tid<(i)) {
            warp_results[tid] += warp_results[tid + i];
        }
    }


    if (tid==0) {
        int loc_converged = (int)(warp_results[tid] <= tol);
        *result =  *result || loc_converged;
    }

        
}

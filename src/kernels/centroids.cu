#include "kernels.cuh"
#include "../utils.cuh"

__global__ void compute_centroids_shfl(DATA_TYPE* centroids, const DATA_TYPE* points, const uint32_t* points_clusters, const uint32_t* clusters_len, const uint64_t n, const uint32_t d, const uint32_t k, const uint32_t round) { 
  const uint32_t block_base = warpSize * round;  // Get in which block of d the kernel works (0 <= d < 32 => block_base = 0; 32 <= d < 64 => block_base = 32; ...)
  if (block_base + threadIdx.y >= d) { return; } // threadIdx.y represents the dim; if the thread is responsible for a dim >= d, then return to avoid illegal writes
  
  const uint32_t cluster_idx   = 2 * blockIdx.y * blockDim.x + threadIdx.x;  // Index of the cluster assignment for the current point 
  const uint32_t point_idx     = block_base + cluster_idx * d + threadIdx.y; // Index of the dim for the current point
  const uint32_t cluster_off   = blockDim.x;                                 // Offset for the cluster assignment
  const uint32_t point_off     = cluster_off * d;                            // Offset for the dim for the current point
  const uint32_t centroids_idx = block_base + blockIdx.x * d + threadIdx.y;  // Index of the current dim in the centroid matrix
  
  DATA_TYPE val = 0;

  // If the point is in the matrix of points and the block is responsible of the cluster assigned, then get the value
  if (point_idx < n * d && blockIdx.x == points_clusters[cluster_idx]) { 
    val = points[point_idx]; 
  }
  
  // If the point with offset is in the matrix of points and the block is responsible of the cluster assigned, then get the value
  if (point_idx + point_off < n * d && blockIdx.x == points_clusters[cluster_idx + cluster_off]) { 
    val += points[point_idx + point_off]; 
  } 
  
  // Perform warp reduction
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    val += __shfl_down_sync(CENTROIDS_SHFL_MASK, val, offset);
  }
  
  // The first thread writes atomically the scaled sum in the centroids matrix
  if (threadIdx.x % warpSize == 0 && centroids_idx < k * d) {
    uint32_t count = clusters_len[blockIdx.x] > 1 ? clusters_len[blockIdx.x] : 1; 
    DATA_TYPE scale = 1.0 / ((double) count); 
    val *= scale;
    atomicAdd(&centroids[centroids_idx], val);
  }
}

__global__ void compute_centroids_shfl_shrd(DATA_TYPE* centroids, const DATA_TYPE* points, const uint32_t* points_clusters, const uint32_t* clusters_len, const uint64_t n, const uint32_t d, const uint32_t k, const uint32_t round){
  const uint32_t block_base = warpSize * round;
  if (block_base + threadIdx.y >= d) { return; } 

  const uint32_t cluster_idx   = 2 * blockIdx.y * blockDim.x + threadIdx.x;
  const uint32_t point_idx     = block_base + cluster_idx * d + threadIdx.y;
  const uint32_t cluster_off   = blockDim.x;
  const uint32_t point_off     = cluster_off * d;
  const uint32_t centroids_idx = block_base + blockIdx.x * d + threadIdx.y;
  
  DATA_TYPE val = 0;
  extern __shared__ DATA_TYPE shrd_mem[];

  if (point_idx < n * d && blockIdx.x == points_clusters[cluster_idx]) { 
    val = points[point_idx]; 
  }
  
  if (point_idx + point_off < n * d && blockIdx.x == points_clusters[cluster_idx + cluster_off]) { 
    val += points[point_idx + point_off]; 
  }
  
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    val += __shfl_down_sync(CENTROIDS_SHFL_MASK, val, offset);
  }
  
  // To avoid some atomicAdd perform reduction in shared memory before writing in the centroids matrix
  if (threadIdx.x % warpSize == 0 && centroids_idx < k * d) {
    uint32_t warp_idx   = threadIdx.x / warpSize;
    uint32_t shrd_dim_y = blockDim.x  / warpSize;
    uint32_t shrd_idx   = threadIdx.y * shrd_dim_y + warp_idx;
    
    shrd_mem[shrd_idx] = val;
    __syncthreads();
    
    for (int offset = shrd_dim_y / 2; offset > 0; offset /= 2) {
      if (warp_idx < offset) {
        shrd_mem[shrd_idx] += shrd_mem[shrd_idx + offset];
      }
      __syncthreads();
    }

    if (shrd_idx % shrd_dim_y == 0) {
      uint32_t count = clusters_len[blockIdx.x] > 1 ? clusters_len[blockIdx.x] : 1; 
      DATA_TYPE scale = 1.0 / ((double) count); 
      val = shrd_mem[shrd_idx] * scale;   
      atomicAdd(&centroids[centroids_idx], val);
    }
  }
}

void schedule_centroids_kernel(const cudaDeviceProp *props, const uint32_t n, const uint32_t d, const uint32_t k, dim3 *grid, dim3 *block) {
  dim3 cent_grid_dim(k);
  dim3 cent_block_dim(max(next_pow_2((n + 1) / 2), props->warpSize), min(props->warpSize, d)); 
  int cent_threads_tot = cent_block_dim.x * cent_block_dim.y;
  
  while (cent_threads_tot > props->maxThreadsPerBlock) {
    cent_block_dim.x /= 2;
    cent_grid_dim.y *= 2;
    cent_threads_tot = cent_block_dim.x * cent_block_dim.y;
  } 

  *grid  = cent_grid_dim;
  *block = cent_block_dim;
}
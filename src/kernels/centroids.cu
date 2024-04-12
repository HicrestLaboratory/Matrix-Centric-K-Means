#include "kernels.cuh"
#include "../cuda_utils.cuh"

__global__ void compute_centroids_shfl(DATA_TYPE* centroids, const DATA_TYPE* points, 
                                       const uint32_t* points_clusters, 
                                       const uint32_t* clusters_len, 
                                       const uint64_t n, const uint32_t d, 
                                       const uint32_t k, const uint32_t round) {
	const uint32_t block_base = round * warpSize; // Get in which block of d the kernel works (0 <= d < 32 => block_base = 0; 32 <= d < 64 => block_base = 32; ...)
	if (block_base + threadIdx.y >= d) { return; } // threadIdx.y represents the dim; if the thread is responsible for a dim >= d, then return to avoid illegal writes

	const uint32_t cluster_idx = ((blockIdx.y * blockDim.x) << 1) + threadIdx.x; // Index of the cluster assignment for the current point
	const uint32_t point_idx = block_base + cluster_idx * d + threadIdx.y; //Index of the dim for the current point
	const uint32_t cluster_off = blockDim.x; // Offset for the cluster assignment
	const uint32_t point_off = cluster_off * d; // Offset for the dim for the current point
	const uint32_t centroids_idx = block_base + blockIdx.x * d + threadIdx.y; // Index of the current dim in the centroid matrix
	const uint32_t nd = n * d;
	const uint32_t kd = k * d;

	DATA_TYPE val = 0;

	// If the point is in the matrix of points and the block is responsible of the cluster assigned, then get the value
	if (point_idx < nd && blockIdx.x == points_clusters[cluster_idx]) {
		val = points[point_idx];
	}

	// If the point with offset is in the matrix of points and the block is responsible of the cluster assigned, then get the value
	if (point_idx + point_off < nd && blockIdx.x == points_clusters[cluster_idx + cluster_off]) {
		val += points[point_idx + point_off];
	}

	// Perform warp reduction
	for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
		val += __shfl_down_sync(CENTROIDS_SHFL_MASK, val, offset);
	}

	// The first thread writes atomically the scaled sum in the centroids matrix
	if ((threadIdx.x & (warpSize - 1)) == 0 && val != 0.0 && centroids_idx < kd) {
		uint32_t count = clusters_len[blockIdx.x] > 1 ? clusters_len[blockIdx.x] : 1;
		DATA_TYPE scale = 1.0 / ((double) count);
		val *= scale;
		atomicAdd(&centroids[centroids_idx], val);
	}
}

__global__ void compute_centroids_shfl_shrd(DATA_TYPE* centroids, const DATA_TYPE* points, const uint32_t* points_clusters, const uint32_t* clusters_len, const uint64_t n, const uint32_t d, const uint32_t k, const uint32_t round){
	const uint32_t block_base = warpSize * round;
	if (block_base + threadIdx.y >= d) { return; }

	const uint32_t cluster_idx	 = 2 * blockIdx.y * blockDim.x + threadIdx.x;
	const uint32_t point_idx		 = block_base + cluster_idx * d + threadIdx.y;
	const uint32_t cluster_off	 = blockDim.x;
	const uint32_t point_off		 = cluster_off * d;
	const uint32_t centroids_idx = block_base + blockIdx.x * d + threadIdx.y;

	DATA_TYPE val = 0;
	extern __shared__ DATA_TYPE shrd_mem[];

	if (point_idx < n * d && blockIdx.x == points_clusters[cluster_idx]) {
		val = points[point_idx];
	}

	if (point_idx + point_off < n * d && blockIdx.x == points_clusters[cluster_idx + cluster_off]) {
		val += points[point_idx + point_off];
	}

	for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
		val += __shfl_down_sync(CENTROIDS_SHFL_MASK, val, offset);
	}

	// To avoid some atomicAdd perform reduction in shared memory before writing in the centroids matrix
	if (threadIdx.x % warpSize == 0 && centroids_idx < k * d) {
		uint32_t warp_idx		= threadIdx.x / warpSize;
		uint32_t shrd_dim_y = blockDim.x	/ warpSize;
		uint32_t shrd_idx		= threadIdx.y * shrd_dim_y + warp_idx;

		shrd_mem[shrd_idx] = val;
		__syncthreads();

		for (int offset = shrd_dim_y >> 1; offset > 0; offset >>= 1) {
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


// k blocks, rounds should be ceil(n/blockDim)
__global__ void compute_v_matrix(DATA_TYPE * d_V,
                                 const uint32_t * d_points_clusters,
                                 const uint32_t * d_clusters_len,
                                 const uint32_t n, const uint32_t k,
                                 const uint32_t rounds)
{
    const uint32_t cluster_idx = blockIdx.x;
    for (uint32_t round=0; round<rounds; round++) {
        const uint32_t point_idx = round * blockDim.x + threadIdx.x;
        if (point_idx < n) {
            const uint32_t this_cluster_idx = d_points_clusters[point_idx];
            DATA_TYPE val = (cluster_idx == this_cluster_idx) ?
                                ((DATA_TYPE) 1) / ((DATA_TYPE) d_clusters_len[cluster_idx]) : 0;
            d_V[cluster_idx + k*point_idx] = val; //col major
        }
    }
}




void compute_centroids_gemm(cublasHandle_t& handle,
                            const uint32_t d, const uint32_t n, const uint32_t k,
                            const DATA_TYPE * d_V, const DATA_TYPE * d_points,
                            DATA_TYPE * d_centroids)
{
    const DATA_TYPE alpha = 1.0;
    const DATA_TYPE beta = 0.0;
    CHECK_CUBLAS_ERROR(cublasSgemm(handle,
                                    CUBLAS_OP_N, CUBLAS_OP_T, 
                                    k, d, n,
                                    &alpha,
                                    d_V, k,
                                    d_points, d,
                                    &beta,
                                    d_centroids, k));
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}


//GLOBAL TODO: Handle case where we can't allocate enough blocks to make this work
__global__ void compute_v_sparse(DATA_TYPE * d_vals,
                                 int32_t * d_rowinds,
                                 int32_t * d_col_offsets,
                                 const uint32_t * d_points_clusters,
                                 const uint32_t * d_clusters_len,
                                 const uint32_t n) 
{
    const uint32_t tid = threadIdx.x + blockDim.x * blockIdx.x; 
    if (tid<n) {
        d_vals[tid] = ((DATA_TYPE) 1) / (DATA_TYPE)(d_clusters_len[d_points_clusters[tid]]);
        d_rowinds[tid] = d_points_clusters[tid];
        d_col_offsets[tid] = tid;
    }
    d_col_offsets[n] = n; //This might be horrible
}


void compute_centroids_spmm(cusparseHandle_t& handle,
                            const uint32_t d, const uint32_t n, const uint32_t k,
                            const DATA_TYPE * d_V_vals,
                            const int32_t * d_V_rowinds,
                            const int32_t * d_V_col_offsets,
                            DATA_TYPE * d_centroids,
                            cusparseSpMatDescr_t& V_descr,
                            cusparseDnMatDescr_t& P_descr,
                            cusparseDnMatDescr_t& C_descr)
{


    CHECK_CUSPARSE_ERROR(cusparseCscSetPointers(V_descr, (void*)d_V_col_offsets, (void*)d_V_rowinds, (void*)d_V_vals));
    CHECK_CUSPARSE_ERROR(cusparseDnMatSetValues(C_descr, (void*)d_centroids));

    
    const DATA_TYPE alpha = 1.0;
    const DATA_TYPE beta = 0.0;
    
    size_t buff_size = 0;

    CHECK_CUSPARSE_ERROR(cusparseSpMM_bufferSize(handle,
                                                  CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                  CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                  &alpha,
                                                  V_descr,
                                                  P_descr,
                                                  &beta,
                                                  C_descr,
                                                  CUDA_R_32F,
                                                  CUSPARSE_SPMM_ALG_DEFAULT, //TODO: Play with this more
                                                  &buff_size));
    
    void * d_buff;
    CHECK_CUDA_ERROR(cudaMalloc(&d_buff, buff_size));

    CHECK_CUSPARSE_ERROR(cusparseSpMM(handle,
                                      CUSPARSE_OPERATION_NON_TRANSPOSE,
                                      CUSPARSE_OPERATION_NON_TRANSPOSE,
                                      &alpha,
                                      V_descr,
                                      P_descr,
                                      &beta,
                                      C_descr,
                                      CUDA_R_32F,
                                      CUSPARSE_SPMM_ALG_DEFAULT, //TODO: Play with this more
                                      d_buff));

    CHECK_CUSPARSE_ERROR(cusparseDnMatGetValues(C_descr, (void**)&d_centroids));

    //TODO: Since we're allowed to have the output matrix in row-major form, we should probably change the stuff in
    // kmeans.cu to store d_centroids in row major form

}







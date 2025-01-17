#ifndef __COMMON__
#define __COMMON__

#include <cassert>

#define DEBUG_DEVICE 0

#define DEBUG_INPUT_DATA 0
#define DEBUG_INIT_CENTROIDS 0

#define DEBUG_KERNELS_INVOKATION 0

#define DEBUG_KERNEL_DISTANCES 0
#define DEBUG_KERNEL_ARGMIN 0
#define DEBUG_KERNEL_CENTROIDS 0 
#define DEBUG_PRUNING 0

#define COUNT_STATIONARY_CLUSTERS 1

#define PERFORMANCES_KERNEL_DISTANCES 1
#define PERFORMANCES_KERNEL_ARGMIN 1
#define PERFORMANCES_KERNEL_CENTROIDS 1
#define PERFORMANCES_MEMCPY 1
#define PERFORMANCES_CENTROIDS_INIT 1
#define PERFORMANCES_BMULT 1

#define PRUNE_CENTROIDS 0
#define USE_RAFT 0

#define PROFILE_MEMORY 1

#define DEBUG_OUTPUT_INFO 1

#define DATA_TYPE float 

#define NAIVE_GPU 0
#define NAIVE_MTX 1
#define OPT_MTX 2
#define REORDER 3
#define FINAL 4

#define GEMM_THRESHOLD 100

//#define NVTX

#endif

#ifndef __KERNEL__FUNCTIONS__
#define __KERNEL__FUNCTIONS__
#include "../include/common.h"

#include <cublas_v2.h>
#include <cusparse.h>

/* CUDA compute kernels */
__global__ static void sigmoid(const uint32_t n,
                               DATA_TYPE * d_B,
                               const DATA_TYPE gamma,
                               const DATA_TYPE coef)
{
    const uint32_t tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid < n*n) {
        d_B[tid] = tanhf(d_B[tid]*gamma + coef);
    }
}


/* Kernel structs */

struct NullKernel {
    static void function(const uint32_t n,
                         const uint32_t d,
                         DATA_TYPE * d_B)
    {}
};


struct SigmoidKernel {
    static void function(const uint32_t n,
                         const uint32_t d,
                         DATA_TYPE * d_B)
    {

        const uint32_t max_tpb = 1024;

        const DATA_TYPE gamma = 1.0 / static_cast<float>(d);
        const DATA_TYPE coef = 1;

        const uint32_t tpb = std::min(n*n, max_tpb);
        const uint32_t blocks = std::ceil( static_cast<float>(n*n) / static_cast<float>(tpb) );

        sigmoid<<<blocks, tpb>>>(n, d_B, gamma, coef);

    }

};



#endif

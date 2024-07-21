#ifndef __KERNEL__FUNCTIONS__
#define __KERNEL__FUNCTIONS__
#include "../include/common.h"
#include "../cuda_utils.cuh"

#include <cublas_v2.h>
#include <cusparse.h>

namespace kernels {




/* Compute kernels */
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


__global__ static void polynomial(const uint32_t n,
                                   DATA_TYPE * d_B,
                                   const DATA_TYPE gamma,
                                   const DATA_TYPE coef,
                                   const uint32_t deg)
{
    const uint32_t tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid < n*n) {
        d_B[tid] = pow(d_B[tid]*gamma + coef, deg);
    }
}




/* Kernel structs */

struct LinearKernel {
    static void function(const uint32_t n,
                         const uint32_t d,
                         DATA_TYPE * d_B)
    {/* Do nothing, since matmul already applied this kernel */}
};


struct SigmoidKernel 
{

    static std::pair<uint32_t, uint32_t> get_grid_params(const uint32_t n) 
    {
        const uint32_t max_tpb = 1024;
        const uint32_t tpb = std::min(n*n, max_tpb);
        const uint32_t blocks = std::ceil( static_cast<float>(n*n) / static_cast<float>(tpb) );
        return {blocks, tpb};
    }

    static void function(const uint32_t n,
                         const uint32_t d,
                         DATA_TYPE * d_B)
    {


        const DATA_TYPE gamma = 1.0 / static_cast<float>(d);
        const DATA_TYPE coef = 1;

        auto params = get_grid_params(n);

        sigmoid<<<params.first, params.second>>>(n, d_B, gamma, coef);

    }

};


struct PolynomialKernel 
{
    static std::pair<uint32_t, uint32_t> get_grid_params(const uint32_t n) 
    {
        const uint32_t max_tpb = 1024;
        const uint32_t tpb = std::min(n*n, max_tpb);
        const uint32_t blocks = std::ceil( static_cast<float>(n*n) / static_cast<float>(tpb) );
        return {blocks, tpb};
    }

    static void function(const uint32_t n,
                         const uint32_t d,
                         DATA_TYPE * d_B)
    {


        const DATA_TYPE gamma = 1.0 / static_cast<float>(d);
        const DATA_TYPE coef = 1;
        const uint32_t deg = 3;

        auto params = get_grid_params(n);

        polynomial<<<params.first, params.second>>>(n, d_B, gamma, coef, deg);

    }

};
}

#endif

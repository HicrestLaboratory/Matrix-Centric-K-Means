#ifndef __KERNEL__FUNCTIONS__
#define __KERNEL__FUNCTIONS__
#include "../include/common.h"

#include <cublas_v2.h>
#include <cusparse.h>


struct NullKernel {

    static void function(const uint32_t n,
                  DATA_TYPE * d_N)
    {}

};



#endif

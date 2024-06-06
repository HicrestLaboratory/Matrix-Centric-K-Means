set(CMAKE_CXX_COMPILER CC)
set(CMAKE_CXX_STANDARD 17)
cmake_minimum_required(VERSION 3.24.3)

# ------------- configure rapids-cmake --------------#

include(cmake/thirdparty/fetch_rapids.cmake)
include(rapids-cmake)
include(rapids-cpm)
include(rapids-cuda)
include(rapids-export)
include(rapids-find)



# ------------- configure project --------------#

rapids_cuda_init_architectures(GPU_Kmeans)

project(GPU_Kmeans VERSION 1.0.0 LANGUAGES CXX CUDA)


# ------------- configure raft -----------------#

rapids_cpm_init()
include(cmake/thirdparty/get_raft.cmake)


#find_package(CUDAToolkit REQUIRED)


add_library(utils STATIC src/cuda_utils.cu src/cuda_utils.cuh)
#set_target_properties(utils PROPERTIES CUDA_ARCHITECTURES "80") # this is to make CMake happy
#set_target_properties(utils PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
#set_target_properties(utils PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "bin")


add_subdirectory(src)
add_subdirectory(tests EXCLUDE_FROM_ALL)

add_executable(
  gpukmeans
  ./kernels/kernels.cuh
  ./kernels/argmin.cu
  ./kernels/centroids.cu
  ./kernels/distances.cu
  ./kernels/convergence.cu
  main.cu
  kmeans.cu
  kmeans.cuh
  ./include/common.h
  ./include/utils.hpp
  )


target_link_libraries(gpukmeans PRIVATE -lcublas -lgomp -lcusparse -L"/opt/nvidia/hpc_sdk/Linux_x86_64/23.9/math_libs/lib64/")
target_link_libraries(gpukmeans PRIVATE utils)
#target_include_directories(gpukmeans PUBLIC /opt/nvidia/hpc_sdk/Linux_x86_64/23.9/cuda/12.2/include )




target_link_libraries(gpukmeans PRIVATE raft::raft raft::compiled)


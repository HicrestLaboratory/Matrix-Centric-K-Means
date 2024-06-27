#include <chrono>

#include "include/common.h"
#include "include/colors.h"
#include "include/input_parser.hpp"
#include "include/utils.hpp"

#include "kmeans.cuh"
#include "cuda_utils.cuh"

#define DEVICE 0

using namespace std;

int main(int argc, char **argv) {
  uint32_t d, k, runs;
  size_t   n, maxiter;
  string   out_file;
  float    tol;
  int     *seed = NULL;
  InputParser<float> *input = NULL;
  bool check_converged;

  parse_input_args(argc, argv, 
                    &d, &n, &k, 
                    &maxiter, out_file, 
                    &tol, &runs, &seed, &input,
                    &check_converged);

  #if DEBUG_INPUT_DATA
    cout << "Points" << endl << *input << endl;
  #endif

  // Check devices
  int deviceCount = 0;
  cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

  if (error_id != cudaSuccess) {
    printf("cudaGetDeviceCount returned %d\n-> %s\n", static_cast<int>(error_id), cudaGetErrorString(error_id));
    exit(EXIT_FAILURE);
  }
  if (deviceCount == 0) {
    printErrDesc(EXIT_CUDA_DEV);
    exit(EXIT_CUDA_DEV);
  } else if (DEBUG_DEVICE) {
    printf("Detected %d CUDA Capable device(s)\n", deviceCount);
  }

  cudaDeviceProp deviceProp;
  getDeviceProps(DEVICE, &deviceProp);
  if (DEBUG_DEVICE) describeDevice(DEVICE, deviceProp);

  printf(BOLDBLUE);
  double tot_time = 0;
  double init_time = 0;
  double score = 0;


  for (uint32_t i = 0; i < runs; i++) {
    const auto init_start = chrono::high_resolution_clock::now();
    Kmeans kmeans(n, d, k, tol, seed, input->get_dataset(), &deviceProp,
                    Kmeans::InitMethod::random);
    const auto init_end = chrono::high_resolution_clock::now();

    if (i>0)
        init_time += (chrono::duration_cast<chrono::duration<double>>(init_end - init_start)).count();

    const auto start = chrono::high_resolution_clock::now();
    uint64_t converged = kmeans.run(maxiter, check_converged);
    const auto end = chrono::high_resolution_clock::now();


    const auto duration = chrono::duration_cast<chrono::duration<double>>(end - start);
    if (i>0) // First iteration is just a warmup
        tot_time += duration.count();

    #if DEBUG_OUTPUT_INFO
      if (converged < maxiter)
        printf("K-means converged at iteration %lu - ", converged);
      else
        printf("K-means did NOT converge - ");
      printf("Time: %lf\n", duration.count());
      printf("Objective score: %lf\n", kmeans.get_score());
    #endif
    score += kmeans.get_score();
  }

  printf("GPU_Kmeans: %lfs (%u runs)\n", tot_time / (runs-1), runs);
  printf("Init_time: %lfs \n", init_time / (runs-1));
  printf("Score: %lf\n", score / runs);
  printf(RESET);

  if (strcmp(out_file.c_str(), "None")!=0) {
      ofstream fout(out_file);
      input->dataset_to_csv(fout);
      fout.close();
  }
  delete seed;

  return 0;
}

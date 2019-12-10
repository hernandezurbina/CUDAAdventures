#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "cuda.h"
#include "common.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void gpuRecursiveRedux(int *g_idata, int *g_odata, unsigned int isize){
  int tid = threadIdx.x;
  int *idata = g_idata + blockIdx.x * blockDim.x;
  int *odata = &g_odata[blockIdx.x];

  if(isize == 2 && tid == 0){
    g_odata[blockIdx.x] = idata[0] + idata[1];
    return;
  }

  int istride = isize >> 1;
  if(istride > 1 && tid < istride) {
    idata[tid] += idata[tid + istride];
  }

  __syncthreads();

  if(tid == 0){
    gpuRecursiveRedux<<<1, istride>>>(idata, odata, istride);
    cudaDeviceSynchronize();
  }
  __syncthreads();
}

int main(){
  int size = 1 << 27;
  int byte_size = size * sizeof(int);

  int block_size = 128;

  int *h_input, *h_ref;
  h_input = (int *) malloc(byte_size);

  initialize(h_input, size, INIT_RANDOM);

  // get reduction result from CPU
  int cpu_result = reduction_cpu(h_input, size);

  dim3 block(block_size);
  dim3 grid(size/block.x);

  printf("Kernel launch params:\n grid.x: %d, block.x: %d\n", grid.x, block.x);

  int temp_array_byte_size = sizeof(int) * grid.x;
  h_ref = (int *) malloc(temp_array_byte_size);

  int *d_input, *d_temp;

  cudaMalloc((void **)&d_input, byte_size);
  cudaMalloc((void **)&d_temp, temp_array_byte_size);

  cudaMemset(d_temp, 0, temp_array_byte_size);
  cudaMemcpy(d_input, h_input, byte_size, cudaMemcpyHostToDevice);

  gpuRecursiveRedux<<<grid, block>>>(d_input, d_temp, size);

  cudaDeviceSynchronize();
  cudaMemcpy(h_ref, d_temp, temp_array_byte_size, cudaMemcpyDeviceToHost);

  int gpu_result = 0;
  for(int i = 0; i < grid.x; i++) {
    gpu_result += h_ref[i];
  }

  compare_results(gpu_result, cpu_result);

  free(h_ref);
  free(h_input);

  cudaFree(d_temp);
  cudaFree(d_input);

  cudaDeviceReset();
  return 0;

}

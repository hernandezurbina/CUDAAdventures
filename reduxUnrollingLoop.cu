#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "common.h"

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void reduxUnrollingBlocks2(int *input, int *temp, int size){
  int tid = threadIdx.x;
  int BLOCK_OFFSET = blockIdx.x * blockDim.x * 2;
  int index = BLOCK_OFFSET + tid;
  int *i_data = input + BLOCK_OFFSET;

  if((index + blockDim.x) < size) {
    input[index] += input[index + blockDim.x];
  }
  __syncthreads();

  for(int offset = blockDim.x/2; offset > 0; offset = offset /2){
    if(tid < offset) {
      temp[blockIdx.x] = i_data[0];
    }
    __syncthreads();
  }
  if(tid == 0) {
    temp[blockIdx.x] = i_data[0];
  }

}

__global__ void reduxUnrollingBlocks4(int *input, int *temp, int size){
  int tid = threadIdx.x;
  int BLOCK_OFFSET = blockIdx.x * blockDim.x * 4;
  int index = BLOCK_OFFSET + tid;
  int *i_data = input + BLOCK_OFFSET;

  if((index + 3 * blockDim.x) < size) {
    int a1 = input[index];
    int a2 = input[index + blockDim.x];
    int a3 = input[index + 2 * blockDim.x];
    int a4 = input[index + 3 * blockDim.x];
    input[index] = a1 + a2 + a3 + a4;
  }
  __syncthreads();

  for(int offset = blockDim.x/2; offset > 0; offset = offset /2){
    if(tid < offset) {
      temp[blockIdx.x] = i_data[0];
    }
    __syncthreads();
  }
  if(tid == 0) {
    temp[blockIdx.x] = i_data[0];
  }

}

int main(int argc, char** argv){
  int size = 1 << 27;
  int byte_size = size * sizeof(int);

  int block_size = 128;

  int *h_input, *h_ref;
  h_input = (int *) malloc(byte_size);

  initialize(h_input, size, INIT_RANDOM);

  // get reduction result from CPU
  int cpu_result = reduction_cpu(h_input, size);

  dim3 block(block_size);
  dim3 grid((size / block_size)/2);

  printf("Kernel launch params:\n grid.x: %d, block.x: %d\n", grid.x, block.x);

  int temp_array_byte_size = sizeof(int) * grid.x;
  h_ref = (int *) malloc(temp_array_byte_size);

  int *d_input, *d_temp;

  cudaMalloc((void **)&d_input, byte_size);
  cudaMalloc((void **)&d_temp, temp_array_byte_size);

  cudaMemset(d_temp, 0, temp_array_byte_size);
  cudaMemcpy(d_input, h_input, byte_size, cudaMemcpyHostToDevice);

  reduxUnrollingBlocks2 <<<grid, block>>>(d_input, d_temp, size);

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

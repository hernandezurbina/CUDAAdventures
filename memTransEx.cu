#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <cstring>
#include <time.h>

// define kernel
__global__ void mem_trans_ex(int *input) {
  int tid = threadIdx.x + blockDim.x * threadIdx.y + ((blockDim.x * blockDim.y * blockDim.z) * gridDim.y) * gridDim.z * threadIdx.z;

  int num_threads_per_block = blockDim.x * blockDim.y;
  int block_offset = blockIdx.x * num_threads_per_block;

  int num_threads_per_row = num_threads_per_block * gridDim.x;
  int row_offset = num_threads_per_row * blockIdx.y;

  int num_threads_per_grid = num_threads_per_row * gridDim.z;
  int grid_offset = num_threads_per_grid * blockIdx.z;

  int gid = tid + block_offset + row_offset + grid_offset;
  //int gid = tid + block_offset + row_offset;

  printf("blockIdx.x: %d, blockIdx.y: %d, blockIdx.z: %d, threadIdx.x: %d, threadIdx.y: %d, threadIdx.z: %d, tid: %d, gid: %d, value: %d\n", blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z, tid, gid, input[gid]);
}

int main() {
  int size = 64;
  int byte_size = size * sizeof(int);

  // initialize array in host
  int *h_input;
  h_input = (int *) malloc(byte_size);
  time_t t;
  srand((unsigned) time(&t));
  for(int i = 0; i < size; i++){
    //h_input[i] = (int) (rand() & 0xff);
    h_input[i] = i;
  }

  // initialize array in device
  int *d_input;
  cudaMalloc((void **)&d_input, byte_size);

  // copy array from host to device
  cudaMemcpy(d_input, h_input, byte_size, cudaMemcpyHostToDevice);

  // define threads
  dim3 block(2, 2, 2);
  dim3 grid(2, 2, 2);

  // launch kernel
  mem_trans_ex <<<grid, block>>>(d_input);

  // closing stuff
  cudaDeviceSynchronize();

  // free mem
  free(h_input);
  cudaFree(d_input);

  cudaDeviceReset();

  return 0;
}

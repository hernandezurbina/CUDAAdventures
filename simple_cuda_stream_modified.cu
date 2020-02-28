#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
// #include <time.h>
// #include <cstring>


#include "common.h"


__global__ void stream_test_modified(int *in, int *out, int size){
  int gid = blockIdx.x * blockDim.x + threadIdx.x;

  if (gid < size) {
    for(int i = 0; i < 25; i++){
      out[gid] = in[gid] + (in[gid] - 1) * (gid % 10);
    }
  }
}


int main(){

  int size = 1 << 18;
  int byte_size = size * sizeof(int);

  // host pointers
  int *h_in, *h_ref;

  cudaMallocHost((void **)&h_in, byte_size);
  cudaMallocHost((void **)&h_ref, byte_size);

  h_in = (int *) malloc(byte_size);
  h_ref = (int *) malloc(byte_size);
  initialize(h_in, INIT_RANDOM);

  // device pointers
  int *d_in, *d_out;

  cudaMalloc((void **)&d_in, byte_size);
  cudaMalloc((void **)&d_out, byte_size);

  cudaStream_t str;
  cudaStreamCreate(&str);

  cudaMemcpyAsync(d_in, h_in, byte_size, cudaMemcpyHostToDevice, str);

  dim3 block(128);
  dim3 grid(size/block.x);

  stream_test_modified <<<grid, block, 0, str>>>(d_in, d_out, size);
  cudaDeviceSynchronize();

  cudaMemcpyAsync(h_ref, d_out, byte_size, cudaMemcpyDeviceToHost, str);
  cudaStreamSynchronize(str);
  cudaStreamDestroy(str);

  cudaDeviceReset();
  return 0;
}

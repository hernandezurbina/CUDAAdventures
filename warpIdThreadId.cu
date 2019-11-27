#include <stdio.h>
#include <stdlib.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void printWarpDetails(){
  int gid = blockIdx.y * gridDim.x * blockDim.x + blockIdx.x * blockDim.x + threadIdx.x;

  int warpId = threadIdx.x / 32;

  int gbid = blockIdx.y * gridDim.x + blockIdx.x;

  printf("tid: %d, bid.x: %d, bid.y: %d, gid: %d, warpId: %d, gbid: %d\n", threadIdx.x, blockIdx.x, blockIdx.y, gid, warpId, gbid);
}


int main(){
  dim3 block(42);
  dim3 grid(2, 2);

  printWarpDetails <<<grid, block>>>();

  cudaDeviceSynchronize();
  cudaDeviceReset();
  return 0;
}

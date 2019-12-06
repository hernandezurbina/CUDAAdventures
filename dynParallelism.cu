#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void dynamic_parallelism_check(int size, int depth){
  printf("Depth: %d - tid: %d - blockIdx: %d\n", depth, threadIdx.x, blockIdx.x);
  if(size == 1){
    return;
  }
  if(threadIdx.x == 0) {
    dynamic_parallelism_check <<<1, size/2>>>(size/2, depth + 1);
  }
}

int main(){

  dim3 blockSize(16, 2);
  dim3 gridSize(1);

  dynamic_parallelism_check <<<gridSize, blockSize>>>(16, 0);

  cudaDeviceSynchronize();
  cudaDeviceReset();
  return 0;
}

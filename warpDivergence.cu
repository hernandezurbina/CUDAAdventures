#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void codeWithoutDivergence(){
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  int a, b;
  a = b = 0;
  int warpId = gid / 32;

  if(warpId % 2 == 0){
    a = 100;
    b = 50;
  }
  else {
    a = 200;
    b = 75;
  }
}

__global__ void codeWithDivergence(){
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  int a, b;
  a = b = 0;

  if(gid % 2 == 0){
    a = 100;
    b = 50;
  }
  else {
    a = 200;
    b = 75;
  }

}

int main(){
  int size = 1 << 22;

  dim3 blockSize(128);
  dim3 gridSize((size + blockSize.x - 1)/blockSize.x);

  codeWithoutDivergence <<<gridSize, blockSize>>>();
  cudaDeviceSynchronize();

  codeWithDivergence <<<gridSize, blockSize>>>();
  cudaDeviceReset();

  return 0;
}

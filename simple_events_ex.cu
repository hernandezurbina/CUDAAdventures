#include <stdio.h>
#include <stdlib.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "common.h"

__global__ void event_test(){
  double sum = 0.0;
  for(int i = 0; i < 1000; i++){
    sum += tan(0.1) * tan(0.1);
  }
}

int main(){

  int size = 1 << 12;

  dim3 block(128);
  dim3 grid(size / block.x);

  cudaEvent_t start, end;

  cudaEventCreate(&start);
  cudaEventCreate(&end);

  cudaEventRecord(start);

  event_test <<<grid, block>>>();

  cudaEventRecord(end);
  cudaEventSynchronize(end);

  float time;
  cudaEventElapsedTime(&time, start, end);

  printf("Elapsed time: %f\n", time);

  cudaEventDestroy(start);
  cudaEventDestroy(end);

  cudaDeviceReset();
  return 0;
}

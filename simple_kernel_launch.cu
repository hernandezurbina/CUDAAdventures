#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "common.h"

__global__ void simple_kernel(){
  printf("Hello from kernel!\n");
}

int main(){
  simple_kernel <<<1, 1>>>();
  simple_kernel <<<1, 1>>>();
  simple_kernel <<<1, 1>>>();

  cudaDeviceSynchronize();
  cudaDeviceReset();
  
  return 0;
}

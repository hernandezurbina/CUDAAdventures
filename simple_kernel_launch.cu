#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "common.h"

__global__ void simple_kernel(){
  printf("Hello from kernel!\n");
}

int main(){

  int dev = 0;
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, dev);

  if(deviceProp.concurrentKernels == 0){
    printf("GPU doesn't support concurrent execution\n");
    printf("Kernel exec will be serialized\n");
  }

  cudaStream_t str1, str2, str3;
  cudaStreamCreate(&str1);
  cudaStreamCreate(&str2);
  cudaStreamCreate(&str3);

  simple_kernel <<<1, 1, 0, str1>>>();
  simple_kernel <<<1, 1, 0, str2>>>();
  simple_kernel <<<1, 1, 0, str3>>>();

  cudaStreamDestroy(str1);
  cudaStreamDestroy(str2);
  cudaStreamDestroy(str3);

  cudaDeviceSynchronize();
  cudaDeviceReset();

  return 0;
}

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

void query_device(){
  int deviceCount = 0;
  cudaGetDeviceCount(&deviceCount);

  if (deviceCount == 0){
    printf("No CUDA support device found\n");
  }

  int devNo = 0;
  cudaDeviceProp iProp;
  cudaGetDeviceProperties(&iProp, devNo);

  printf("Device %d: %s\n", devNo, iProp.name);
  printf("Clock rate:  %d\n", iProp.clockRate);
  printf("Number of multiprocessors:  %d\n", iProp.multiProcessorCount);
  printf("Compute capability:  %d.%d\n", iProp.major, iProp.minor);
  printf("Amount of global memory:  %4.2f KB\n", (double) (iProp.totalGlobalMem/1024));
  printf("Amount of constant memory:  %4.2f KB\n", (double) (iProp.totalConstMem/1024));
  printf("Amount of shared memory per block:  %4.2f KB\n", (double) (iProp.sharedMemPerBlock/1024));
  printf("Max threads per block: %d\n", iProp.maxThreadsPerBlock);
  printf("Max block dimension: (%d, %d, %d)\n", iProp.maxThreadsDim[0], iProp.maxThreadsDim[1], iProp.maxThreadsDim[2]);
  printf("Max grid size: (%d, %d, %d)\n", iProp.maxGridSize[0], iProp.maxGridSize[1], iProp.maxGridSize[2]);
  return;
}

int main(){
  query_device();
  return 0;
}

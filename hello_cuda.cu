#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>


// DEVICE CODE:
// Kernel:
__global__ void hello_cuda(){
	printf("Hello CUDA world!\n");
}

// HOST CODE
int main(){

	// launching kernel:
	hello_cuda<<<1,20>>>();
	cudaDeviceSynchronize();
	cudaDeviceReset();

	return 0;
}
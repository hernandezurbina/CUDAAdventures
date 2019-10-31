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

	int nx, ny;
	nx = 16;
	ny = 4;

	dim3 block(8, 2, 1);
	dim3 grid(nx/block.x, ny/block.y, 1);

	// launching kernel:
	hello_cuda <<<grid, block>>>();
	cudaDeviceSynchronize();
	cudaDeviceReset();

	return 0;
}
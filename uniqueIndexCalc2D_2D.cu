#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>


__global__ void unique_gid_calculation_2D_2D(int *input){
	int tid = threadIdx.x + blockDim.x * threadIdx.y;

	int num_threads_per_block = blockDim.x * blockDim.y;
	int block_offset = blockIdx.x * num_threads_per_block;

	int num_threads_per_row = num_threads_per_block * gridDim.x;
	int row_offset = num_threads_per_row * blockIdx.y;
	int gid = tid + block_offset + row_offset;

	printf("blockIdx.x: %d, blockIdx.y: %d, threadIdx.x: %d, gid: %d, value: %d\n", blockIdx.x, blockIdx.y, tid, gid, input[gid]);
}

int main(){

	int array_size = 16;
	int array_bit_size = sizeof(int) * array_size;
	int h_data[] = {23, 9, 4, 53, 65, 12, 1, 33, 3, 92, 41, 54, 68, 11, 45, 21};

	for(int i = 0; i < array_size; i++){
		printf("%d ", h_data[i]);
	}
	printf("\n\n");

	int *d_data;

	cudaMalloc((void **)&d_data, array_bit_size);
	cudaMemcpy(d_data, h_data, array_bit_size, cudaMemcpyHostToDevice);

	dim3 block(2, 2);
	dim3 grid(2, 2);

	unique_gid_calculation_2D_2D <<<grid, block>>>(d_data);
	cudaDeviceSynchronize();

	cudaDeviceReset();
	return 0;
}
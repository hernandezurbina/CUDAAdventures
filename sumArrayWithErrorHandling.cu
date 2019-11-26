#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cstring>


#include "common.h"


__global__ void sum_array_gpu(int *a, int *b, int *c, int size){
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) {
    c[gid] = a[gid] + b[gid];
  }
}

void sum_array_cpu(int *a, int *b, int *c, int size){
  for(int i =0; i < size; i++){
    c[i] = a[i] + b[i];
  }
}

int main(){

  int size = 10000;
  int block_size = 128;
  int NO_BYTES = sizeof(int) * size;
  cudaError error;

  // host pointers
  int *h_a, *h_b, *h_c, *gpu_results;

  h_a = (int *) malloc(NO_BYTES);
  h_b = (int *) malloc(NO_BYTES);
  h_c = (int *) malloc(NO_BYTES);
  gpu_results = (int *) malloc(NO_BYTES);

  time_t t;
  srand((unsigned) time(&t));
  for(int i = 1; i < size; i++){
    h_a[i] = (int) (rand() & 0xFF);
    h_b[i] = (int) (rand() & 0xFF);
  }

  clock_t cpu_start, cpu_end;
  cpu_start = clock();
  sum_array_cpu(h_a, h_b, h_c, size);
  cpu_end = clock();

  // device pointers
  int *d_a, *d_b, *d_c;

  error = cudaMalloc((void **)&d_a, NO_BYTES);
  if (error != cudaSuccess) {
    fprintf(stderr, "Error: %s\n", cudaGetErrorString(error));
  }

  error = cudaMalloc((void **)&d_b, NO_BYTES);
  if (error != cudaSuccess) {
    fprintf(stderr, "Error: %s\n", cudaGetErrorString(error));
  }

  error = cudaMalloc((void **)&d_c, NO_BYTES);
  if (error != cudaSuccess) {
    fprintf(stderr, "Error: %s\n", cudaGetErrorString(error));
  }

  clock_t htod_start, htod_end;
  htod_start = clock();
  cudaMemcpy(d_a, h_a, NO_BYTES, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, NO_BYTES, cudaMemcpyHostToDevice);
  htod_end = clock();

  dim3 block(block_size);
  dim3 grid((size/block.x) + 1);

  clock_t gpu_start, gpu_end;
  gpu_start = clock();
  sum_array_gpu <<<block, grid>>>(d_a, d_b, d_c, size);
  cudaDeviceSynchronize();
  gpu_end = clock();

  clock_t dtoh_start, dtoh_end;
  dtoh_start = clock();
  cudaMemcpy(gpu_results, d_c, NO_BYTES, cudaMemcpyDeviceToHost);
  dtoh_end = clock();

  compare_arrays(gpu_results, h_c, size);


  printf("Sum array CPU exec time: %4.6f\n",(double)((double)(cpu_end - cpu_start)/CLOCKS_PER_SEC));
  printf("Sum array GPU exec time: %4.6f\n",(double)((double)(gpu_end - gpu_start)/CLOCKS_PER_SEC));
  printf("H2D transfer time: %4.6f\n",(double)((double)(htod_end - htod_start)/CLOCKS_PER_SEC));
  printf("D2H transfer time: %4.6f\n",(double)((double)(dtoh_end - dtoh_start)/CLOCKS_PER_SEC));

  printf("Total GPU exec time: %4.6f\n", (double)((double)(dtoh_end - htod_start)/CLOCKS_PER_SEC));

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  free(h_a);
  free(h_b);
  free(gpu_results);

  return 0;
}

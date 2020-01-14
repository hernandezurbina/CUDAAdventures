#include <stdio.h>
#include <stdlib.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "common.h"


__global__ void transpose_read_row_write_column(int *mat, int *transpose, int nx, int ny){
  int ix = blockIdx.x * blockDim.x + threadIdx.x;
  int iy = blockIdx.y * blockDim.y + threadIdx.y;

  if(ix < nx && iy < ny){
    transpose[ix * ny + iy] = mat[iy * nx + ix];
  }
}

int main(){

  int nx = 1024;
  int ny = 1024;

  int block_x = 128;
  int block_y = 8;

  int size = nx * ny;
  int byte_size = sizeof(int) * size;

  printf("Matrix transpose for %d X %d matrix with block size %d X %d\n", nx, ny, block_x, block_y);

  int *h_mat_array = (int *) malloc(byte_size);
  int *h_trans_array = (int *) malloc(byte_size);
  int *h_ref = (int *) malloc(byte_size);

  initialize(h_mat_array, size, INIT_ONE_TO_TEN);

  // matrix transpose in CPU
  mat_transpose_cpu(h_mat_array, h_trans_array, nx, ny);

  return 0;
}

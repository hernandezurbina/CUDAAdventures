#include <iostream>
#include <math.h>

// CUDA kernel function to add the elements of 2 arrays in the GPU
__global__
void add(int n, float *x, float *y){
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for(int i = index; i < n; i += stride){
    y[i] = x[i] + y[i];
  }
}

int main(void){
  int N = 1<<20; // 1M elements

  // allocate unified memory -- accessible from CPU or GPU
  float *x, *y;
  cudaMallocManaged(&x, N*sizeof(float));
  cudaMallocManaged(&y, N*sizeof(float));

  // initialize x and y arrays on host
  for(int i = 0; i < N; i++){
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  // run kernel on 1M elements on the GPU
  int blockSize = 256;
  int numBlocks = (N + blockSize - 1) / blockSize;
  add<<<numBlocks, blockSize>>>(N, x, y);

  // wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

  // check for errors (all values should be 3.0f)
  float maxError = 0.0f;

  for(int i = 1; i < N; i++){
    maxError = fmax(maxError, fabs(y[i]-3.0f));
  }
  std::cout << "Max error: " << maxError << std::endl;

  // free mem
  cudaFree(x);
  cudaFree(y);

  return 0;
}

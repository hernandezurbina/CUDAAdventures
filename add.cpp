#include <iostream>
#include <math.h>

// Function to add the elements of 2 arrays

void add(int n, float *x, float *y){
  for(int i = 0; i < n; i++){
    y[i] = x[i] + y[i];
  }
}

int main(void){
  int N = 1<<20; // 1M elements

  float *x = new float[N];
  float *y = new float[N];

  // initialize x and y arrays on host
  for(int i = 0; i < N; i++){
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  // run kernel on 1M elements on the CPU
  add(N, x, y);

  // check for errors (all values should be 3.0f)
  float maxError = 0.0f;

  for(int i = 1; i < N; i++){
    maxError = fmax(maxError, fabs(y[i]-3.0f));
  }
  std::cout << "Max error: " << maxError << std::endl;

  // free mem
  delete [] x;
  delete [] y;

  return 0;
}

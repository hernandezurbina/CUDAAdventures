
#include "common.h"


void compare_arrays(int * a, int * b, int size){
  for(int i = 0; i < size; i++){
    if(a[i] != b[i]){
      printf("Arrays are different!\n");
      return;
    }
  }
  printf("Arrays are the same!\n");
}

void compare_float_arrays(float * a, float * b, int size){
  for(int i = 0; i < size; i++){
    if(a[i] != b[i]){
      printf("Arrays are different!\n");
      return;
    }
  }
  printf("Arrays are the same!\n");
}

void sum_array_cpu(float* a, float* b, float *c, int size)
{
	for (int i = 0; i < size; i++)
	{
		c[i] = a[i] + b[i];
	}
}

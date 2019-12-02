#ifndef COMMON_H
#define COMMON_H

#include <stdio.h>
#include <cstring>
#include <time.h>
#include <stdlib.h>
#include <fstream>

enum INIT_PARAM{
	INIT_ZERO,INIT_RANDOM,INIT_ONE,INIT_ONE_TO_TEN,INIT_FOR_SPARSE_METRICS,INIT_0_TO_X
};

//simple initialization
void initialize(int * input, const int array_size, INIT_PARAM PARAM = INIT_ONE_TO_TEN, int x = 0);
void initialize(float * input, const int array_size, INIT_PARAM PARAM = INIT_ONE_TO_TEN);

void compare_arrays(int * a, int * b, int size);
void sum_array_cpu(float* a, float* b, float *c, int size);
void compare_float_arrays(float * a, float * b, int size);
int reduction_cpu(int *input, const int size);
void compare_results(int gpu_results, int cpu_results);


#endif // !COMMON_H

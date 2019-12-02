#ifndef COMMON_H
#define COMMON_H

#include <stdio.h>
#include <cstring>
#include <time.h>
#include <stdlib.h>
#include <fstream>


//compare two arrays

void compare_arrays(int * a, int * b, int size);
void sum_array_cpu(float* a, float* b, float *c, int size);
void compare_float_arrays(float * a, float * b, int size);

#endif // !COMMON_H

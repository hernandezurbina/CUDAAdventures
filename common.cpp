#include "common.h"

//simple initialization
void initialize(int * input, const int array_size, INIT_PARAM PARAM, int x)
{
	if (PARAM == INIT_ONE)
	{
		for (int i = 0; i < array_size; i++)
		{
			input[i] = 1;
		}
	}
	else if (PARAM == INIT_ONE_TO_TEN)
	{
		for (int i = 0; i < array_size; i++)
		{
			input[i] = i % 10;
		}
	}
	else if (PARAM == INIT_RANDOM)
	{
		time_t t;
		srand((unsigned)time(&t));
		for (int i = 0; i < array_size; i++)
		{
			input[i] = (int)(rand() & 0xFF);
		}
	}
	else if (PARAM == INIT_FOR_SPARSE_METRICS)
	{
		srand(time(NULL));
		int value;
		for (int i = 0; i < array_size; i++)
		{
			value = rand() % 25;
			if (value < 5)
			{
				input[i] = value;
			}
			else
			{
				input[i] = 0;
			}
		}
	}
	else if (PARAM == INIT_0_TO_X)
	{
		srand(time(NULL));
		int value;
		for (int i = 0; i < array_size; i++)
		{
			input[i] = (int)(rand() & 0xFF);
		}
	}
}

void initialize(float * input, const int array_size, INIT_PARAM PARAM)
{
	if (PARAM == INIT_ONE)
	{
		for (int i = 0; i < array_size; i++)
		{
			input[i] = 1;
		}
	}
	else if (PARAM == INIT_ONE_TO_TEN)
	{
		for (int i = 0; i < array_size; i++)
		{
			input[i] = i % 10;
		}
	}
	else if (PARAM == INIT_RANDOM)
	{
		srand(time(NULL));
		for (int i = 0; i < array_size; i++)
		{
			input[i] = rand() % 10;
		}
	}
	else if (PARAM == INIT_FOR_SPARSE_METRICS)
	{
		srand(time(NULL));
		int value;
		for (int i = 0; i < array_size; i++)
		{
			value = rand() % 25;
			if (value < 5)
			{
				input[i] = value;
			}
			else
			{
				input[i] = 0;
			}
		}
	}
}


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

int reduction_cpu(int *input, const int size){
  int sum = 0;
  for(int i = 0; i < size; i++){
    sum += input[i];
  }
  return sum;
}

void compare_results(int gpu_results, int cpu_results){
  printf("GPU result: %d, CPU result: %d\n", gpu_results, cpu_results);
  if(gpu_results == cpu_results) {
    printf("GPU and CPU results are the same!\n");
    return;
  }
  printf("GPU and CPU results are different!\n");
  return;
}

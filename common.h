#ifndef COMMON_H
#define COMMON_H

#include <stdio.h>
#include <cstring>
#include <time.h>
#include <stdlib.h>
#include <sys/utime.h>
#include <fstream>

#define HANDLE_NULL( a ){if (a == NULL) { \
                            printf( "Host memory failed in %s at line %d\n", \
                                    __FILE__, __LINE__ ); \
                            exit( EXIT_FAILURE );}}

enum INIT_PARAM{
	INIT_ZERO,INIT_RANDOM,INIT_ONE,INIT_ONE_TO_TEN,INIT_FOR_SPARSE_METRICS,INIT_0_TO_X
};


//compare two arrays
void compare_arrays(int * a, int * b, int size);


#endif // !COMMON_H

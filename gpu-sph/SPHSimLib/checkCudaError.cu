
#include "cudpp/cudpp.h"
//#include <algorithm>
#include <stdio.h>

typedef unsigned int uint;

extern "C"
void checkCudaError(const char *msg)
{
#if defined(_DEBUG) || defined(DEBUG)
    cudaError_t e = cudaThreadSynchronize();
    if( e != cudaSuccess )
    {
        fprintf(stderr, "CUDA Error %s : %s\n", msg, cudaGetErrorString(e));
        exit(EXIT_FAILURE);
    }
    e = cudaGetLastError();
    if( e != cudaSuccess )
    {
        fprintf(stderr, "CUDA Error %s : %s\n", msg, cudaGetErrorString(e));
        exit(EXIT_FAILURE);
    }
#endif
}


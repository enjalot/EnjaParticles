#ifndef _SDOT_CL_
#define _SDOT_CL_

#include "float_type.h"

#define CUBLAS_SDOT_LOG_THREAD_COUNT    (7)
#define CUBLAS_SDOT_THREAD_COUNT        (1 << CUBLAS_SDOT_LOG_THREAD_COUNT)
#define CUBLAS_SDOT_CTAS                (80)

//----------------------------------------------------------------------
//__kernel void sdot_gld_main (int n, const FLOAT* sx, int incx,
__kernel void sdot_gld_main (int n, __global FLOAT* sx, int incx,
__global FLOAT* sy, int incy, __global FLOAT* result)
{
//extern __shared__ FLOAT partialSum[];
__local FLOAT partialSum[CUBLAS_SDOT_THREAD_COUNT];

#undef fetchx
#undef fetchy
#define fetchx(i)  sx[i]
#define fetchy(i)  sy[i]

    unsigned int i, tid, totalThreads, ctaStart;
    FLOAT sum = 0.0f;
    //tid = threadIdx.x;
    tid = get_local_id(0);
    //totalThreads = gridDim.x * CUBLAS_SDOT_THREAD_COUNT;
    totalThreads = get_num_groups(0) * CUBLAS_SDOT_THREAD_COUNT;
    //ctaStart = CUBLAS_SDOT_THREAD_COUNT * blockIdx.x;
    ctaStart = CUBLAS_SDOT_THREAD_COUNT * get_group_id(0);

        /* equal, positive, increments */
        //if (incx == 1) {
            /* both increments equal to 1 */
            for (i = ctaStart + tid; i < n; i += totalThreads) {
                sum += fetchy(i) * fetchx(i);
            }
        //} 

    partialSum[tid] = sum;
    //partialSum[tid] = totalThreads;
    //partialSum[tid] = CUBLAS_SDOT_THREAD_COUNT;
    //partialSum[tid] = get_num_groups(0);

	#if 1
    for (i = (CUBLAS_SDOT_THREAD_COUNT >> 1); i > 0; i >>= 1) {
        //__syncthreads(); 
		barrier(CLK_LOCAL_MEM_FENCE);
        if (tid < i) {
            partialSum[tid] += partialSum[tid + i];
		} 
    }
	#endif

    //__syncthreads(); 
	barrier(CLK_LOCAL_MEM_FENCE);
    if (tid == 0) {
        //result[blockIdx.x] = partialSum[tid];
        result[get_group_id(0)] = partialSum[tid];
    }
}
//----------------------------------------------------------------------


#endif

#ifndef _SAXPY_CL_
#define _SAXPY_CL_
/* This file contains the implementation of the BLAS-1 function saxpy */

#include "float_type.h"

#define USE_TEX 0

#define blockIdxx  get_group_id(0)
#define blockIdxy  get_group_id(1)
#define blockIdxz  get_group_id(2)

#define gridDimx get_num_groups(0)
#define gridDimy get_num_groups(1)
#define gridDimz get_num_groups(2)

#define blockDimx get_local_size(0);
#define blockDimy get_local_size(1);
#define blockDimz get_local_size(2);

#define threadIdxx get_local_id(0);
#define threadIdxy get_local_id(1);
#define threadIdxz get_local_id(2);

#define _syncthreads barrier(CLK_LOCAL_MEM_FENCE)


__kernel void saxpy_gld_main (int n, FLOAT alpha, __global FLOAT *sx, int incx, __global FLOAT* sy, int incy)
{
#define USE_TEX 0

#undef fetchx
#undef fetchy
#define fetchx(i)  sx[i]
#define fetchy(i)  sy[i]

    int i, tid, totalThreads, ctaStart;
	FLOAT alp = alpha;  // bring to local memory?
    
    /* NOTE: wrapper must ensure that parms.n > 0  */
    //tid = threadIdxx;
    tid = get_local_id(0);

    //totalThreads = gridDimx*blockDimx;
    //ctaStart = blockDimx*blockIdxx;

    totalThreads = get_num_groups(0)*get_local_size(0);
    ctaStart = get_local_size(0)*get_group_id(0); // compiles (WHY?)
   
	//if (incx == 1) {
		/* both increments equal to 1 */
		for (i = ctaStart + tid; i < n; i += totalThreads) {
			sy[i] = fetchy(i) + alp * fetchx(i);
		}
	//} 
}


#endif


/* This file contains the implementation of the BLAS-1 function sscal */
/* with incx = 1 */

//#include "macros.h"
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

#define USE_TEX 0


__kernel void sscal_gld_main (int n, __global float* sx, float alpha, int incx) {

#undef fetchx
#define fetchx(i)  sx[i]

    int i,  tid, totalThreads, ctaStart;

    /* NOTE: wrapper must ensure that parms.n > 0 and parms.incx > 0 */
    //tid = threadIdxx;
    tid = get_local_id(0);
    totalThreads = gridDimx * blockDimx;
    //ctaStart = blockDimx * blockIdxx; // does not compile (WHY?)
    ctaStart = get_local_size(0)*get_group_id(0); // compiles (WHY?)
    
    /* increment equal to 1 */
    for (i = ctaStart + tid; i < n; i += totalThreads) {
        sx[i] = fetchx(i) * alpha;
    }
}


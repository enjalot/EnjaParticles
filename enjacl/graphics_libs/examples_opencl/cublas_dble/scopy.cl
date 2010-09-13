#ifndef _SCOPY_CL_
#define _SCOPY_CL_

#include "float_type.h"

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

__kernel void scopy_gld_main (int n, __global FLOAT* sxorig, int incx, __global FLOAT* sydest, int incy)
{
#undef fetchx
#define fetchx(i)  sxorig[i]

    //int i, n, tid, totalThreads, ctaStart;
    int i, tid, totalThreads, ctaStart;

    /* NOTE: wrapper must ensure that parms.n > 0  */
    //tid = threadIdxx;
    tid = get_local_id(0);
    totalThreads = gridDimx * blockDimx;
    //ctaStart = blockDimx * blockIdxx;
    ctaStart = get_local_size(0)*get_group_id(0); // compiles (WHY?)

        //if (incx == 1) {
            /* both increments equal to 1 */
            for (i = ctaStart + tid; i < n; i += totalThreads) {
                sydest[i] = fetchx(i);
            }
        //}
}


#endif

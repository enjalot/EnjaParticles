
/* element by element multiplication */
/*  result = sx * sy */

//#include "cublasP.h"

__global__ void svecvec_gld_main (int n, float* sx, float* sy, float* result) {

#undef fetchx
#undef fetchy
#define fetchx(i)  sx[i]
#define fetchy(i)  sy[i]

    int i,  tid, totalThreads, ctaStart;

    tid = threadIdx.x;
    totalThreads = gridDim.x * blockDim.x;
    ctaStart = blockDim.x * blockIdx.x;

	/* increment equal to 1 */
	for (i = ctaStart + tid; i < n; i = i+totalThreads) {
		/* __syncThreads(); */
		result[i] = fetchx(i) * fetchy(i);
	}
}


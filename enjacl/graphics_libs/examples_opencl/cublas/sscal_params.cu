
/* This file contains the implementation of the BLAS-1 function sscal */

#include "cublasP.h"
#define USE_TEX 0

struct cublas_paramsp {
	float* x;
    int   n;
    float sa;
    int   incx;
};



__global__ void sscal_params_gld_main(struct cublas_paramsp* parms) {
//__global__ void sscal_params_gld_main (int n, float* sx, float alpha, int incx) {


#undef fetchx
#define fetchx(i)  sx[i]

    int i,  tid, totalThreads, ctaStart;

	//cublasSscalParams& parms = *pparms;

    /* NOTE: wrapper must ensure that parms.n > 0 and parms.incx > 0 */
    tid = threadIdx.x;
    int n = parms->n;
    float* sx = parms->x;
	float alpha = parms->sa;
	int incx = parms->incx;


    totalThreads = gridDim.x * blockDim.x;
    ctaStart = blockDim.x * blockIdx.x;
    
    if (incx == 1) {
        /* increment equal to 1 */
        for (i = ctaStart + tid; i < n; i += totalThreads) {
            //sx[i] = fetchx(i) * parms.sa;
            sx[i] = fetchx(i) * alpha;
			//alpha = sx[5];
        }
    } 
}


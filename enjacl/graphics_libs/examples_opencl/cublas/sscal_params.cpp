
/* This file contains the implementation of the BLAS-1 function sscal */

#include <cublasP.h>

#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <stdio.h>
#include <limits.h>
#include <math.h>

#include "swan_types.h"
#include "swan_defines.h"
//#include "swan_api.h"


struct cublas_paramsp {
	float* x;
    int   n;
    float sa;
    int   incx;
};

#include "cublasP.h"
#include "sscal_params.kh"

//GE texture<float> texX;

//__global__ void sscal_gld_main (struct cublasSscalParams parms);

//----------------------------------------------------------------------
// From Nvidia
/* 
 * For a given vector size, cublasVectorSplayParams() determines what CTA grid 
 * size to use, and how many threads per CTA.
 */
void cublasVectorSplayParams (int n, int tMin, int tMax, int gridW, int *nbrCtas, 
                        int *elemsPerCta, int *threadsPerCta)
{
    if (n < tMin) {
        *nbrCtas = 1;
        *elemsPerCta = n;
        *threadsPerCta = tMin;
    } else if (n < (gridW * tMin)) {
        *nbrCtas = ((n + tMin - 1) / tMin);
        *threadsPerCta = tMin;
        *elemsPerCta = *threadsPerCta;
    } else if (n < (gridW * tMax)) {
        int grp;
        *nbrCtas = gridW;
        grp = ((n + tMin - 1) / tMin);
        *threadsPerCta = (((grp + gridW -1) / gridW) * tMin);
        *elemsPerCta = *threadsPerCta;
    } else {
        int grp;
        *nbrCtas = gridW;
        *threadsPerCta = tMax;
        grp = ((n + tMin - 1) / tMin);
        grp = ((grp + gridW - 1) / gridW);
        *elemsPerCta = grp * tMin;
    }
}
//----------------------------------------------------------------------

/*
 * void
 * cublasSscal (int n, float alpha, float *x, int incx)
 *
 * replaces single precision vector x with single precision alpha * x. For i 
 * = 0 to n - 1, it replaces x[ix + i * incx] with alpha * x[ix + i * incx], 
 * where ix = 1 if incx >= 0, else ix = 1 + (1 - n) * incx.
 *
 * Input
 * -----
 * n      number of elements in input vectors
 * alpha  single precision scalar multiplier
 * x      single precision vector with n elements
 * incx   storage spacing between elements of x
 *
 * Output
 * ------
 * x      single precision result (unchanged if n <= 0 or incx <= 0)
 *
 * Reference: http://www.netlib.org/blas/sscal.f
 *
 * Error status for this function can be retrieved via cublasGetError().
 * 
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
 */
__host__ void CUBLASAPI cublasSscalParams (int n, float alpha, float *x, int incx)
{
    struct cublasSscalParams params;
    //cudaError_t cudaStat;
    int nbrCtas;
    int elemsPerCta;
    int threadsPerCta;
    int sizeX = n * (imax (1, abs(incx)));
    //size_t texXOfs = 0;
    int useTexture = 0;

    /* early out if nothing to do */
    if ((n <= 0) || (incx <= 0)) {
		printf("nothing to do\n");
        return;
    }

    struct cublas_paramsp paramsp;
    memset (&paramsp, 0, sizeof(paramsp));
	//We'll try without pointers
    paramsp.n  = n;
    paramsp.x = x; // setting this can create problems
    paramsp.sa = alpha;
    paramsp.incx = incx;

	struct cublas_paramsp *paramsp_d = 
		(struct cublas_paramsp*) swanMalloc(sizeof(paramsp));

    //memset (&params, 0, sizeof(params));
	swanMemcpyHtoD(&paramsp, paramsp_d, sizeof(paramsp));

//printf("sizeof(paramsp)= %d\n", sizeof(paramsp));


    cublasVectorSplayParams (n, CUBLAS_SSCAL_THREAD_MIN, CUBLAS_SSCAL_THREAD_MAX,
                       CUBLAS_SSCAL_CTAS_MAX, &nbrCtas, &elemsPerCta, 
                       &threadsPerCta);

    //GE cudaStat = cudaGetLastError(); /* clear error status */

	dim3 grid; 
	grid.x = nbrCtas;
	grid.y = 1;
	grid.z = 1;
	dim3 block;
	block.x = threadsPerCta;
	block.y = 1;
	block.z = 1;

	//printf("nbctas= %d, threadsPerCta= %d\n", nbrCtas, threadsPerCta);


    //GE sscal_gld_main<<<nbrCtas,threadsPerCta>>>(params);
    //k_sscal_gld_main(grid, block, 0, &params);
    //k_sscal_params_gld_main(grid, block, 0, n, x, alpha, incx);
    k_sscal_params_gld_main_async(grid, block, 0, paramsp_d);

    //cudaStat = cudaGetLastError(); /* check for launch error */
}


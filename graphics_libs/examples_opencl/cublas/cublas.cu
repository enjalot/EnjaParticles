/*
 * Copyright 1993-2008 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:   
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and
 * international Copyright laws.  
 *
 * This software and the information contained herein is being provided 
 * under the terms and conditions of a Source Code License Agreement.     
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR 
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
 * OF USE, DATA OR PROFITS,  WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
 * OR OTHER TORTIOUS ACTION,  ARISING OUT OF OR IN CONNECTION WITH THE USE
 * OR PERFORMANCE OF THIS SOURCE CODE.  
 *
 * U.S. Government End Users.   This source code is a "commercial item" as 
 * that term is defined at  48 C.F.R. 2.101 (OCT 1995), consisting  of
 * "commercial computer  software"  and "commercial computer software 
 * documentation" as such terms are  used in 48 C.F.R. 12.212 (SEPT 1995)
 * and is provided to the U.S. Government only as a commercial end item.
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the 
 * source code with only those rights set forth herein.
 */

/* This file contains the implementation of the CUBLAS helper functions */

#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <stdio.h>
#include <limits.h>
#include <math.h>
#include "cublas.h"   /* CUBLAS public header file  */
#include "cublasP.h"  /* CUBLAS private header file */

#include "swan_defines.h"
#include "swan_api.h"

/* the next two macro definitions trigger
 * code generation when tlsHook.h is included
 */ 
#define __tlsHookIdentifier cublasThreadContext
#define __tlsHookType       struct cublasContext
//GE #include <tlshook.h>
typedef int __tlsHookStatus;
__tlsHookStatus __tlsHookStatusFAIL;
__tlsHookStatus __tlsHookStatusOK;
//__tlsHookInitTlsValueForcublasThreadContext
//cublas.cu(169): error: identifier "cublasThreadContext" is undefined
//cublas.cu(169): error: identifier "__tlsHookClearTlsValue" is undefined
//cublas.cu(187): error: identifier "CUBLAS_GET_CTX" is undefined


void cublasSetError (struct cublasContext *ctx, cublasStatus error)
{
    if (ctx) {
        ctx->cublasLastError = error;
    }
}

int cublasInitialized (const struct cublasContext *ctx)
{
    if (!ctx) {
        return 0;
    } else {
        return ctx->cublasIsInitialized;
    }
}

/* 
 * For a given vector size, cublasVectorSplay() determines what CTA grid 
 * size to use, and how many threads per CTA.
 */

void cublasShutDownCtx (struct cublasContext *ctx)
{
}

__tlsHookStatus cublasInitCtx (struct cublasContext *ctx, void *_status)
{
    return __tlsHookStatusOK;
}

/* --------------------------- CUBLAS API functions ------------------------ */

/*
 * cublasStatus 
 * cublasInit (void)
 *
 * initializes the CUBLAS library and must be called before any other 
 * CUBLAS API function is invoked. It allocates hardware resources 
 * necessary for accessing the GPU.
 *
 * Return Values
 * -------------
 * CUBLAS_STATUS_ALLOC_FAILED     if resources could not be allocated
 * CUBLAS_STATUS_SUCCESS          if CUBLAS library initialized successfully
 */
cublasStatus CUBLASAPI cublasInit (void)
{
    cublasStatus status = CUBLAS_STATUS_SUCCESS;
    
    //GE (void)__tlsHookInitTlsValueForcublasThreadContext(cublasInitCtx, 
	//                                                      cublasShutDownCtx, 
	//                                                      &status);
    return status;
}

/*
 * cublasStatus 
 * cublasShutdown (void)
 *
 * releases CPU-side resources used by the CUBLAS library. The release of 
 * GPU-side resources may be deferred until the application shuts down.
 *
 * Return Values
 * -------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_SUCCESS          if CUBLAS library shut down successfully
 */
cublasStatus CUBLASAPI cublasShutdown (void)
{
    //GE __tlsHookClearTlsValue(&cublasThreadContext);
    return CUBLAS_STATUS_SUCCESS;
}

/* 
 * cublasStatus 
 * cublasGetError (void)
 *
 * returns the last error that occurred on invocation of any of the
 * CUBLAS BLAS functions. While the CUBLAS helper functions return status
 * directly, the BLAS functions do not do so for improved 
 * compatibility with existing environments that do not expect BLAS
 * functions to return status. Reading the error status via 
 * cublasGetError() resets the internal error state to 
 * CUBLAS_STATUS_SUCCESS.
 */
cublasStatus CUBLASAPI cublasGetError (void)
{
	//GE
	return 0;

	#if 0 
	#endif
}

/*
 * cublasStatus 
 * cublasAlloc (int n, int elemSize, void **devicePtr)
 *
 * creates an object in GPU memory space capable of holding an array of
 * n elements, where each element requires elemSize bytes of storage. If 
 * the function call is successful, a pointer to the object in GPU memory 
 * space is placed in devicePtr. Note that this is a device pointer that
 * cannot be dereferenced in host code.
 *
 * Return Values
 * -------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_INVALID_VALUE    if n <= 0, or elemSize <= 0
 * CUBLAS_STATUS_ALLOC_FAILED     if the object could not be allocated due to
 *                                lack of resources.
 * CUBLAS_STATUS_SUCCESS          if storage was successfully allocated
 */
cublasStatus CUBLASAPI cublasAlloc (int n, int elemSize, void **devicePtr)
{

    //GE cudaError_t cudaStat;

	//GE
	#if 0
    struct cublasContext *ctx = CUBLAS_GET_CTX();
	#endif

    *devicePtr = 0;

	//GE
	#if 0
    if (!cublasInitialized (ctx)) {
        return CUBLAS_STATUS_NOT_INITIALIZED;
    }
	#endif

    if ((n <= 0) || (elemSize <= 0)) {        
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    //GE cudaStat = cudaMalloc (devicePtr, elemSize * n);
    *devicePtr = swanMalloc (elemSize * n);
    //GE if (cudaStat != cudaSuccess) {
        //GE return CUBLAS_STATUS_ALLOC_FAILED;
    //GE }
    return CUBLAS_STATUS_SUCCESS;
}

/*
 * cublasStatus 
 * cublasFree (const void *devicePtr)
 *
 * destroys the object in GPU memory space pointed to by devicePtr.
 *
 * Return Values
 * -------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_INTERNAL_ERROR   if the object could not be deallocated
 * CUBLAS_STATUS_SUCCESS          if object was destroyed successfully
 */
cublasStatus CUBLASAPI cublasFree (const void *devicePtr)
{
	return 0;

	# if 0
	#endif
}

/* 
 * cublasStatus 
 * cublasSetVector (int n, int elemSize, const void *x, int incx, 
 *                  void *y, int incy) 
 *
 * copies n elements from a vector x in CPU memory space to a vector y 
 * in GPU memory space. Elements in both vectors are assumed to have a 
 * size of elemSize bytes. Storage spacing between consecutive elements
 * is incx for the source vector x and incy for the destination vector
 * y. In general, y points to an object, or part of an object, allocated
 * via cublasAlloc(). Column major format for two-dimensional matrices
 * is assumed throughout CUBLAS. Therefore, if the increment for a vector 
 * is equal to 1, this access a column vector while using an increment 
 * equal to the leading dimension of the respective matrix accesses a 
 * row vector.
 *
 * Return Values
 * -------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library not been initialized
 * CUBLAS_STATUS_INVALID_VALUE    if incx, incy, or elemSize <= 0
 * CUBLAS_STATUS_MAPPING_ERROR    if an error occurred accessing GPU memory   
 * CUBLAS_STATUS_SUCCESS          if the operation completed successfully
 */
cublasStatus CUBLASAPI cublasSetVector (int n, int elemSize, 
                                        const void *hostPtr, int incx, 
                                        void *devicePtr, int incy)
{
	return 0;

	#if 0
	#endif
}

/* 
 * cublasStatus 
 * cublasGetVector (int n, int elemSize, const void *x, int incx, 
 *                  void *y, int incy)
 * 
 * copies n elements from a vector x in GPU memory space to a vector y 
 * in CPU memory space. Elements in both vectors are assumed to have a 
 * size of elemSize bytes. Storage spacing between consecutive elements
 * is incx for the source vector x and incy for the destination vector
 * y. In general, x points to an object, or part of an object, allocated
 * via cublasAlloc(). Column major format for two-dimensional matrices
 * is assumed throughout CUBLAS. Therefore, if the increment for a vector 
 * is equal to 1, this access a column vector while using an increment 
 * equal to the leading dimension of the respective matrix accesses a 
 * row vector.
 *
 * Return Values
 * -------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library not been initialized
 * CUBLAS_STATUS_INVALID_VALUE    if incx, incy, or elemSize <= 0
 * CUBLAS_STATUS_MAPPING_ERROR    if an error occurred accessing GPU memory   
 * CUBLAS_STATUS_SUCCESS          if the operation completed successfully
 */
cublasStatus CUBLASAPI cublasGetVector (int n, int elemSize,
                                        const void *devicePtr, int incx,
                                        void *hostPtr, int incy)
{
	// GE
	#if 0
    struct cublasContext *ctx = CUBLAS_GET_CTX();
	#endif

    //GE cudaError_t cudaStat = cudaSuccess;
    const char *sp = (const char *)devicePtr;
    char *dp = (char *)hostPtr;

	//GE
	#if 0
    if (!cublasInitialized (ctx)) {
        return CUBLAS_STATUS_NOT_INITIALIZED;
    }
	#endif

    if ((incx <= 0) || (incy <= 0) || (elemSize <= 0)) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    /* early out if nothing to do */
    if (n == 0) {
        return CUBLAS_STATUS_SUCCESS;
    }
    if ((incx == 1) && (incy == 1)) {
        //GE cudaStat = cudaMemcpy (dp, sp, n * elemSize, cudaMemcpyDeviceToHost);
       swanMemcpyDtoH ((void *) devicePtr, dp, n * elemSize);
    } else {
		printf("incx or inc != 1, not implemented\n"); exit(0);
        //Ge cudaStat = cudaMemcpy2D (dp, incy * elemSize, sp, incx * elemSize,
                                 //GE elemSize, n, cudaMemcpyDeviceToHost);
    }
	#if 0
    if (cudaStat != cudaSuccess) {
        return CUBLAS_STATUS_MAPPING_ERROR;
    }
	#endif
    return CUBLAS_STATUS_SUCCESS;
}

/*
 * cublasStatus 
 * cublasSetMatrix (int rows, int cols, int elemSize, const void *A, 
 *                  int lda, void *B, int ldb)
 *
 * copies a tile of rows x cols elements from a matrix A in CPU memory
 * space to a matrix B in GPU memory space. Each element requires storage
 * of elemSize bytes. Both matrices are assumed to be stored in column 
 * major format, with the leading dimension (i.e. number of rows) of 
 * source matrix A provided in lda, and the leading dimension of matrix B
 * provided in ldb. In general, B points to an object, or part of an 
 * object, that was allocated via cublasAlloc().
 *
 * Return Values 
 * -------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_INVALID_VALUE    if rows or cols < 0, or elemSize, lda, or 
 *                                ldb <= 0
 * CUBLAS_STATUS_MAPPING_ERROR    if error occurred accessing GPU memory
 * CUBLAS_STATUS_SUCCESS          if the operation completed successfully
 */
cublasStatus CUBLASAPI cublasSetMatrix (int rows, int cols, int elemSize,
                                        const void *A, int lda, void *B,
                                        int ldb)
{
	return 0;

	#if 0
	#endif
}

/*
 * cublasStatus 
 * cublasGetMatrix (int rows, int cols, int elemSize, const void *A, 
 *                  int lda, void *B, int ldb)
 *
 * copies a tile of rows x cols elements from a matrix A in GPU memory
 * space to a matrix B in CPU memory space. Each element requires storage
 * of elemSize bytes. Both matrices are assumed to be stored in column 
 * major format, with the leading dimension (i.e. number of rows) of 
 * source matrix A provided in lda, and the leading dimension of matrix B
 * provided in ldb. In general, A points to an object, or part of an 
 * object, that was allocated via cublasAlloc().
 *
 * Return Values 
 * -------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_INVALID_VALUE    if rows, cols, eleSize, lda, or ldb <= 0
 * CUBLAS_STATUS_MAPPING_ERROR    if error occurred accessing GPU memory
 * CUBLAS_STATUS_SUCCESS          if the operation completed successfully
 */
cublasStatus CUBLASAPI cublasGetMatrix (int rows, int cols, int elemSize,
                                        const void *A, int lda, void *B,
                                        int ldb)
{
	return 0;

	#if 0
	#endif
}

/* -------------------------- stub functions ------------------------------- */


/* Add a GUID to the compiled library for tracking purposes */
//GE #include "../../cuda/common/version.h"
//GE CUDA_STAMP_GUID;


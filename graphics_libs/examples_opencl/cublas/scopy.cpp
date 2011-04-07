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

/* This file contains the implementation of the BLAS-1 function scopy */


#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <stdio.h>
#include <limits.h>
#include <math.h>

#include <cl.h>
using namespace std;

#include <timege.h>
extern GE::Time clock_scopy_cpu;
extern GE::Time clock_scopy_gpu;

//----------------------------------------------------------------------
/*
 * void 
 * scopy (int n, const float *x, int incx, float *y, int incy)
 *
 * copies the single precision vector x to the single precision vector y. For 
 * i = 0 to n-1, copies x[lx + i * incx] to y[ly + i * incy], where lx = 1 if 
 * incx >= 0, else lx = 1 + (1 - n) * incx, and ly is defined in a similar 
 * way using incy.
 *
 * Input
 * -----
 * n      number of elements in input vectors
 * x      single precision vector with n elements
 * incx   storage spacing between elements of x
 * y      single precision vector with n elements
 * incy   storage spacing between elements of y
 *
 * Output
 * ------
 * y      contains single precision vector x
 *
 * Reference: http://www.netlib.org/blas/scopy.f
 *
 * Error status for this function can be retrieved via cublasGetError(). 
 *
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
 */
void cublasScopy (int n, cl_mem xsrc, int incx,
                                     cl_mem ydst, int incy)
{
	static cll_Program* prog = 0;
    int nbrCtas;
    int elemsPerCta;
    int threadsPerCta;
    int useTexture = 0;

	CL cl(true);
	cl.waitForKernelsToFinish();
	clock_scopy_cpu.begin();

    /* early out if nothing to do */
    if ((n <= 0) || (incx <= 0)) {
        return;
    }
    

	if (prog == 0) {
		string path(CL_SOURCE_DIR);
		path = path + "/scopy.cl";
		cll_Program& progr = cl.addProgramR(path.c_str());
		prog = &progr;
	}


	cll_Kernel kern = prog->addKernel("scopy_gld_main");

	//printf("after addKernel\n"); exit(0);

	kern.setArg(n, 0);
	kern.setArg(xsrc, 1);
	kern.setArg(incx, 2);
	kern.setArg(ydst, 3);
	kern.setArg(incy, 4);


	size_t global = (size_t) n;
	size_t local = cl.getMaxWorkSize(kern.getKernel());

	clock_scopy_gpu.begin();
    cl_event exec = kern.exec(1, &global, &local);

	cl.waitForKernelsToFinish();
	clock_scopy_gpu.end();

	clock_scopy_cpu.end();

	// methods are asynchronous by default
	// synchronized version
	//cl.waitForKernelsToFinish();
	//cl.profile(exec); // cannot use without waitForKernelsToFinish()
}



#include <timege.h>
extern GE::Time clock_scale_cpu;
extern GE::Time clock_scale_gpu;

/* This file contains the implementation of the BLAS-1 function sscal */

#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <stdio.h>
#include <limits.h>
#include <math.h>

#include "float_type.h"
#include "cl.h"

using namespace std;

//__global__ void sscal_gld_main (struct cublasSscalParams parms);

//----------------------------------------------------------------------

/*
 * void
 * cublasSscal (int n, FLOAT alpha, FLOAT *x, int incx)
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

// arguments are points in device space
void cublasSscal (int n, FLOAT alpha, cl_mem x, int incx)
//void cublasSscal (int n, FLOAT alpha, FLOAT *x, int incx)
{
	static cll_Program* prog = 0;
    int nbrCtas;
    int elemsPerCta;
    int threadsPerCta;
    int useTexture = 0;

	CL cl(true);
	cl.waitForKernelsToFinish();

	clock_scale_cpu.begin();

    /* early out if nothing to do */
    if ((n <= 0) || (incx <= 0)) {
        return;
    }
    

	if (prog == 0) {
		string path(CL_SOURCE_DIR);
		path += "/sscal.cl"; 

		string double_option;
#ifdef DOUBLE 
		double_option = " -DDOUBLE ";
#else
		double_option = "";
#endif

		string graphic_libs_dir = getenv("GRAPHIC_LIBS_HOME");
		string options = double_option + "-I" + graphic_libs_dir + "/include";
		cl.setCompilerOptions(options.c_str());

		cll_Program& progr = cl.addProgramR(path.c_str());
		prog = &progr;
	}


	cll_Kernel kern = prog->addKernel("sscal_gld_main");

	kern.setArg(n, 0);
	kern.setArg(x, 1);
	kern.setArg(alpha, 2);
	kern.setArg(incx, 3);

	size_t global = (size_t) n;
	size_t local = cl.getMaxWorkSize(kern.getKernel());


	clock_scale_gpu.begin();
    cl_event exec = kern.exec(1, &global, &local);

	cl.waitForKernelsToFinish();
	clock_scale_gpu.end();

	clock_scale_cpu.end();

	//cl.profile(exec);

	// methods are asynchronous by default
	// synchronized version
	//cl.waitForKernelsToFinish();
	//cl.profile(exec); // cannot use without waitForKernelsToFinish()
}


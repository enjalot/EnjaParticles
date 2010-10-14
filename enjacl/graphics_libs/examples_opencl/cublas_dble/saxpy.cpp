 /* This file contains the implementation of the BLAS-1 function saxpy */

#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <stdio.h>
#include <limits.h>
#include <math.h>
#include <timege.h>

#include "cl.h"
#include "float_type.h"

using namespace std;

extern GE::Time clock_saxpy_cpu;
extern GE::Time clock_saxpy_gpu;

//----------------------------------------------------------------------
/*
 * void
 * saxpy (int n, FLOAT alpha, const FLOAT *x, int incx, FLOAT *y, int incy)
 *
 * multiplies single precision vector x by single precision scalar alpha 
 * and adds the result to single precision vector y; that is, it overwrites 
 * single precision y with single precision alpha * x + y. For i = 0 to n - 1, 
 * it replaces y[ly + i * incy] with alpha * x[lx + i * incx] + y[ly + i *
 * incy], where lx = 1 if incx >= 0, else lx = 1 +(1 - n) * incx, and ly is 
 * defined in a similar way using incy.
 *
 * Input
 * -----
 * n      number of elements in input vectors
 * alpha  single precision scalar multiplier
 * x      single precision vector with n elements
 * incx   storage spacing between elements of x
 * y      single precision vector with n elements
 * incy   storage spacing between elements of y
 *
 * Output
 * ------
 * y      single precision result (unchanged if n <= 0)
 *
 * Reference: http://www.netlib.org/blas/saxpy.f
 *
 * Error status for this function can be retrieved via cublasGetError().
 * 
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has not been initialized
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to launch on GPU
 */
//void CUBLASAPI cublasSaxpy (int n, FLOAT alpha, const FLOAT* FLOAT *x,
//                                     int incx, FLOAT *y, int incy)
void cublasSaxpy (int n, FLOAT alpha, cl_mem x,
                                     int incx, cl_mem y, int incy)
{
	static cll_Program* prog = 0;
    int nbrCtas;
    int elemsPerCta;
    int threadsPerCta;

    /* early out if nothing to do */

	CL cl(true);
	cl.waitForKernelsToFinish();
	clock_saxpy_cpu.begin();

    if ((alpha == 0.0f) || (n <= 0)) {
        return;
    }

    /* Currently, the overhead for using textures is high. Do not use texture
     * for vectors that are short, or those that are aligned and have unit
     * stride and thus have nicely coalescing GLDs.
     */

	if (prog == 0) {
		string path(CL_SOURCE_DIR);
		path = path + "/saxpy.cl";

        string double_option;
#ifdef DOUBLE
        double_option = " -DDOUBLE ";
#else
        double_option = "";
#endif

        string graphic_libs_dir = getenv("GRAPHIC_LIBS_HOME");
        string options = double_option + "-I" + graphic_libs_dir +
"/include";
        cl.setCompilerOptions(options.c_str());

		cll_Program& progr = cl.addProgramR(path.c_str());
		prog = &progr;
	}

	cll_Kernel kern = prog->addKernel("saxpy_gld_main");

	kern.setArg(n, 0);
	kern.setArg(alpha, 1);
	kern.setArg(x, 2);
	kern.setArg(incx, 3);
	kern.setArg(y, 4);
	kern.setArg(incy, 5);

	size_t global = (size_t) n;
	size_t local = cl.getMaxWorkSize(kern.getKernel());

	clock_saxpy_gpu.begin();
    cl_event exec = kern.exec(1, &global, &local);

	cl.waitForKernelsToFinish();
	clock_saxpy_gpu.end();
	clock_saxpy_cpu.end();
}



/* This file contains the implementation of the BLAS-1 function sdot */

#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <stdio.h>
#include <limits.h>
#include <math.h>
#include <array_opencl_1d.h>
#include <timege.h>

#include "cl.h"
#include "float_type.h"

using namespace std;

extern GE::Time clock_sdot_cpu;
extern GE::Time clock_sdot_gpu;

#define CUBLAS_SDOT_LOG_THREAD_COUNT    (7)
#define CUBLAS_SDOT_THREAD_COUNT        (1 << CUBLAS_SDOT_LOG_THREAD_COUNT)
#define CUBLAS_SDOT_CTAS                (80)


/*
 * FLOAT 
 * sdot (int n, const FLOAT *x, int incx, const FLOAT *y, int incy)
 *
 * computes the dot product of two single precision vectors. It returns the 
 * dot product of the single precision vectors x and y if successful, and
 * 0.0f otherwise. It computes the sum for i = 0 to n - 1 of x[lx + i * 
 * incx] * y[ly + i * incy], where lx = 1 if incx >= 0, else lx = 1 + (1 - n)
 * *incx, and ly is defined in a similar way using incy.
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
 * returns single precision dot product (zero if n <= 0)
 *
 * Reference: http://www.netlib.org/blas/sdot.f
 *
 * Error status for this function can be retrieved via cublasGetError().
 *
 * Error Status
 * ------------
 * CUBLAS_STATUS_NOT_INITIALIZED  if CUBLAS library has nor been initialized
 * CUBLAS_STATUS_EXECUTION_FAILED if function failed to execute on GPU
 */


//__host__ FLOAT cublasSdot (int n, const FLOAT* x, int incx,
                                  //const FLOAT* y, int incy)

FLOAT cublasSdot (int n, cl_mem x, int incx,
                                  cl_mem y, int incy)
{
	static cll_Program* prog = 0;
    FLOAT *devPtrT;
    int nbrCtas;
    int threadsPerCta;
    FLOAT dot = 0.0f;
    int i;
	//ArrayOpenCL1D<FLOAT> tx(nbrCtas); // one per block
	static ArrayOpenCL1D<FLOAT>* tx;

	CL cl(true);
	cl.waitForKernelsToFinish();
	clock_sdot_cpu.begin();

    if (n < CUBLAS_SDOT_CTAS) {
         nbrCtas = n; // nb blocks
         threadsPerCta = CUBLAS_SDOT_THREAD_COUNT;
    } else {
         nbrCtas = CUBLAS_SDOT_CTAS;
         threadsPerCta = CUBLAS_SDOT_THREAD_COUNT;
    }

    /* early out if nothing to do */
    if (n <= 0) {
        return dot;
    }

    /* allocate memory to collect results, one per CTA */
	// perhaps inefficient use of ArrayCuda1D?  since allocated each time//
	//ArrayOpenCL1D<FLOAT> tx(nbrCtas); // one per block
	//tx.setTo(0.);
	//tx.copyToDevice();

	if (prog == 0) {
		string path(CL_SOURCE_DIR);
		path = path + "/sdot.cl";

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
		tx = new ArrayOpenCL1D<FLOAT>(nbrCtas);
	}

	cll_Kernel kern = prog->addKernel("sdot_gld_main");

	kern.setArg(n, 0);
	kern.setArg(x, 1);
	kern.setArg(incx, 2);
	kern.setArg(y, 3);
	kern.setArg(incy, 4);
	kern.setArg(tx->getDevicePtr(), 5);

	size_t global = nbrCtas * CUBLAS_SDOT_THREAD_COUNT;
	size_t local = CUBLAS_SDOT_THREAD_COUNT;

	clock_sdot_gpu.begin();
    cl_event exec = kern.exec(1, &global, &local);

	cl.waitForKernelsToFinish();
	clock_sdot_gpu.end();

    /* Currently, the overhead for using textures is high. Do not use texture
     * for vectors that are short, or those that are aligned and have unit
     * stride and thus have nicely coalescing GLDs.
     */

	tx->copyToHost();

    for (i = 0; i < nbrCtas; i++) {
        dot += (*tx)(i);
    }

	clock_sdot_cpu.end();

    return dot;
}
//----------------------------------------------------------------------


#ifndef ISPH_GPUBITONICSORT_H
#define ISPH_GPUBITONICSORT_H

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include "vec.h"

#include <cmath>
#include <iostream>


namespace isph
{
class GpuBitonicSort
{
public:
	GpuBitonicSort(cl_context GPUContext,
				   cl_command_queue CommandQue,
  				   unsigned int numElements);
	~GpuBitonicSort();

	void sort(cl_mem d_keys,
			  cl_mem d_values);
    
private:
    cl_context cxGPUContext;            // OpenCL context
    cl_command_queue cqCommandQueue;    // OpenCL command que 
    cl_kernel ckBitonicSortStep;		// OpenCL kernels
    cl_program cpProgram;               // OpenCL program

	unsigned int mNumElements;
	unsigned long  mNumElementsP2;     // Closest power of 2 above  number of elements to be sorted
	unsigned int  mP2Exponent;         // Exponent of closest power of 2 
	unsigned int  mNumPhases;          // Number of phases in bitonic sorting
    unsigned int  mNumThreads;         // Emulate number of threads
    unsigned int  mMaxVal;             // Padding value

	void bitonicStep(cl_mem d_keys,
			         cl_mem d_values, 
			        unsigned int phase, 
					unsigned int step, 
					unsigned int inv, 
					unsigned int ssize, 
					unsigned int stride);

	inline unsigned int iLog2(unsigned long value)
    {
      unsigned int l = 0;
      while( (value >> l) > 1 ) ++l;
      return l;
    }

};
}
#endif

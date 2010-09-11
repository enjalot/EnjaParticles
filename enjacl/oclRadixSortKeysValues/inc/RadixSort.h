/*
* Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
*
* NVIDIA Corporation and its licensors retain all intellectual property and 
* proprietary rights in and to this software and related documentation.
* Any use, reproduction, disclosure, or distribution of this software
* and related documentation without an express license agreement from 
* NVIDIA Corporation is strictly prohibited.
* 
* Please refer to the applicable NVIDIA end user license agreement (EULA) 
* associated with this source code for terms and conditions that govern 
* your use of this NVIDIA software.
* 
*/
#ifndef _RADIXSORT_H_
#define _RADIXSORT_H_

#if defined (__APPLE__) || defined(MACOSX)
    #include <OpenCL/opencl.h>
#else
    #include <CL/opencl.h>
#endif 
#include "Scan.h"

class RadixSort
{
public:
	RadixSort(cl_context GPUContext,
		      cl_command_queue CommandQue,
			  unsigned int maxElements, 
			  const char *path,
			  const int ctaSize,
			  bool keysOnly);
	RadixSort() {}
	~RadixSort();

	void sort(cl_mem d_keys,
			  unsigned int *values, 
			  unsigned int  numElements,
			  unsigned int  keyBits);

private:
	cl_context cxGPUContext;             // OpenCL context
    cl_command_queue cqCommandQueue;     // OpenCL command que 
    cl_program cpProgram;                // OpenCL program
	cl_mem d_tempKeys;                   // Memory objects for original keys and work space
	cl_mem mCounters;                    // Counter for each radix
	cl_mem mCountersSum;                 // Prefix sum of radix counters
	cl_mem mBlockOffsets;                // Global offsets of each radix in each block
    cl_kernel ckRadixSortBlocksKeysOnly; // OpenCL kernels
	cl_kernel ckFindRadixOffsets;
	cl_kernel ckScanNaive;
	cl_kernel ckReorderDataKeysOnly;

	int CTA_SIZE; // Number of threads per block
    static const unsigned int WARP_SIZE = 32;
	static const unsigned int bitStep = 4;

	unsigned int  mNumElements;     // Number of elements of temp storage allocated
    unsigned int *mTempValues;      // Intermediate storage for values
         
	Scan scan;

	void radixSortKeysOnly(cl_mem d_keys, unsigned int numElements, unsigned int keyBits);
	void radixSortStepKeysOnly(cl_mem d_keys, unsigned int nbits, unsigned int startbit, unsigned int numElements);
	void radixSortBlocksKeysOnlyOCL(cl_mem d_keys, unsigned int nbits, unsigned int startbit, unsigned int numElements);
	void findRadixOffsetsOCL(unsigned int startbit, unsigned int numElements);
	void scanNaiveOCL(unsigned int numElements);
	void reorderDataKeysOnlyOCL(cl_mem d_keys, unsigned int startbit, unsigned int numElements);
};
#endif

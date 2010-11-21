#ifndef _COMPACTIFY_CL_
#define _COMPACTIFY_CL_

#include "bank_conflicts.h"

#define T int


//----------------------------------------------------------------------
#if 0
__kernel void compactifySub3Kernel(__global int* output)
#else
__kernel void compactifySub3Kernel(
			__global int* processorOffsets, 
			__global int* sum_temp, // store temporary sums 
			int offset_size, 
			int accu_size)
			//int nb)
#endif
// Compactify an array
// (0,0,1,2,0,7) ==> (2,1,7,0,0,0)  
// order is not important
{

	// count: number of valid elements for each block
	// assume 32 threads per block
	int bid = get_group_id(0);
	int block_size = get_local_size(0);

	//int tid = get_global_id(0);
	//if (tid >= offset_size) return;

	int lid = get_local_id(0);

#if 1
	// in each block
	int s = sum_temp[bid];   // sum_temp: 16 elements

	// update processorOffsets (2048 elements)
	processorOffsets[bid*block_size+lid] += s;
#endif

	return;
}
//----------------------------------------------------------------------

#endif

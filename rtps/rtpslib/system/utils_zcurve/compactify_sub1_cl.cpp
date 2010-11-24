#ifndef _COMPACTIFY_CL_
#define _COMPACTIFY_CL_

#include "bank_conflicts.h"

#define T int


//----------------------------------------------------------------------
int sumReduce(__local int* s_data, int nb_elem)
{
	int lid = get_local_id(0);
	int sum;

	barrier(CLK_LOCAL_MEM_FENCE);

// reduce and sum
// typical in GPU computings

	#if 1
	for (int i=(nb_elem>>1); i>0; (i>>=1)) {
    	if (lid < i) {
        	s_data[lid] = s_data[lid] + s_data[i+lid];
			barrier(CLK_LOCAL_MEM_FENCE);
    	}
	}
	#endif
	
	sum = s_data[0];
	return sum;
}
//----------------------------------------------------------------------
#if 0
__kernel void compactifyArrayKernel(__global int* output)
#else
__kernel void compactifySub1Kernel(
            __global int* input,
			__global int* processorCounts, 
			int nb)
#endif
// Compactify an array
// (0,0,1,2,0,7) ==> (2,1,7,0,0,0)  
// order is not important
{
	// count: number of valid elements for each block
	int bid = get_group_id(0);
	int block_size = get_local_size(0);

	int tid = get_global_id(0);
	//if (tid >= nb) return;

	int lid = get_local_id(0);
	//int nb_blocks = get_num_groups(0);

	__local int count_loc[512]; // avoid bank conflicts: extra memory


	int count = 0;

	// case where chunk = block_size
	int in = input[block_size*bid+lid];
	if (in != 0) count++;

	count_loc[lid] = count;
	int count_sum;

	barrier(CLK_LOCAL_MEM_FENCE);
	count_sum = sumReduce(count_loc, block_size);
	barrier(CLK_LOCAL_MEM_FENCE);

	// total number of valid entries in block bid
	processorCounts[bid] = count_sum; // global access

	return;
}
//----------------------------------------------------------------------

#endif

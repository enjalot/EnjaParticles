#ifndef _COMPACTIFY_CL_
#define _COMPACTIFY_CL_

#include "bank_conflicts.h"

#define BLOCK_SIZE 64

#define T int

//----------------------------------------------------------------------
// non-efficient
int compactSIMD(__global int* a, __local int* result, __local int* cnt)
{
	// compact a single block
	// Could compact warps separately, but then I need a processorOffset for 
	// each warp as opposed to each block!
	int count = 0;
	int lid = get_local_id(0);
	int block_size = get_local_size(0);
	if (lid == 0) {
		for (int i=0; i < block_size; i++) {
			if (a[i] != 0) {
				result[count] = a[i];
				count++;
			}
		}
		*cnt = count;
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	return *cnt;
}
//----------------------------------------------------------------------
#if 0
__kernel void compactifyDownKernel(__global int* output)
#else
__kernel void compactifyDownKernel(
            __global int* output, 
            __global int* input,
			__global int* processorCounts, 
			__global int* processorOffsets, 
			int nb)
#endif
// Compactify an array
// (0,0,1,2,0,7) ==> (2,1,7,0,0,0)  
// order is not important
{

	// count: number of valid elements for each block
	// assume 32 threads per block
	int block_size = get_local_size(0);

	int tid = get_global_id(0);
	int lid = get_local_id(0);
	int bid = get_group_id(0);
	int nb_blocks = get_num_groups(0);



	//output[bid] = processorOffsets[bid]; // 32,000 max
	return;

	// BLOCK_SIZE 128
	__local int b[128]; // *2 not required. 
	__local int cnt[1];

	// each block considers a section of input array
	// number of elements treated per block
	int chunk = nb/nb_blocks;
	if (chunk*nb_blocks != nb) chunk++;

	barrier(CLK_LOCAL_MEM_FENCE);

	int j = processorOffsets[bid];
	cnt[0] = 0;
	//output[bid*block_size+lid] = input[bid*block_size+lid];
	//return;

	// FOR DEBUGGING
	//if (bid != 1) return;

	barrier(CLK_LOCAL_MEM_FENCE);

	// case where chunk == block_size
	int numValid = compactSIMD(input+block_size*bid, b, cnt);
	//int numValid = block_size;
	// b[0..s), numValid <-- compactSIMD a(0..S)
	// if numValid == block_size, there is no serialization

	if (lid < numValid) { 
		output[j+lid] = b[lid];
		//output[j+lid] = bid;
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	#if 0
	for (int i=0; i < chunk; i += block_size) {
		//int a = input[block_size*bid+i+lid];
		// compaction of a single block
		// a block has multiple warps 
		//int numValid = compactSIMD(input+i+chunk*bid, b, cnt);
		barrier(CLK_LOCAL_MEM_FENCE);
		// b[0..s), numValid <-- compactSIMD a(0..S)
		// if numValid == block_size, there is no serialization

		if (lid < numValid) { 
			output[j+lid] = b[lid];
			//output[j+lid] = bid;
		}
		j = j + numValid;
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	#endif

	return;
}
//----------------------------------------------------------------------

#endif


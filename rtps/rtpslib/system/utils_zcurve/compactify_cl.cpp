#ifndef _COMPACTIFY_CL_
#define _COMPACTIFY_CL_

#include "bank_conflicts.h"

#define BLOCK_SIZE 32

#define T int

//----------------------------------------------------------------------
void  prescan(
    __global T* g_odata, 
	__global T* g_idata, 
	__local T* temp, 
	int n) 
{
	int offset = 1; 
	int lid = get_local_id(0);
	int ai = lid;
	int bi = lid + (n>>1);
	int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
	int bankOffsetB = CONFLICT_FREE_OFFSET(bi);
	temp[ai + bankOffsetA] = g_idata[ai];
	temp[bi + bankOffsetB] = g_idata[bi];
	
	for (int d = n>>1; d > 0; d >>= 1) // build sum in place up the tree 
	{
		barrier(CLK_LOCAL_MEM_FENCE);
		if (lid < d) {
			int ai = offset*(2*lid+1)-1;
			int bi = offset*(2*lid+2)-1;
			ai += CONFLICT_FREE_OFFSET(ai);
			bi += CONFLICT_FREE_OFFSET(bi);
			temp[bi] += temp[ai];
		}
		offset <<= 1;
	}

	if (lid == 0) {
		temp[n-1+CONFLICT_FREE_OFFSET(n-1)] = 0;
	}

	for (int d=1; d < n; d <<= 1) {
		offset >>= 1;
		barrier(CLK_LOCAL_MEM_FENCE);

		if (lid < d) {
			int ai = offset*(2*lid+1)-1;
			int bi = offset*(2*lid+2)-1;
			ai += CONFLICT_FREE_OFFSET(ai);
			bi += CONFLICT_FREE_OFFSET(bi);

			T t = temp[ai];
			temp[ai] = temp[bi];
			temp[bi] += t;
		}
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	g_odata[ai] = temp[ai + bankOffsetA];
	g_odata[bi] = temp[bi + bankOffsetB];
}
//----------------------------------------------------------------------
// non-efficient
int compactSIMD(__global int* a, __local int* result, __local int* cnt)
{
	int count = 0;
	int lid = get_local_id(0);
	if (lid == 0) {
		for (int i=0; i < BLOCK_SIZE; i++) {
			if (a[i] != 0) {
				result[count] = a[i];
				count++;
			}
		}
		*cnt = count;
	}
	return *cnt;
}
//----------------------------------------------------------------------
#if 0
__kernel void compactifyArrayKernel(__global int* output)
#else
__kernel void compactifyArrayKernel(
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

#if 1
	int tid = get_global_id(0);
	int lid = get_local_id(0);
	int bid = get_group_id(0);
	int nb_blocks = get_num_groups(0);
	int block_size = get_local_size(0);

	__local int count_loc[BLOCK_SIZE+5]; // avoid bank conflicts: extra memory
	__local int prefix_loc[BLOCK_SIZE];
	__local int b[BLOCK_SIZE]; // *2 not required. 
	__local int cnt[1];



	//prescan(output, input, count_loc, nb);  // seems to work

	// each blocks considers a section of input array
	// number of elements treated per block
	int chunk = nb/nb_blocks;
	if (chunk*nb_blocks != nb) chunk++;

	// phase 1
	// figure out offsets
	int count = 0;

	for (int i=0; i < chunk; i += block_size) {
		int in = input[block_size*bid+i+lid];
		if (in != 0) {
			count++;
		}
	}

	count_loc[lid] = count;

	// count: number of valid elements for each block
	// assume 32 threads per block
	if (block_size != 32) return;

	// HOW DOES IT WORK IF THERE ARE TWO BLOCKS??? ERROR? 

	barrier(CLK_LOCAL_MEM_FENCE);

	#if 1
	{ count_loc[lid] += count_loc[lid + 16]; }
	{ count_loc[lid] += count_loc[lid +  8]; }
	{ count_loc[lid] += count_loc[lid +  4]; }
	{ count_loc[lid] += count_loc[lid +  2]; }
	{ count_loc[lid] += count_loc[lid +  1]; }
	#endif

	int count_sum = count_loc[0];

	// total number of valid entires in block bid
	//processorCounts[bid] = count_sum; // global access
	processorCounts[bid] = count_loc[0]; // global access

	barrier(CLK_LOCAL_MEM_FENCE);
	//output[bid*block_size+lid] = 0;  // SHOULD NOT BE NEEDED

	// brute force prefix sum
	// all blocks (for now) do same calculation
	// only block 0 and thread 0 does the calculation
	// update global memory

	// Must make sure that every block does this operation, but a single thread in each block
	// otherwise, other blocks will retrieve value from global memory BEFORE it is updated, since
	// I cannot synchronize multiple blocks together
	// So now, each block is accessing the same global memory multiple times. VERY INEFFICIENT!!
	if (lid == 0)
	{
		processorOffsets[0] = 0;
		// very expensive access to global memory
		for (int i=1; i < nb_blocks; i++) {
			processorCounts[i] += processorCounts[i-1];
			processorOffsets[i] = processorCounts[i-1];
		}
	}
	//int jj = processorOffsets[bid];
	int jj = processorCounts[bid];
	//output[bid] = jj; return;

	// these counts are now offsets!

	barrier(CLK_LOCAL_MEM_FENCE); // only acts on a single block (all other threads wait)
	//output[bid] = processorCounts[bid];
	//return;

	
	barrier(CLK_LOCAL_MEM_FENCE);


	//output[bid*block_size+lid] = processorCounts[bid];
	//output[bid*block_size+lid+1] = 2;//bid;
	//return;
	#if 1
	// each block does same prescan
	#if 0
	if (lid == 0) {
		prescan(processorOffsets, processorCounts, count_loc, nb_blocks);
	}
	#endif
	//output[bid] = processorOffsets[bid];
	output[bid] = processorCounts[bid];
	//return;
	#endif

	//processorOffsets[0] = 0; // single block
	barrier(CLK_LOCAL_MEM_FENCE);

	int j = processorOffsets[bid];
	//int j = processorCounts[bid];
	//output[bid] = j; return;
	cnt[0] = 0;

	int numValid = 0;
	for (int i=0; i < chunk; i += block_size) {
		//int a = input[block_size*bid+i+lid];
		// compaction of a single block
		int numValid = compactSIMD(input+i+block_size*bid, b, cnt);
		// b[0..s), numValid <-- compactSIMD a(0..S)
		if (lid < numValid) {
			output[j+lid] = b[lid];
		}
		j = j + numValid;
	}
#endif

	return;
}
//----------------------------------------------------------------------

#endif

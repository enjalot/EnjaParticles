#ifndef _COMPACTIFY_CL_
#define _COMPACTIFY_CL_

#include "bank_conflicts.h"

#define T int

//----------------------------------------------------------------------
#if 0
// more efficient
// do operation on each of n warps. 
int compactSIMDwarp(__global int* a, __local int* result, __local int* bout, __local int* cnt, __local int* cnt_out)
{
	// compact a single block
	// Could compact warps separately, but then I need a processorOffset for 
	// each warp as opposed to each block!
	int count;
	int lid = get_local_id(0);
	int block_size = get_local_size(0);
	int nb_warps = block_size >> 5;

	// assume blocks of 128 threads

	int which_warp = lid >> 5;
	int start = which_warp << 5;
	int warp_id = lid - start;

	if (which_warp == 0) {
		// 1 0 3 0 || 5 2 0 4  ==> 1 3 0 0 || 5 2 4 0 and cnt = 2 3
		// cnt = 2 3 ==> 0 2 (prefix scan sum) 

		// each warp works independently
		for (int i=start; i < (start+32); i++) {
			if (a[i] != 0) {
				result[start+count] = a[i];
				count++;
			}
		}
		cnt[which_warp] = count;
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	barrier(CLK_GLOBAL_MEM_FENCE);

	#if 0
	c[0] = 0;
	c[1] = count[0];
	c[2] = count[0] + count[1];
	c[3] = count[0] + count[1] + count[2];
	#endif

	#if 1
	if (lid == 0) {
		cnt_out[3] = cnt[2] + cnt[1] + cnt[0];
		cnt_out[2] = cnt[1] + cnt[0];
		cnt_out[1] = cnt[0];
		cnt_out[0] = 0;
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	// reconstitute result

	int j = cnt_out[which_warp];
	if (warp_id < cnt[which_warp]) {
		bout[j+warp_id] = result[32*which_warp+warp_id];
	}

	// I have 4 warps, each 
	#endif

	barrier(CLK_LOCAL_MEM_FENCE);
	return cnt_out[3];
}
#endif
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
__kernel void compactifySub4Kernel(__global int* output)
#else
__kernel void compactifySub4Kernel(
			__global int* output, 
			__global int* input, 
			__global int* processorOffsets, 
			int nb)
#endif
// Compactify an array
// (0,0,1,2,0,7) ==> (2,1,7,0,0,0)  
// order is not important
{

	// count: number of valid elements for each block
	// assume 32 threads per block
	int bid = get_group_id(0);
	int block_size = get_local_size(0);

	int tid = get_global_id(0);
	//if (tid >= offset_size) return;

	int lid = get_local_id(0);
	int nb_blocks = get_num_groups(0);

	//__local int count_loc[512]; // avoid bank conflicts: extra memory
	__local int cnt[4]; // 4 warps per block
	__local int cnt_out[4]; // 4 warps per block
	__local int b[128]; // *2 not required. 
	__local int bout[128]; // *2 not required. 

	// phase 1
	// figure out offsets
	int count = 0;

	output[tid] = 0;
	barrier(CLK_LOCAL_MEM_FENCE);

	#if 1
	//count_loc[lid] = count;
	int j = processorOffsets[bid];
	int numValid = compactSIMD(input+block_size*bid, b, cnt);
	if (lid < numValid) {
		output[j+lid] = b[lid]; //numValid; //b[id];
	}
	return;

	// compactSIMD: too expensive!!
	//int numValid = compactSIMDwarp(input+block_size*bid, b, bout, cnt, cnt_out);
	//int numValid = bid;
		//input[tid] = tid;
	//return;



	if (lid < numValid) {
		output[j+lid] = b[lid];
		//output[j+lid] = bout[lid];
	}
	return;
	#endif

	#if 0
	//for (int i=0; i < block_size; i += block_size) {
		//int a = input[block_size*bid+i+lid];
		// compaction of a single block
		// a block has multiple warps 
		int numValid = compactSIMD(input+i+block_size*bid, b, cnt);
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

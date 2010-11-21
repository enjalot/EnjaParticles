#ifndef _COMPACTIFY_CL_
#define _COMPACTIFY_CL_

#include "bank_conflicts.h"

#define T int


//----------------------------------------------------------------------
#if 0
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
#endif
//----------------------------------------------------------------------
void  prescan(
    __global T* g_odata, 
	__global T* g_idata, 
	__global T* temp_sum, 
	__local T* temp, 
	int n) 
{
// executed within a single block with id = get_group_id(0)

	int offset = 1; 
	int lid = get_local_id(0);

	int ai = lid;
	int bi = lid + (n>>1);
	temp[2*lid]   = g_idata[2*lid];
	temp[2*lid+1] = g_idata[2*lid+1];
	
	for (int d = n>>1; d > 0; d >>= 1) // build sum in place up the tree 
	{
		barrier(CLK_LOCAL_MEM_FENCE);
		if (lid < d) {
			int ai = offset*(2*lid+1)-1;
			int bi = offset*(2*lid+2)-1;
			temp[bi] += temp[ai];
		}
		offset <<= 1;
	}

	if (lid == 0) {
		temp_sum[get_group_id(0)] = temp[n-1];
		temp[n-1] = 0;
	}

	for (int d=1; d < n; d <<= 1) {
		offset >>= 1;
		barrier(CLK_LOCAL_MEM_FENCE);

		if (lid < d) {
			int ai = offset*(2*lid+1)-1;
			int bi = offset*(2*lid+2)-1;

			T t       = temp[ai];
			temp[ai]  = temp[bi];
			temp[bi] += t;
		}
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	// ERROR!!
	g_odata[2*lid]   = temp[2*lid];
	g_odata[2*lid+1] = temp[2*lid+1];
	barrier(CLK_LOCAL_MEM_FENCE);
	//g_odata[2*lid] = 2;
	//g_odata[2*lid+1] = 3;
}
//----------------------------------------------------------------------
#if 0
__kernel void compactifySub2Kernel(__global int* output)
#else
__kernel void compactifySub2Kernel(
			__global int* processorCounts, 
			__global int* processorOffsets, 
			__global int* sum_temp, // store temporary sums 
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
	if (tid >= nb) return;

	int lid = get_local_id(0);
	int nb_blocks = get_num_groups(0);

	__local int count_loc[512]; // avoid bank conflicts: extra memory
	__local int cnt[1];
	__local int temp[256]; // at least block_size

	// each blocks considers a section of input array
	// number of elements treated per block
	int chunk = nb/nb_blocks;
	if (chunk*nb_blocks != nb) chunk++;
	//processorOffsets[bid] = chunk;
	//return;

	// for now, chunk*nb_blocks = nb exactly
	//output[bid] = chunk; return;

	// phase 1
	// figure out offsets
	int count = 0;

	// case where chunk = block_size
	//int in = input[block_size*bid+lid];
	//if (in != 0) count++;

	count_loc[lid] = count;

	// I NEED A NEW KERNEL!!

#if 1
	// At this point, I must synchronize between all blocks, 
	// so I need another kernel with processorCounts as input, and processorOffsets
	// as output

	// divide arrays among blocks (even division in this case)
	// n threads can handle 2*n elements
	int nb_sub_blocks = nb / (2*block_size); // perhaps add 1?
	processorOffsets[bid] = nb_sub_blocks;
	// processorCounts: 2k elements (nb blocks)

	int offset = bid*block_size*2; // 2*block_size elements per block for sum
	//processorOffsets[bid] = offset;
	//return;

	if (bid < nb_sub_blocks) {
		// prefix sum scan, block (128 elements) by block
		prescan(processorOffsets+offset, processorCounts+offset, sum_temp, temp, block_size*2);
		// works when output=3
		// Obviously, I am going outside the designated block in prescan
		barrier(CLK_LOCAL_MEM_FENCE);
	}
#endif

	return;
}
//----------------------------------------------------------------------

#endif

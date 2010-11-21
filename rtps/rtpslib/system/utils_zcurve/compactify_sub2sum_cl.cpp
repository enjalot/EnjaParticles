#ifndef _COMPACTIFY_CL_
#define _COMPACTIFY_CL_

#include "bank_conflicts.h"

#define T int

//----------------------------------------------------------------------
void  prescan(
    __global T* g_odata, 
	__global T* g_idata, 
	__local T* temp, 
	int n) 
{
// executed within a single block with id = get_group_id(0)

	int offset = 1; 
	int lid = get_local_id(0);
	int block_size = get_local_size(0);

	if (lid >= (block_size>>1)) return; 

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
	return;
}
//----------------------------------------------------------------------
#if 0
__kernel void sumScanSingleBlock(__global int* output)
#else
__kernel void sumScanSingleBlock(
			__global int* output, 
			__global int* input, 
			int nb)
#endif
// Compactify an array
// (0,0,1,2,0,7) ==> (2,1,7,0,0,0)  
// order is not important
{
	// count: number of valid elements for each block
	// assume 32 threads per block
	int bid = get_group_id(0);
	if (bid > 0) return;

	int lid = get_local_id(0);
	int block_size = get_local_size(0);

	int tid = get_global_id(0);
	if (tid >= nb) return;

	int nb_blocks = get_num_groups(0);

	__local int temp[512]; // avoid bank conflicts: extra memory (< 128)
	if (nb >= 512) return;

	//output[lid] = 0;
	// output: 128 el, need 64 threads (1/2 of the block)
	prescan(output, input, temp, nb);
	return;
}
//----------------------------------------------------------------------

#endif

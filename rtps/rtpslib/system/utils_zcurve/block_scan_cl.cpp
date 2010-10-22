// This software contains source code provided by NVIDIA Corporation.
// Specifically code from the CUDA 2.3 SDK "Particles" sample

#ifndef _DATASTRUCTURES_
#define _DATASTRUCTURES_

#include "cl_macros.h"
#include "cl_structures.h"

//----------------------------------------------------------------------
__kernel void block_scan(
					__global float4*   vars_unsorted,
					__global float4*   vars_sorted, 
		   			__global uint* sort_hashes,
		   			__global uint* sort_indices,
		   			__global uint* cell_indices_start,
		   			__global uint* cell_indices_end,
		   			__global uint* cell_indices,   // (i,j,k,1)
		   			__constant int4* cell_offset, 
		   			__constant struct SPHParams* sphp,
		   			__constant struct GridParams* gp,
					__local  uint* sharedHash   // blockSize+1 elements
			  )
{
// Hash size: (32+26) particles per cell * 4 bytes * (densi + vel + vel) 
//  32  = nb particles in a cell, 26: 1 particle 26 different neihgbor cells
// Total: 58*4*10 = 2320 bytes  (with float3 in cuda: 7 variables)
// maximum of 6 blocks per multi-processor, on the mac (that is fine). 

	// block id
	int lid = get_local_id(0);
	int gid = get_global_id(0);

	// next four lines would not be necessary once blocks are concatenated
	int nb = cell_indices_start[lid];
	if (nb < 0) return;
	nb = cell_indices_end[lid] - nb;
	if (nb <= 0) return;

	// bring local particles into shared memory
	// limit to 32 particles per block
	// this leads to errors
	if (nb > 32) nb = 32;  

	// First assume blocks of size 32

	if (lid < nb) {
		// bring particle data into local memory
		// LINE AAAA (mapping between global index and local index
		// structure of local memory: 
		// (pos, density): 16 bytes per particle
		
		// loc is cast to (float4*)
		// loc[lid]   = pos[index_start + lid];
		// compute density in loc[lid].w
	}

	float local_density = 0.;

	if (gid < 27 && gid != 4) {   // gid == 4: center cell
		for (int i=0; i < 32; i++) {   // max of 32 cells
			int nb = cell_indices_start[lid];
			if (nb < 0) continue;
			nb = cell_indices_end[lid] - nb;
			if (nb <= 0) continue;

			int4 c = cell_offset[gid];
			// bring single particle into local memory + attributes

			barrier(CLK_LOCAL_MEM_FENCE);

			// update density for particles within smoothing sphere
			local_density += .......
		}
	}

	// update global density // WHAT IS THE GLOBAL_ID? SAME AS LINE AAAA
	if (lid < nb) {
		// sorted array
		density(global id) = loc[lid].w; // local density
	}
}
//----------------------------------------------------------------------

#endif

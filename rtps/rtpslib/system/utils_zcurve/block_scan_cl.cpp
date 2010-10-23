// This software contains source code provided by NVIDIA Corporation.
// Specifically code from the CUDA 2.3 SDK "Particles" sample

#ifndef _DATASTRUCTURES_
#define _DATASTRUCTURES_

#include "cl_macros.h"
#include "cl_structures.h"
#include "neighbors.cpp"

//----------------------------------------------------------------------
uint calcGridHash(int4 gridPos, float4 grid_res, bool wrapEdges)
{
	// each variable on single line or else STRINGIFY DOES NOT WORK
	int gx;
	int gy;
	int gz;

	if(wrapEdges) {
		int gsx = (int)floor(grid_res.x);
		int gsy = (int)floor(grid_res.y);
		int gsz = (int)floor(grid_res.z);

//          //power of 2 wrapping..
//          gx = gridPos.x & gsx-1;
//          gy = gridPos.y & gsy-1;
//          gz = gridPos.z & gsz-1;

		// wrap grid... but since we can not assume size is power of 2 we can't use binary AND/& :/
		gx = gridPos.x % gsx;
		gy = gridPos.y % gsy;
		gz = gridPos.z % gsz;
		if(gx < 0) gx+=gsx;
		if(gy < 0) gy+=gsy;
		if(gz < 0) gz+=gsz;
	} else {
		gx = gridPos.x;
		gy = gridPos.y;
		gz = gridPos.z;
	}


	//We choose to simply traverse the grid cells along the x, y, and z axes, in that order. The inverse of
	//this space filling curve is then simply:
	// index = x + y*width + z*width*height
	//This means that we process the grid structure in "depth slice" order, and
	//each such slice is processed in row-column order.

	return (gz*grid_res.y + gy) * grid_res.x + gx; 
}

//----------------------------------------------------------------------
__kernel void block_scan(
					//int nb_grid_points, // nb blocks
					__global float4*   vars_sorted, 
		   			__global uint* cell_indices_start,
		   			__global uint* cell_indices_end,
		   			__global uint* cell_indices_nb,
		   			__global int4* hash_to_grid_index,
		   			__constant int4* cell_offset, 
		   			__constant struct SPHParams* sphp,
		   			__constant struct GridParams* gp,
					__local  float4* loc   
					DUMMY_ARGS
			  )
{
// Hash size: (32+26) particles per cell * 4 bytes * (densi + vel + vel) 
//  32  = nb particles in a cell, 26: 1 particle 26 different neihgbor cells
// Total: 58*4*10 = 2320 bytes  (with float3 in cuda: 7 variables)
// maximum of 6 blocks per multi-processor, on the mac (that is fine). 

	int numParticles = gp->numParticles;

	// block id
	int lid  = get_local_id(0);
	int gid  = get_global_id(0);
	int hash = get_group_id(0);


	// next four lines would not be necessary once blocks are concatenated
	int nb = cell_indices_nb[hash]; // for each grid cell

	cli[lid].x = hash;
	return;
	//cli[gid].y = cell_indices_start[gid];
	//cli[gid].z = cell_indices_end[gid];
	cli[gid].w = nb;
	//cli[gid] = hash_to_grid_index[gid];

	if (nb <= 0) return;

	// First assume blocks of size 32
	// bring local particles into shared memory
	// limit to 32 particles per block
	// this leads to errors
	if (nb > 32) nb = 32;  

	int start = cell_indices_start[hash];

	if (lid < nb) {
		// bring particle data into local memory
		// (pos, density): 16 bytes per particle

		loc[lid]   = pos(start+lid);
		loc[lid].w = 0.0f;     // to store density
	}
	barrier(CLK_LOCAL_MEM_FENCE); // should not be required


	int offset = nb; 
	//float local_density = 0.;

	float local_density;

	for (int i=0; i < 32; i++) {   	// max of 32 particles per cell
		if (lid >= 27) continue; 	// limit to 27 neighbor cells
		if (lid < 27) {
			int4 c = cell_offset[lid];
			c = c + hash_to_grid_index[hash];
			uint cellHash = calcGridHash(c, gp->grid_res, false);
			uint cstart = cell_indices_start[cellHash];
			#if 1
			if (cell_indices_nb[cellHash] <= 0) {   // global access
				// outside smoothing radius
				loc[nb+lid] = (float4)(900., 900., 900., 1.);
			} else {
				//return;
				loc[nb+lid] = pos(cstart+i); // ith particle in cell
			}
			#endif
		}
		//return;

		barrier(CLK_LOCAL_MEM_FENCE);
		
		// UPDATE THE DENSITIES
		float4 ri = (float4)(loc[lid].xyz, 0.);
		for (int j=0; j < 27; j++) {   	// cell 13 is the center
			float4 rj = (float4)(loc[nb+lid].xyz, 0.);
			float4 r = rj-ri;
			float rad = length(r);
				cli[gid].x += 1;
			if (rad < sphp->smoothing_distance) {
				loc[lid].w += sphp->mass * Wpoly6(r, sphp->smoothing_distance, sphp);
			}
		}

		// update density for particles within smoothing sphere
	}
	
	if (lid < nb) {
		int st = cell_indices_start[hash];
		density(st+lid) = loc[lid].w;    // bid is the hash
		clf[st+lid] = loc[lid];
	}
}
//----------------------------------------------------------------------

#endif

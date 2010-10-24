// This software contains source code provided by NVIDIA Corporation.
// Specifically code from the CUDA 2.3 SDK "Particles" sample

#ifndef _DATASTRUCTURES_
#define _DATASTRUCTURES_

#include "cl_macros.h"
#include "cl_structures.h"
#include "neighbors.cpp"

//----------------------------------------------------------------------
int calcGridHash(int4 gridPos, float4 grid_res, bool wrapEdges)
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
					__global float4*   vars_sorted, 
		   			__global int* cell_indices_start,
		   			__global int* cell_indices_nb,
		   			__global int4* hash_to_grid_index,
		   			//__constant int4* cell_offset,  // DOES NOT WORK OK
		   			__global int4* cell_offset, 
		   			__constant struct SPHParams* sphp,
		   			__global struct GridParams* gp,
					__local  float4* locc   
					DUMMY_ARGS
			  )
{
// Hash size: (32+26) particles per cell * 4 bytes * (densi + vel + vel) 
//  32  = nb particles in a cell, 26: 1 particle 26 different neihgbor cells
// Total: 58*4*10 = 2320 bytes  (with float3 in cuda: 7 variables)
// maximum of 6 blocks per multi-processor, on the mac (that is fine). 

	int numParticles = gp->numParticles;
	int num_grid_points = gp->nb_points;

	// tot nb threads = number of particles
	// tot nb blocks = to number of grid cells

	// not related to particles. 32 threads per cell (there are often less
	// than 32 particles per cell
	//int gid  = get_global_id(0);

	// workgrup: lid in [0,31]
	int lid  = get_local_id(0);

	// block id (one block per cell with particles)
	// save value of all threads in this group
	int hash = get_group_id(0);

	// next four lines would not be necessary once blocks are concatenated
	// nb particles in cell with hash
	int nb = cell_indices_nb[hash]; 

	// the cell is empty
	if (nb <= 0) return;

	// First assume blocks of size 32
	// bring local particles into shared memory
	// limit to 32 particles per block
	// this leads to errors
	if (nb > 32) nb = 32;  


	// nb is nb particles in cell

	int start = cell_indices_start[hash];

	// nb threads = nb cells * 32
	// Initialize all particles [start, start+1, ..., start+nb-1]
	// start+lid refers to a particular particle

	if (lid < nb) {
		// bring particle data from central cell into local memory
		locc[lid]   = pos(start+lid);
		locc[lid].w = 0.0f;     // to store density
	}

	barrier(CLK_LOCAL_MEM_FENCE); // should not be required

	// same value for 32 threads
	// index of central cell
	int4 c = hash_to_grid_index[hash];
	int cellHash=-1;
	int cstart=0;

	#if 0
	if (lid < 27) {
	int cellHash = 421;
	int cstart = cell_indices_start[cellHash];
	int cc = (int) cstart;
	vel(start+lid).z = (float) cc; //cstart;
	vel(start+lid).w = (float) cellHash;
	return;
	} else {
		;
	}
	return;
	#endif


	if (lid < 27) { // FOR DEBUGGING
		// index of neighbor cell (including center)
		//c = c + cell_offset[lid]; 
		cellHash = calcGridHash(c, gp->grid_res, false);
		//cellHash = 421;
		// cstart not always correct
		cstart = cell_indices_start[cellHash];
		//vel(start+lid).z = (float) cstart; 		//cstart; 
		//vel(start+lid).w = (float) cellHash; 	//cstart; 
		//pos(start+lid).x = (float) c.x;
		//pos(start+lid).y = (float) c.y;
		// memory problem 
		//pos(start+lid).z = (float) c.z;
		//return;
	} else {
		;
		//vel(start+lid).z = -3.;
	}
	//return;

	barrier(CLK_LOCAL_MEM_FENCE);

	float rho = 0;

	for (int i=0; i < 32; i++) {   	// max of 32 particles per cell
		barrier(CLK_LOCAL_MEM_FENCE);
		if (cellHash < 0) continue; // limit to 27 neighbor cells
		//if (lid >= 27) continue; 	// limit to 27 neighbor cells (ORIG)

		// Bring in single particle from neighboring blocks, one per thread

		// each thread takes care of one neighboring block
		if (cell_indices_nb[cellHash] > i) {   // global access (called 32x)
			locc[nb+lid] = pos(cstart+i); // ith particle in cell
		} else {
			// outside smoothing radius
			locc[nb+lid] = (float4)(900., 900., 900., 1.);
		}

		barrier(CLK_LOCAL_MEM_FENCE);
		
		// UPDATE THE DENSITIES
		float4 ri = (float4)(locc[lid].xyz, 0.);


		for (int j=0; j < 27; j++) {   	// cell 13 is the center
			float4 rj = (float4)(locc[nb+lid].xyz, 0.);
			float4 r = rj-ri;
			float rad = length(r);

			if (rad < sphp->smoothing_distance) {
				// cannot use x,y,z from loc (position and is required)
				locc[lid].w += sphp->wpoly6_coef * sphp->mass * Wpoly6(r, sphp->smoothing_distance, sphp);
			}
		}
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	// The values in local memory appear to be lost!!! HOW!
	
	if (lid < nb) {
	//{
		// position cannot be used (since used for distances)
		// using vel for debugging
		vel(start+lid).y = locc[lid].w;

		// only 1st 8 positions (within first block) have densities with 
		// two values: 1100 and 8100. Do not know where 8100 comes from!
	}

	return;
}
//----------------------------------------------------------------------

#endif

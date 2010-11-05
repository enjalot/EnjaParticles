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
// about 450 out of 512 densities are correct! WHY? 
// Probably because something wrong with threads, blocks, etc. 
// By definition, a particle should have a non-zero density even if it has 
// no neighbors! I must be confusing blocks and particles at some point!
//----------------------------------------------------------------------
__kernel void block_scan(
					__global float4*   vars_sorted, 
		   			__global int* cell_indices_start,
		   			__global int* cell_indices_nb,

		   			//__constant int4* hash_to_grid_index,
		   			__global int4* hash_to_grid_index,

					// __constant is not working
		   			//__constant int4* cell_offset,  // DOES NOT WORK OK
		   			__global int4* cell_offset, 

		   			//__global struct SPHParams* sphp,
		   			__constant struct SPHParams* sphp,

		   			//__global struct GridParams* gp,
		   			__constant struct GridParams* gp,
					__local  float4* locc   
					DUMMY_ARGS
			  )
{
// Hash size: (32+26) particles per cell * 4 bytes * (densi + vel + vel) 
//  32  = nb particles in a cell, 26: 1 particle 26 different neihgbor cells
// Total: 58*4*10 = 2320 bytes  (with float3 in cuda: 7 variables)
// maximum of 6 blocks per multi-processor, on the mac (that is fine). 

	int numParticles = gp->numParticles; // needed for macros
	//int num_grid_points = gp->nb_points;

	// tot nb threads = number of particles
	// tot nb blocks = to number of grid cells

	// not related to particles. 32 threads per cell (there are often less
	// than 32 particles per cell

	// work item thread: lid in [0,31]
	int lid  = get_local_id(0);

	// block id (one block per cell)
	// save value of all threads in this group
	int hash = get_group_id(0);

	// next four lines would not be necessary once blocks are concatenated
	// nb particles in cell with given hash
	int nb = cell_indices_nb[hash]; 


	// the cell is empty
	if (nb <= 0) return;

	// First assume blocks of size 32
	// bring local particles in block into shared memory
	// limit to 32 particles per block
	// this leads to errors
	if (nb > 32) nb = 32;  


	// nb is number of particles in cell
	// nb >= 1

	// maybe start is wrong? 
	// particle position (index) in sorted array
	int start = cell_indices_start[hash];

	// TEMPORARY
	//if (lid < nb) {
		//density(start+lid) = locc[lid].w;
	//}

	// nb threads = nb cells * 32
	// Initialize all particles [start, start+1, ..., start+nb-1]
	// start+lid refers to a particular particle

	if (lid < nb) {
		// bring particle data from central cell into local memory
		locc[lid]   = pos(start+lid);
		locc[lid].w = 0.0f;
	} else { 		// should not be required
		locc[lid] = 0.;
	}

	barrier(CLK_LOCAL_MEM_FENCE); // should not be required

	// same value for 32 threads
	// index of central cell
	int4 c = hash_to_grid_index[hash];
	int cellHash=-1;
	int cstart=0;
	int cnb=0;

	int4 cell = 0;

	barrier(CLK_LOCAL_MEM_FENCE);


	if (lid < 27) { // FOR DEBUGGING
		// index of neighbor cell (including center)
		// perhaps I could define cell_offset[28,29,30,31] 
		// to avoid if statement?
		cell = c + cell_offset[lid]; 

		// check whether cellHash is valid? It is in principle
		// if fluid is always off by 2-3 cells from the boundary. 
		cellHash = calcGridHash(cell, gp->grid_res, false);

		// cstart not always correct
		// list of particles in cell with hash cellHash
		cstart = cell_indices_start[cellHash];

		// one values for each thread in warp (first 27)
		cnb = cell_indices_nb[cellHash];

	} else {
		return;
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	// ri only makes sense for particles 0 through nb-1
	//   only if lid < nb
	float4 ri = (float4)(locc[lid].xyz, 0.);

	// accumulate data from neighboring cells (lid in [0,26])
	//locc[nb+lid] = (float4) (0., 0., 0., 0.);

	float rho = 0.;

	for (int i=0; i < 32; i++) {   	// max of 32 particles per cell
	// since cnb < 32, how can density be higher when using cnb? 

	// there are cnb particles per cell
	// cnb is the number of neighbors in each cell
	// each thread treats a different neighbor, which has a different number
	// of points

		// neighbor cell lid positions loaded to shared memory
		locc[nb+lid] = (float4)(900., 900., 900., 1.);
		barrier(CLK_LOCAL_MEM_FENCE);

		// Bring in single particle from neighboring blocks, one per thread

		// each thread takes care of one neighboring block
		// densities are 1000 and 5000. WHY? WHY? 
		// bring data from global memory to shared memory
		if (lid < 27) {  // only 27 neighbors
			// next statement has an effect
			if (i < cnb) {   // global access (called 32 times)
				locc[nb+lid] = pos(cstart+i); // ith particle in cell
			} 
		}

		barrier(CLK_LOCAL_MEM_FENCE);
		
		// go over single particle read in from  neighboring cells
		// loop over neighboring cell, consider single particle in each cell
		// including the center cell

		// This loop is very expensive! Even if empty!
		//float acoef = sphp->wpoly6_coef;

		for (int j=0; j < 27; j++) {   	// cell 13 is the center
		#if 1
			if (lid >= nb) continue;
			float4 rj = (float4)(locc[nb+j].xyz, 0.); // CORRECT LINE????
			float4 r = rj-ri;
			float rad = length(r); // users sqrt without a need

			if (rad < sphp->smoothing_distance) {
				// cannot use x,y,z from loc (position and is required)
				// This line accounts for 70 ms!!! Without it, time is 15 ms
				//locc[lid].w += 1.;
				rho += Wpoly6_glob(r, sphp->smoothing_distance);
				//locc[lid].w += Wpoly6_glob(r, sphp->smoothing_distance);
			}
		#endif
		}
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	// The values in local memory appear to be lost!!! HOW!
	//return;

	
	if (lid < nb) {
		density(start+lid) = rho * sphp->wpoly6_coef * sphp->mass;
		//density(start+lid) = locc[lid].w * sphp->wpoly6_coef * sphp->mass;
	}

	return;
}
//----------------------------------------------------------------------

#endif

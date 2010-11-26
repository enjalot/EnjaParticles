// This software contains source code provided by NVIDIA Corporation.
// Specifically code from the CUDA 2.3 SDK "Particles" sample

#ifndef _DATASTRUCTURES_
#define _DATASTRUCTURES_

#include "cl_macros.h"
#include "cl_structures.h"
#include "neighbors.cpp"
#include "sum_cl.cpp"
#include "bank_conflicts.h"
#include "get_indices_cl.cpp"
//#include "block_scan_one_warp_multi_warp_cl.cpp"

float4 int2float(int4 i4)
{
	return (float4)(i4.x,i4.y,i4.z,i4.w);
}
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
// Work of a single warp. 
void block_scan_one_warp(
					int warp_id, 
					//__global int4* cell_compact, 
					int4 cell_compact, 
					__global float4*   vars_sorted, 
		   			__global int* cell_indices_start,
		   			__global int* cell_indices_nb,

		   			__global int4* hash_to_grid_index,

		   			//__constant int4* cell_offset, 
		   			//__global int4* cell_offset, 
		   			__local int4* cell_offset, 
		   			//__constant struct CellOffsets* cell_offset,  

		   			__constant struct SPHParams* sphp,
		   			__constant struct GridParams* gp,
					//int hash,
					__local  float4* locc   
					DUMMY_ARGS
			  )
{
	// next four lines would not be necessary once blocks are concatenated
	// nb particles in cell with given hash

	int hash  = cell_compact.x;  // grid cell
	int nb    = cell_compact.y;  // grid cell
	int start = cell_compact.z;
	// advantage (??) of nb32=32 : warp alignment in shared memory
	int nb32 = 32; // first 32 elements of shared memory: center particles

	// the cell is empty
	// cell can not be empty since am using compacted array
	//if (nb <= 0) return;

	// First assume blocks of size 32
	// bring local particles in block into shared memory
	// limit to 32 particles per block
	// this leads to errors
	// This cannot happen if I break up blocks that are larger than 32
	// if (nb > 32) nb = 32;  

// Hash size: (32+26) particles per cell * 4 bytes * (densi + vel + vel) 
//  32  = nb particles in a cell, 26: 1 particle 26 different neihgbor cells
// Total: 58*4*10 = 2320 bytes  (with float3 in cuda: 7 variables)
// maximum of 6 blocks per multi-processor, on the mac (that is fine). 

	// work item thread: lid in [0,31]
	int lid  = get_local_id(0);
	int numParticles = gp->numParticles;

	// First value is 5. Should be 6. Where did it go? 
	if (lid < nb) {
		density(start+lid) = gp->expo.x;
	}
	return;

	//density(start+lid) = nb;
	//return;

	// this updates every values of density array
	//density(start+lid) = 2;
	//return;

	// I am assuming that continuous global ids correspond 
	// to continuous local ids
	// warp_id = 0,..,nb_warps-1
	lid = lid - (warp_id<<5); // in [0,31]

	//if (lid > 31) density(start) = lid;
	//return;

	// each cell has its area in shared memory
	if (lid < nb) { // serialized since only 8 particles per cell
		// bring particle data from central cell into local memory
		locc[lid]   = pos(start+lid);
		locc[lid].w = 0.0f;
	} else { 		// should not be required
		locc[lid] = 0.0f;
	}

	barrier(CLK_LOCAL_MEM_FENCE); // should not be required

	// same value for 32 threads
	// get indices of central cell
	//int4 c = hash_to_grid_index[hash];

	int4 c = get_indices(hash, gp->expo); // only slightly faster
	density(start+lid) = hash;
	pos(start+lid) = int2float(gp->expo); 
	return;

	int cellHash=-1;
	int cstart=0;
	int cnb=0;

	int4 cell = (int4) (0,0,0,0);
	pos(start+lid) = (float4)(0., 0., 0., 0.); // not required

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
		// NEED TO combine (start and nb into an int2)
		cstart = cell_indices_start[cellHash]; // global access

		// one value for each thread in warp (first 27)
		cnb = cell_indices_nb[cellHash];  // global access
		density(start+lid) = (float) cellHash;
	//	pos(start+lid) = (float4)(1.,2.,3.,4.);
		pos(start+lid) = int2float(c); // WRONG
	} else {
		//return;
		cnb = 0;
		cstart = 0;
		cellHash = 0;
		//pos(start+lid) = (float4)(-1.,2.,3.,4.);
	}
	return;

	// start: particle starting position
	//clf[start+lid] = int2float(cell_offset[lid]);
	//cli[start+lid] = (int4)(1,2,3,4); //int2float(c);
    //return;

	barrier(CLK_LOCAL_MEM_FENCE);

	// ri only makes sense for particles 0 through nb-1
	//   only if lid < nb
	float4 ri = (float4)(locc[lid].xyz, 0.);

	// accumulate data from neighboring cells (lid in [0,26])

	float rho = 0.;

	// if there are 8 neighbors, I should only loop 8 times!
	for (int i=0; i < 8; i++)
	{   		// max of 8 particles per cell

		locc[nb32+lid] = (float4)(900., 900., 900., 1.);
		//barrier(CLK_LOCAL_MEM_FENCE);

// ERROR BETWEEN HERE AND AAAA
		// Bring in single particle from neighboring blocks, one per thread

		{
			// if: no influence on speed, affects results
		 return; //no bug
			if (i < cnb) 
			{
		//return; // Bug
				locc[nb32+lid] = pos(cstart+i); // ith particle in cell
			} 
		}
			return;

		barrier(CLK_LOCAL_MEM_FENCE);
		
		{
				float4 r;
				float rho1=0;
				float rho2=0;
#if 1
			for (int j=0; j < 27; j++)
			{ 
				// no need to zero out .w component since I am not using it
				// and it is initialized to zero
				float4 rj = locc[nb32+j]; // CORRECT LINE????
				float4 r = rj-ri;

				// put the check for distance INSIDE the routine. 
				// much faster then check outside the routine. 
				rho += Wpoly6_glob(r, sphp->smoothing_distance);
				rho = 2;
			}
#endif
		}
	}

#endif

	barrier(CLK_LOCAL_MEM_FENCE);

	if (lid < nb) {
		density(start+lid) = rho * sphp->wpoly6_coef * sphp->mass;
		//density(start+lid) = rho;
		//density(start+lid) = locc[lid].w;
	}

	return;
}
//----------------------------------------------------------------------

//----------------------------------------------------------------------
// about 450 out of 512 densities are correct! WHY? 
// Probably because something wrong with threads, blocks, etc. 
// By definition, a particle should have a non-zero density even if it has 
// no neighbors! I must be confusing blocks and particles at some point!

// Warp size is now 64. 
//----------------------------------------------------------------------
__kernel void block_scan(
					__global float4*  vars_sorted, 
		   			__global int*     cell_indices_start,
		   			__global int*     cell_indices_nb,

		   			//__constant int4* hash_to_grid_index,
		   			__global int4*    hash_to_grid_index,

					// __constant is not working
		   			//__constant int4* cell_offset,  // DOES NOT WORK OK
		   			__global int4*    cell_offset, 
		   			//__constant struct CellOffsets* cell_offset,  

		   			//__global struct SPHParams* sphp,
		   			__constant struct SPHParams* sphp,

		   			//__global struct GridParams* gp,
		   			__constant struct GridParams* gp,
					__global int4* cell_compact, 
					__constant struct GPUReturnValues* rv, 
					__local  float4* locc   
					DUMMY_ARGS
			  )
{
	int lid  = get_local_id(0);
	int nb_warps = get_local_size(0) >> 5;
	int warp_id = lid >> 5; // should equal 0, ..., nb_warps-1
	int bid = get_group_id(0);

	__local int4 l_cell_offset[32];
	if (lid < 27) {
		l_cell_offset[lid] = cell_offset[lid];
		// shift is a constant variable
		//l_cell_offset[lid] = gp->shift[lid];
	}

	barrier(CLK_LOCAL_MEM_FENCE); // should not be required

	// hash: grid cell number
	// SOMETHING WRONG WITH HASH

	// if group has 32 threads (=single warp), hash = group number (nb_warp=1, warp=0)
	// if group has 64 threads (2 warps), 32 threads still handle single group
	int pt = warp_id + nb_warps*bid;
	int numParticles = gp->numParticles;
	//density(pt) = pt; // seems ok: pt = [0,4806)
	//return;

	int4 compact = cell_compact[pt];
	//density(pt) = compact.y;
	//return;

	// I am now having 2-4 warps per block, each warp handles one cell
	// I should also have each warp handle two cells (possibly use 
	// more registers, but that does not matter)
	// if 2-4 warps per block, it helps WHY? (no synchronization, but less
	// calls to kernel? Not sure). Each warp still has equal nb of 
	// memory accesses. 
	// Gain on Fermi should be much greater than gain on mac 
	// (due to L1/L2 cache)
	// one warp with multiple cells: more work per thread. Good if less memory
	// transfer

	// warp deals with one cell
	block_scan_one_warp(
					warp_id, // which warp: 0,..., nb_warps-1
					compact,
					vars_sorted, 
		   			cell_indices_start,
		   			cell_indices_nb,
		   			hash_to_grid_index,
		   			l_cell_offset, 
		   			sphp,
		   			gp,
					locc+warp_id*64 // unit is float4
					ARGS);
	return;


	barrier(CLK_LOCAL_MEM_FENCE); // should not be required
}
//----------------------------------------------------------------------

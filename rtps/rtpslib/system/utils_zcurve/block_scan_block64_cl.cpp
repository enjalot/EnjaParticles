// This software contains source code provided by NVIDIA Corporation.
// Specifically code from the CUDA 2.3 SDK "Particles" sample

#ifndef _DATASTRUCTURES_
#define _DATASTRUCTURES_

#include "cl_macros.h"
#include "cl_structures.h"
#include "neighbors.cpp"
#include "sum_cl.cpp"
#include "bank_conflicts.h"
//#include "block_scan_one_warp_multi_warp_cl.cpp"

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
					int hash,
					__local  float4* locc   
					DUMMY_ARGS
			  )
{
	// next four lines would not be necessary once blocks are concatenated
	// nb particles in cell with given hash
	int nb = cell_indices_nb[hash];  // grid cell
	// advantage (??) of nb32=32 : warp alignment in shared memory
	int nb32 = 32; // first 32 elements of shared memory: center particles


	// the cell is empty
	if (nb <= 0) return;

	// First assume blocks of size 32
	// bring local particles in block into shared memory
	// limit to 32 particles per block
	// this leads to errors
	if (nb > 32) nb = 32;  


// Hash size: (32+26) particles per cell * 4 bytes * (densi + vel + vel) 
//  32  = nb particles in a cell, 26: 1 particle 26 different neihgbor cells
// Total: 58*4*10 = 2320 bytes  (with float3 in cuda: 7 variables)
// maximum of 6 blocks per multi-processor, on the mac (that is fine). 

	// work item thread: lid in [0,31]
	int lid  = get_local_id(0);

	// I am assuming that continuous global ids correspond 
	// to continuous local ids
	int warp = lid >> 5; // should equal 0,...,nb_warps-1
	lid = lid - (warp<<5); // in [0,31]
	//lid = lid % 32;


//----------------------------------

	int numParticles = gp->numParticles; // needed for macros
	int start = cell_indices_start[hash];

	// Time: 2.7ms if call to density and return
	//density(start+lid) = sphp->wpoly6_coef * sphp->mass;
	//density(start+lid) = 1.; // difference with previous line
	//return;
//----------------------------------------------------------------------
	//if (get_group_id(0) > 500) return;

//----------------------------------------------------------------------

	// nb threads = nb cells * 32
	// Initialize all particles [start, start+1, ..., start+nb-1]
	// start+lid refers to a particular particle
 	//clf[start+lid] = ri; //ri; always zero for warp 1

	// each cell has its area in shared memory
	if (lid < nb) {
		// bring particle data from central cell into local memory
		locc[lid]   = pos(start+lid);
		locc[lid].w = 0.0f;
		//cli[start+lid] = warp; //ri;
		//cli[start+lid].y = nb; // nb is number of particles in cell[hash]
		//cli[start+lid].z = hash;
		//cli[start+lid].w = lid;

		// works
		//clf[start+lid] = pos(start+lid);//[lid]; //ri;

		// does not work with 2 warps and beyond: WHY NOT?
		//clf[start+lid] = locc[lid];
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

	//barrier(CLK_LOCAL_MEM_FENCE);


	if (lid < 27) { // FOR DEBUGGING
		// index of neighbor cell (including center)
		// perhaps I could define cell_offset[28,29,30,31] 
		// to avoid if statement?
		cell = c + cell_offset[lid]; 
		//cell = c + cell_offset->offsets[lid]; 

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

	float rho = 0.;

	// if there are 8 neighbors, I should only loop 8 times!
	for (int i=0; i < 8; i++)
	{   		// max of 8 particles per cell

		locc[nb32+lid] = (float4)(900., 900., 900., 1.);
		//barrier(CLK_LOCAL_MEM_FENCE);

		// Bring in single particle from neighboring blocks, one per thread

		{
			// if: no influence on speed, affects results
			if (i < cnb)   // global access (called 32 times)  // 
			{
				locc[nb32+lid] = pos(cstart+i); // ith particle in cell
			} 
		}

		barrier(CLK_LOCAL_MEM_FENCE);
		
		{
				float4 r;
				float rho1=0;
				float rho2=0;
#if 1
			#if 0
				r = locc[nb32+0]-ri; 
				rho1 += Wpoly6_glob(r, sphp->smoothing_distance);
				r = locc[nb32+1]-ri;
				rho2 += Wpoly6_glob(r, sphp->smoothing_distance);
				r = locc[nb32+2]-ri;
				rho1 += Wpoly6_glob(r, sphp->smoothing_distance);
				r = locc[nb32+3]-ri;
				rho2 += Wpoly6_glob(r, sphp->smoothing_distance);
				r = locc[nb32+4]-ri;
				rho1 += Wpoly6_glob(r, sphp->smoothing_distance);
				r = locc[nb32+5]-ri;
				rho2 += Wpoly6_glob(r, sphp->smoothing_distance);
				r = locc[nb32+6]-ri;
				rho1 += Wpoly6_glob(r, sphp->smoothing_distance);
				r = locc[nb32+7]-ri;
				rho2 += Wpoly6_glob(r, sphp->smoothing_distance);
				r = locc[nb32+8]-ri;
				rho1 += Wpoly6_glob(r, sphp->smoothing_distance);
				r = locc[nb32+9]-ri;
				rho2 += Wpoly6_glob(r, sphp->smoothing_distance);
				r = locc[nb32+10]-ri;
				rho1 += Wpoly6_glob(r, sphp->smoothing_distance);
				r = locc[nb32+11]-ri;
				rho2 += Wpoly6_glob(r, sphp->smoothing_distance);
				r = locc[nb32+12]-ri;
				rho1 += Wpoly6_glob(r, sphp->smoothing_distance);
				r = locc[nb32+13]-ri;
				rho2 += Wpoly6_glob(r, sphp->smoothing_distance);
				r = locc[nb32+14]-ri;
				rho1 += Wpoly6_glob(r, sphp->smoothing_distance);
				r = locc[nb32+15]-ri;
				rho2 += Wpoly6_glob(r, sphp->smoothing_distance);
				r = locc[nb32+16]-ri;
				rho1 += Wpoly6_glob(r, sphp->smoothing_distance);
				r = locc[nb32+17]-ri;
				rho2 += Wpoly6_glob(r, sphp->smoothing_distance);
				r = locc[nb32+18]-ri;
				rho1 += Wpoly6_glob(r, sphp->smoothing_distance);
				r = locc[nb32+19]-ri;
				rho2 += Wpoly6_glob(r, sphp->smoothing_distance);
				r = locc[nb32+20]-ri;
				rho1 += Wpoly6_glob(r, sphp->smoothing_distance);
				r = locc[nb32+21]-ri;
				rho2 += Wpoly6_glob(r, sphp->smoothing_distance);
				r = locc[nb32+22]-ri;
				rho1 += Wpoly6_glob(r, sphp->smoothing_distance);
				r = locc[nb32+23]-ri;
				rho2 += Wpoly6_glob(r, sphp->smoothing_distance);
				r = locc[nb32+24]-ri;
				rho1 += Wpoly6_glob(r, sphp->smoothing_distance);
				r = locc[nb32+25]-ri;
				rho2 += Wpoly6_glob(r, sphp->smoothing_distance);
				r = locc[nb32+26]-ri;
				rho += Wpoly6_glob(r, sphp->smoothing_distance);
				rho += rho1 + rho2;

			#else
			for (int j=0; j < 27; j++)
			{ 
				// no need to zero out .w component since I am not using it
				// and it is initialized to zero
				float4 rj = locc[nb32+j]; // CORRECT LINE????
				float4 r = rj-ri;

				// put the check for distance INSIDE the routine. 
				// much faster then check outside the routine. 
				rho += Wpoly6_glob(r, sphp->smoothing_distance);
			}
			#endif
#endif
		}
	}

#endif

	barrier(CLK_LOCAL_MEM_FENCE);

	if (lid < nb) {
		density(start+lid) = rho * sphp->wpoly6_coef * sphp->mass;
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
					__global float4*   vars_sorted, 
		   			__global int* cell_indices_start,
		   			__global int* cell_indices_nb,

		   			//__constant int4* hash_to_grid_index,
		   			__global int4* hash_to_grid_index,

					// __constant is not working
		   			//__constant int4* cell_offset,  // DOES NOT WORK OK
		   			__global int4* cell_offset, 
		   			//__constant struct CellOffsets* cell_offset,  

		   			//__global struct SPHParams* sphp,
		   			__constant struct SPHParams* sphp,

		   			//__global struct GridParams* gp,
		   			__constant struct GridParams* gp,
					__local  float4* locc   
					DUMMY_ARGS
			  )
{
	int numParticles = gp->numParticles; // needed for macros
	if (get_group_id(0) >= gp->nb_points) return;

	int lid  = get_local_id(0);

	__local int4 l_cell_offset[32];
	if (lid < 32) {
		l_cell_offset[lid] = cell_offset[lid];
	}

	//int nb_warps = get_local_size(0) >> 5;
	int nb_warps = get_local_size(0) / 32;
	//int warp = lid >> 5; // should equal 0, ..., nb_warps-1
	int warp = lid / 32; // should equal 0, ..., nb_warps-1
	// 1024 = (32+32)*sizeof(float4) // memory for single warp

	//if (lid == 0) {
	// unit is float4, so 64 and not 64*sizeof(float4) = 1024
	//__local float4* locc1 = locc+warp*64; //(warp << 10);   // *1024; 
	//}
	barrier(CLK_LOCAL_MEM_FENCE); // should not be required

	// hash: grid cell number
	// SOMETHING WRONG WITH HASH
	int hash = warp + nb_warps*get_group_id(0);


	// I am now having 2-4 warps per block, each warp handles one cell
	// I should also have each warp handle two cells (possibly use 
	// more registers, but that does not matter)
	// if 2-4 warps per block, it helps WHY? (no synchronization, but less
	// calls to kernel? Not sure). Each warp still has equal nb memory access. 
	// Gain on Fermi should be much greater than gain on mac 
	// (due to L1/L2 cache)
	// one warp with multiple cells: more work per thread. Good if less memory
	// transfer

	// warp deals with one cell
	block_scan_one_warp(
					vars_sorted, 
		   			cell_indices_start,
		   			cell_indices_nb,
		   			hash_to_grid_index,
		   			//cell_offset, 
		   			l_cell_offset, 
		   			sphp,
		   			gp,
					hash,
					locc+warp*64 // unit is float4
					ARGS);

	// same warp deals with next cell
	#if 0
	block_scan_one_warp(
					vars_sorted, 
		   			cell_indices_start,
		   			cell_indices_nb,
		   			hash_to_grid_index,
		   			//cell_offset, 
		   			l_cell_offset, 
		   			sphp,
		   			gp,
					hash+1, // change the hash for next cell
					locc+warp*64 // unit is float4
					ARGS);
	#endif

	barrier(CLK_LOCAL_MEM_FENCE); // should not be required
}
//----------------------------------------------------------------------

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
		   			//__constant int4* cell_offset,  // DOES NOT WORK OK
		   			__global int4* cell_offset, 
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
	int num_grid_points = gp->nb_points;

	// tot nb threads = number of particles
	// tot nb blocks = to number of grid cells

	int gid  = get_global_id(0);
	//if (gid >= num_grid_points) return;

	// block id
	int lid  = get_local_id(0);
	int hash = get_group_id(0);
	//cli[hash].y = gp->nb_points; // incorrect
	//cli[hash].z = gp->nb_vars; // incorrect


	// next four lines would not be necessary once blocks are concatenated
	int nb = cell_indices_nb[hash]; // for each grid cell

	//cli[hash].x = lid;
	//cli[hash].y= gid;
	//return;
	//cli[hash].y = cell_indices_start[hash];
	//cli[hash].z = cell_indices_end[hash];
	//cli[hash].w = nb;
	//cli[hash] = hash_to_grid_index[hash];

	if (nb <= 0) return;


	// First assume blocks of size 32
	// bring local particles into shared memory
	// limit to 32 particles per block
	// this leads to errors
	if (nb > 32) nb = 32;  

	int start = cell_indices_start[hash];

	#if 1
	if (lid < nb) {
		// bring particle data into local memory
		// (pos, density): 16 bytes per particle

		loc[lid]   = pos(start+lid);
		loc[lid].w = 0.0f;     // to store density
	}
	#endif
	barrier(CLK_LOCAL_MEM_FENCE); // should not be required

	// same value for 32 threads
	int4 c = hash_to_grid_index[hash];
	//cli[hash] = hash_to_grid_index[hash];

	int4 co;
	if (lid < 27) {
		co = (int4) (-1,2,3,4);
		co = cell_offset[lid];  //ORIG
		//cli[hash] = co;
	} else { 
		;
	}

	#if 0
	if (lid == 1) {
		cli[hash] = c;
		cli[hash].w = lid;
		uint cellHash = calcGridHash(c, gp->grid_res, false);
		return;
	} else { 
		return; 
	}
	#endif

	//return;

	#if 1
	if (lid < 27) { // FOR DEBUGGING
	//if (lid == 1) { // FOR DEBUGGING
		//c = c + co;
		//clf[hash] = gp->grid_res;
		if (lid == 5) {
		clf[hash].w = -10.;
		clf[hash].z = (float) (lid+1);  //OUTPUT
		cli[hash] = co;
		return;
		uint cellHash = calcGridHash(c, gp->grid_res, false);
		} else { return; }
		//clf[hash] = (float) cellHash;
		//cli[hash].w = (int) cellHash;
		//return;
	}
	#endif

	for (int i=0; i < 1; i++) {   	// max of 32 particles per cell
		if (lid >= 27) continue; 	// limit to 27 neighbor cells (ORIG)

		//cli[hash] = c; return;

		//int4 cc = c + cell_offset[i];
		//c = c + hash_to_grid_index[hash];
		//c = hash_to_grid_index[hash];
		// What if I am out of bounds of grid?

		// perhaps prestore? 
		//uint cellHash = calcGridHash(c, gp->grid_res, false);
		clf[hash] = gp->grid_res;
		cli[hash] = c;
		uint cellHash = (c.z*(int) gp->grid_res.y + c.y) * (int) gp->grid_res.x + c.x; 
		if ((cellHash >= gp->nb_points) || (cellHash < 0)) {
			cli[hash].w =  -47;
			//cli[hash].x = hash;
			return;
		}
		return;

		//cli[hash] = cellHash; return;
		//cli[hash] = c;

		// cellHash should NOT go beyond gp->nb_points!
		if ((cellHash >= gp->nb_points) || (cellHash < 0)) {
			cli[hash] = c;  
			cli[hash].w = cellHash; // c.z is wrong
			clf[hash].w = (float) lid;
			return;
		}
		// cellHash is out of bounds!
		uint cstart = cell_indices_start[cellHash];
		return;
		#if 1
		if (cell_indices_nb[cellHash] <= 0) {   // global access (called 32x)
			// outside smoothing radius
			//loc[nb+lid] = (float4)(900., 900., 900., 1.);
			loc[nb+lid]= 3;
			return;
		} else {
			return;
			//loc[nb+lid] = pos(cstart+i); // ith particle in cell
			;
		}
		return;
		#endif

		barrier(CLK_LOCAL_MEM_FENCE);
		return;
		
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

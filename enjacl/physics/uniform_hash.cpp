#ifndef _UNIFORM_HASH_H_
#define _UNIFORM_HASH_H_



#include "cl_structures.h"


//----------------------------------------------------------------------
// find the grid cell from a position in world space
// WHY static?
int4 calcGridCell(float4 p, float4 grid_min, float4 grid_delta)
{
	// subtract grid_min (cell position) and multiply by delta
	//return make_int4((p-grid_min) * grid_delta);

	//float4 pp = (p-grid_min)*grid_delta;
	float4 pp;
	pp.x = (p.x-grid_min.x)*grid_delta.x;
	pp.y = (p.y-grid_min.y)*grid_delta.y;
	pp.z = (p.z-grid_min.z)*grid_delta.z;
	pp.w = (p.w-grid_min.w)*grid_delta.w;

	int4 ii;
	ii.x = (int) pp.x;
	ii.y = (int) pp.y;
	ii.z = (int) pp.z;
	ii.w = (int) pp.w;
	return ii;
}

//----------------------------------------------------------------------
uint calcGridHash(int4 gridPos, float4 grid_res, __constant bool wrapEdges)
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
// Calculate a grid hash value for each particle


//  Have to make sure that the data associated with the pointers is on the GPU
//struct GridData
//{
//    uint* sort_hashes;          // particle hashes
//    uint* sort_indexes;         // particle indices
//    uint* cell_indexes_start;   // mapping between bucket hash and start index in sorted list
//    uint* cell_indexes_end;     // mapping between bucket hash and end index in sorted list
//};

//----------------------------------------------------------------------
// comes from K_Grid_Hash
// CANNOT USE references to structures/classes as aruguments!
__kernel void hash(
		   __constant unsigned int	numParticles,
		   __global float4*	  		dParticlePositions,	
		   __global uint* sort_hashes,
		   __global uint* sort_indexes,
		   __constant struct GridParams* cGridParams)
{
	// particle index
	//uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	uint index = get_global_id(0);
	if (index >= numParticles) return;

	// particle position
	float4 p = dParticlePositions[index];

	// get address in grid
	int4 gridPos = calcGridCell(p, cGridParams->grid_min, cGridParams->grid_delta);
	bool wrap_edges = false;
	uint hash = (uint)calcGridHash(gridPos, cGridParams->grid_res, wrap_edges);

	// store grid hash and particle index

	sort_hashes[index] = hash;
	//int pp = (int) ((p.z-cGridParams->grid_min.z)*cGridParams->grid_delta.z);
	//int pp = (int) cGridParams->grid_delta.z;
	int pp = (int) p.x;

	//sort_hashes[index] = (int) cGridParams->grid_res.x;
	//sort_hashes[index] = hash;
	//sort_hashes[index] = pp;
	//int grid_size = get_global_size(0);
	//sort_hashes[index] = numParticles;
	//sort_hashes[index] = grid_size;

	sort_indexes[index] = index;

	//sort_indexes[index] = 6;
}
//----------------------------------------------------------------------


#endif

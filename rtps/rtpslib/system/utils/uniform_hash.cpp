#ifndef _UNIFORM_HASH_H_
#define _UNIFORM_HASH_H_



#include "cl_structures.h"
#include "cl_macros.h"


//----------------------------------------------------------------------
// find the grid cell from a position in world space
// WHY static?
int4 calcGridCell(float4 p, float4 grid_min, float4 grid_inv_delta)
{
    // subtract grid_min (cell position) and multiply by delta
    //return make_int4((p-grid_min) * grid_delta);

    //float4 pp = (p-grid_min)*grid_delta;
    float4 pp;
    pp.x = (p.x-grid_min.x)*grid_inv_delta.x;
    pp.y = (p.y-grid_min.y)*grid_inv_delta.y;
    pp.z = (p.z-grid_min.z)*grid_inv_delta.z;
    pp.w = (p.w-grid_min.w)*grid_inv_delta.w;

    int4 ii;
    ii.x = (int) pp.x;
    ii.y = (int) pp.y;
    ii.z = (int) pp.z;
    ii.w = (int) pp.w;
    return ii;
}

//----------------------------------------------------------------------
uint calcGridHash(int4 gridPos, float4 grid_res, bool wrapEdges
           , __global float4* fdebug,
           __global int4* idebug
		 )
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

	int index = get_global_id(0);
	//fdebug[index] = grid_res;
	//idebug[index] = (int4)(gx,gy,gz,1.);
    //idebug[index] = (gz*grid_res.y + gy) * grid_res.x + gx; 

	// uint(-3) = 0   (so hash is never less than zero)
	// But if particle leaves boundary to the right of the grid, the hash
	// table can go out of bounds and the code might crash. This can happen
	// either if the boundary does not catch the particles or if the courant
	// condition is violated and the code goes unstable. 

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
// CANNOT USE references to structures/classes as arguments!
__kernel void hash(
           __global float4* vars_unsorted,
           __global uint* sort_hashes,
           __global uint* sort_indexes,
           __global uint* cell_indices_start,
           __constant struct GridParams* gp
           , __global float4* fdebug,
           __global int4* idebug
		   )
{
#if 1
    // particle index
    uint index = get_global_id(0);
	// do not use gp->numParticles (since it numParticles changed via define)
	int num = get_global_size(0);
    if (index >= num) return; 


	// initialize to -1 (used in kernel datastructures in build_datastructures_wrap.cpp
	int grid_size = (int) (gp->grid_res.x*gp->grid_res.y*gp->grid_res.z);
	if (index < grid_size) {
		cell_indices_start[index] = 0xffffffff;
	}

    // particle position
    float4 p = unsorted_pos(index); // macro

    // get address in grid
    int4 gridPos = calcGridCell(p, gp->grid_min, gp->grid_inv_delta);
    bool wrap_edges = false;
    uint hash = (uint) calcGridHash(gridPos, gp->grid_res, wrap_edges, fdebug, idebug);

	//idebug[index] = hash;
	//fdebug[index] = p;


    // store grid hash and particle index

    sort_hashes[index] = hash;
    int pp = (int) p.x;

    sort_indexes[index] = index;
#endif

	//idebug[index] = gridPos;
}
//----------------------------------------------------------------------


#endif

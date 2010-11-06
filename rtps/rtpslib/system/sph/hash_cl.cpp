#ifndef _HASH_CL_H_
#define _HASH_CL_H_



#include "cl_structs.h"
#include "cl_macros.h"
#include "cl_hash.h"


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
           int num,
           __global float4* vars_unsorted,
           __global uint* sort_hashes,
           __global uint* sort_indexes,
           //__global uint* cell_indices_start,
           __constant struct GridParams* gp
           DEBUG_ARGS
           //__global float4* fdebug,
           //__global int4* idebug
		   )
{
    // particle index
    uint index = get_global_id(0);
	//int num = get_global_size(0);
    if (index >= num) return;

	// initialize to -1 (used in kernel datastructures in build_datastructures_wrap.cpp
	//int grid_size = (int) (gp->grid_res.x*gp->grid_res.y*gp->grid_res.z);
	//if (index < grid_size) {   // grid_size: 1400
		//cell_indices_start[index] = 0xffffffff; 
	//}

    // particle position
    float4 p = unsorted_pos(index); // macro

    // get address in grid
    //int4 gridPos = calcGridCell(p, gp->grid_min, gp->grid_inv_delta);
    int4 gridPos = calcGridCell(p, gp->grid_min, gp->grid_delta);
    bool wrap_edges = false;
    uint hash = (uint) calcGridHash(gridPos, gp->grid_res, wrap_edges);//, fdebug, idebug);

    // store grid hash and particle index
    sort_hashes[index] = hash;
    //int pp = (int) p.x;

    sort_indexes[index] = index;

    //fdebug[index] = gp->grid_inv_delta;
    //fdebug[index] = (float4)((p.x - gp->grid_min.x) * gp->grid_inv_delta.x, p.x, 0,0);
    clf[index] = (float4)((p.x - gp->grid_min.x) * gp->grid_delta.x, p.x, 0,0);
	cli[index] = gridPos;
    cli[index].w = num;
}
//----------------------------------------------------------------------


#endif

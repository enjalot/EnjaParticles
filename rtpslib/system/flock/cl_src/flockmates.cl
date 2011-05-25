#ifndef _FLOCKMATES_CL_
#define _FLOCKMATES_CL_

//These are passed along through cl_neighbors.h
//only used inside ForNeighbor defined in this file
#define ARGS __global float4* pos, __global int4* flockmates 
#define ARGV pos, flockmates 

#include "cl_macros.h"
#include "cl_structs.h"

//----------------------------------------------------------------------
inline void ForNeighbor(//__global float4*  vars_sorted,
				ARGS,
                Boid *pt,
				uint index_i,
				uint index_j,
				float4 position_i,
	  			__constant struct GridParams* gp,
 	  			__constant struct FLOCKParameters* flockp
                DEBUG_ARGS
				)
{
    int num = flockp->num;
	
	// get the particle info (in the current grid) to test against
	float4 position_j = pos[index_j] * flockp->simulation_scale; 

	float4 r = (position_i - position_j); 
	r.w = 0.f; 
	
    // |r|
	float rlen = length(r);

    // neighbor within the radius?    
    if(rlen <= flockp->search_radius)
    {
        if(index_i != index_j){
	
	        // number of flockmates
            pt->num_flockmates++;

		    if(rlen <= flockp->min_dist){
               pt->num_nearestFlockmates++;
            } 
        }
    }
}


//Contains Iterate...Cells methods and ZeroPoint
#include "cl_neighbors.h"


//--------------------------------------------------------------
// count the number of flockmates 

__kernel void flockmates(
                ARGS,
        		__global int*    cell_indexes_start,
        		__global int*    cell_indexes_end,
	  			__constant struct GridParams* gp,
				__constant struct FLOCKParameters* flockp 
				DEBUG_ARGS
				)
{
    // particle index
	int num = flockp->num;

    int index = get_global_id(0);
    if (index >= num) return;

    float4 position_i = pos[index] * flockp->simulation_scale;

    // Do calculations on particles in neighboring cells
	Boid pt;
	zeroPoint(&pt);

    IterateParticlesInNearbyCells(/*vars_sorted*/ ARGV, &pt, num, index, position_i, cell_indexes_start, cell_indexes_end, gp,/* fp,*/ flockp DEBUG_ARGV);
		
    flockmates[index].x = pt.num_flockmates;
    flockmates[index].y = pt.num_nearestFlockmates;
}

#endif


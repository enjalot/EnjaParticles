#ifndef _RULE_SEPARATION_CL_
#define _RULE_SEPARATION_CL_

//These are passed along through cl_neighbors.h
//only used inside ForNeighbor defined in this file
#define ARGS __global float4* pos, __global float4* separation, __global int4* flockmates 
#define ARGV pos, separation, flockmates 

#include "cl_macros.h"
#include "cl_structs.h"

//----------------------------------------------------------------------
inline void ForNeighbor(
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
	        // positions
	        float4 pj = position_j;
	
	        // setup for Rule 1. Separation
            float4 s = r;       //pi - pj;
	        float  d = rlen;    //length(s);
	
		     if(d <= flockp->min_dist){ 
                s.w = 0.0f;
                s = normalize(s);
                s /= d;
	            pt->separation+= s;        // accumulate the separation vector
	        }
        }
    }
}


//Contains Iterate...Cells methods and ZeroPoint
#include "cl_neighbors.h"


//--------------------------------------------------------------
// compute the separation 

__kernel void rule_separation(
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

    // mymese debbug
    //clf[index] = (float4)(flockp->smoothing_distance,flockp->min_dist, flockp->search_radius, 10.);
	//cli[index] = (int4)(flockp->num,flockp->num,0,0);

    float4 position_i = pos[index] * flockp->simulation_scale;

    // Do calculations on particles in neighboring cells
	Boid pt;
	zeroPoint(&pt);

    IterateParticlesInNearbyCells(/*vars_sorted*/ ARGV, &pt, num, index, position_i, cell_indexes_start, cell_indexes_end, gp,/* fp,*/ flockp DEBUG_ARGV);

    // TODO: finish the computation of the separation
    if(pt.num_nearestFlockmates > 0){
        pt.separation /= (float)pt.num_nearestFlockmates;
        pt.separation.w =0.f;
        pt.separation = normalize(pt.separation);
    }

    separation[index] = pt.separation;

}

#endif


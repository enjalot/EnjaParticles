#ifndef _RULE_COHESION_CL_
#define _RULE_COHESION_CL_

//These are passed along through cl_neighbors.h
//only used inside ForNeighbor defined in this file
#define ARGS __global float4* pos, __global float4* cohesion, __global int4* flockmates 
#define ARGV pos, cohesion, flockmates 

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
	float4 position_j = pos[index_j]; 

	float4 r = (position_i - position_j); 
	r.w = 0.f; 
	
    // |r|
	float rlen = length(r);
	
    // is this particle within cutoff?
    if(rlen <= flockp->search_radius)
    {
        if(index_i != index_j){
	        // positions
	        float4 pj = pos[index_j];
	
	        // setup for rule 3. cohesion
            // xflock is the cohesion vector
            pt->cohesion+= pj; 		// center of the flock
	        pt->cohesion.w = 1.f;
        }
    }
}


//Contains Iterate...Cells methods and ZeroPoint
#include "cl_neighbors.h"


//--------------------------------------------------------------
// compute the cohesion 

__kernel void rule_cohesion(
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

    float4 position_i = pos[index];

    // Do calculations on particles in neighboring cells
	Boid pt;
	zeroPoint(&pt);

    IterateParticlesInNearbyCells(/*vars_sorted*/ ARGV, &pt, num, index, position_i, cell_indexes_start, cell_indexes_end, gp,/* fp,*/ flockp DEBUG_ARGV);
	
	// dividing by the number of flockmates to get the actual average
	pt.cohesion = flockmates[index].x > 0 ? pt.cohesion/(float)flockmates[index].x: pt.cohesion;

	// steering towards the average velocity 
	pt.cohesion -= position_i;
    pt.cohesion.w = 0.f;
	pt.cohesion = normalize(pt.cohesion);

    cohesion[index] = pt.cohesion;

}

#endif


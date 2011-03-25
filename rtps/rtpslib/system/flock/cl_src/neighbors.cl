#ifndef _NEIGHBORS_CL_
#define _NEIGHBORS_CL_

#include "cl_macros.h"
#include "cl_structs.h"

//----------------------------------------------------------------------
inline void ForNeighbor(__global float4*  vars_sorted,
				PointData* pt,
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
	float4 position_j = pos(index_j); 

	float4 r = (position_i - position_j); 
	r.w = 0.f; 
	
    // |r|
	float rlen = length(r);

    float min_dist = flockp->min_dist;
    float smooth_dist = flockp->smoothing_distance;
    float radius = flockp->search_radius;


    // is this particle within cutoff?
    if (smooth_dist >= radius && rlen <= radius) 
    {
        if (flockp->choice == 0) {
            // are the boids the same? 

            // compute the rules 
            #include "cl_density.h"
        }

        if (flockp->choice == 1) {
        
        }

        if (flockp->choice == 2) {
        
        }

        if (flockp->choice == 3) {
        
        }

    }
}


//Contains Iterate...Cells methods and ZeroPoint
#include "cl_neighbors.h"


//--------------------------------------------------------------
// compute forces on particles

__kernel void neighbors(
				__global float4* vars_sorted,
        		__global int*    cell_indexes_start,
        		__global int*    cell_indexes_end,
				__constant struct GridParams* gp,
				__constant struct FLOCKParameters* flockp 
				DEBUG_ARGS
				)
{
    // particle index
	int nb_vars = flockp->nb_vars;
	int num = flockp->num;

    int index = get_global_id(0);
    if (index >= num) return;

	clf[index] = (float4)(0.,0.,0.,10.);
	cli[index] = (int4)(0.,0.,0.,0.);

    float4 position_i = pos(index);

    // Do calculations on particles in neighboring cells
	PointData pt;
	zeroPoint(&pt);

	if (flockp->choice == 0) { // update density
    	IterateParticlesInNearbyCells(vars_sorted, &pt, num, index, position_i, cell_indexes_start, cell_indexes_end, gp,/* fp,*/ flockp DEBUG_ARGV);
		
        den(index) = pt.density;
		xflock(index) = pt.xflock;
        force(index) = pt.force;
        surface(index) = pt.surf_tens;

	}
}

#endif


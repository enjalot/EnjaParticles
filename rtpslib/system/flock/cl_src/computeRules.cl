#ifndef _NEIGHBORS_CL_
#define _NEIGHBORS_CL_

//These are passed along through cl_neighbors.h
//only used inside ForNeighbor defined in this file
#define ARGS __global float4* pos,__global float4* vel,  __global float4* separation, __global float4* alignment, __global float4* cohesion, __global int4* flockmates         
#define ARGV pos, vel, separation, alignment, cohesion, flockmates 

//#define ARGS __global float4* pos, __global float* density
//#define ARGV pos, density

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
	float4 position_j = pos[index_j]; 

	float4 r = (position_i - position_j); 
	r.w = 0.f; 
	
    // |r|
	float rlen = length(r);

    //float min_dist = flockp->min_dist;
    //float smooth_dist = flockp->smoothing_distance;
    //float radius = flockp->search_radius;


    // is this particle within cutoff?
    if (flockp->smoothing_distance >= flockp->search_radius && rlen <= flockp->search_radius) 
    {
        
        if(index_i != index_j){
	
	        // positions
	        float4 pj = pos[index_j];
	
            // velocities
	        float4 vj = vel[index_j];
            

	        // number of flockmates
            pt->num_flockmates++;

	        // setup for Rule 1. Separation
	        // force is the separation vector
            float4 s = r;       //pi - pj;
	        float  d = rlen;    //length(s);
	
            if(flockp->smoothing_distance >= flockp->min_dist && d <= flockp->min_dist){
		        s.w = 0.0f;
                s = normalize(s);
                s /= d;
	            pt->separation+= s;        // accumulate the separation vector
                pt->num_nearestFlockmates++;  // count how many flockmates are with in the separation distance
	        }

	        // setup for rule 2. alignment
	        // surf_tens is the alignment vector
            pt->alignment+= vj;   // desired velocity
	        pt->alignment.w = 1.f;

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
// compute forces on particles

__kernel void computeRules_update(
				//__global float4* vars_sorted,
                ARGS,
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

	//clf[index] = (float4)(0.,0.,0.,10.);
	//cli[index] = (int4)(0.,0.,0.,0.);

    float4 position_i = pos[index];

    // Do calculations on particles in neighboring cells
	Boid pt;
	zeroPoint(&pt);

	//if (flockp->choice == 0) { // update density
    	IterateParticlesInNearbyCells(/*vars_sorted*/ ARGV, &pt, num, index, position_i, cell_indexes_start, cell_indexes_end, gp,/* fp,*/ flockp DEBUG_ARGV);
		
        separation[index] = pt.separation;
        alignment[index] = pt.alignment;
        cohesion[index] = pt.cohesion;
        flockmates[index].x = pt.num_flockmates;
        flockmates[index].y = pt.num_nearestFlockmates;

        //den(index) = pt.density;
		//xflock(index) = pt.xflock;
        //force(index) = pt.force;
        //surface(index) = pt.surf_tens;

	//}
}

#endif


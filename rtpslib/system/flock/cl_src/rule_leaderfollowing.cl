#ifndef _RULE_LEADERFOLLOWING_CL_
#define _RULE_LEADERFOLLOWING_CL_

//These are passed along through cl_neighbors.h
//only used inside ForNeighbor defined in this file
#define ARGS __global float4* pos, __global float4* vel, __global float4* leaderfollowing, __global int4* flockmates 
#define ARGV pos, vel, leaderfollowing, flockmates 

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

    // neighbor within the radius?    
    if(rlen <= flockp->search_radius)
    {
        if(index_i != index_j){
	        // positions
	        float4 pj = pos[index_j];
	
	        // setup for Rule 1. Separation
            float4 s = r;       //pi - pj;
	        float  d = rlen;    //length(s);
	
		     if(d <= flockp->min_dist){ 
                s.w = 0.0f;
                s = normalize(s);
                s /= d;
	            pt->leaderfollowing+= s;        // accumulate the leaderfollowing vector
	        }
        }
    }
}


//Contains Iterate...Cells methods and ZeroPoint
#include "cl_neighbors.h"


//--------------------------------------------------------------
// compute the leaderfollowing 

__kernel void rule_leaderfollowing(
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

    float4 position_i = pos[index];

	Boid pt;
	zeroPoint(&pt);
   
    // set which boid is going to be the leader
    // for now boid[0] is going to be the leader
    
    if(index != 0){ 
        // is a follower: compute arrival and separation
    
        // create the rectangle in front of the leader
        float4 corner = (float4)(position_i.x + flockp->min_dist/2, position_i.y, position_i.z, 0.f);

        float4 vcorner1 = corner * 5.f * flockp->min_dist; 
        float4 vcorner2 = corner * 3.f * flockp->min_dist;
    
        float4 vposition = position_i - corner;    

        if(dot(vposition,vcorner1) >= 0.f && dot(vposition,vcorner1) <= dot(vcorner1,vcorner1) && dot(vposition,vcorner2) >= 0.f && dot(vposition,vcorner2) >= dot(vcorner2,vcorner2)){
            // steer away from the rectangle 
        }

        // set the offset point behind the leader
        float4 target_offset_point = pos[0] - (float4)(flockp->min_dist/2, flockp->min_dist/2, flockp->min_dist/2,0.f);

        // compute arrival behavior
        float4 target_offset = target_offset_point - position_i;
        float distance = length(target_offset);
        float ramped_speed = flockp->max_speed * (distance/flockp->slowing_distance);
        float clipped_speed = min(ramped_speed, flockp->max_speed);
        float4 desired_velocity = (clipped_speed/distance) * target_offset;

        // compute the leader following behavior
        pt.leaderfollowing = desired_velocity - vel[index];
    
        // apply separation 
        // Do calculations on particles in neighboring cells
        IterateParticlesInNearbyCells(/*vars_sorted*/ ARGV, &pt, num, index, position_i, cell_indexes_start, cell_indexes_end, gp,/* fp,*/ flockp DEBUG_ARGV);

        // finish the computation of the separation 
        if(pt.num_nearestFlockmates > 0){
            pt.leaderfollowing /= (float)pt.num_nearestFlockmates;
            pt.leaderfollowing.w =0.f;
            pt.leaderfollowing = normalize(pt.leaderfollowing);
        }

        leaderfollowing[index] = pt.leaderfollowing;
    }
    else{
        // is the leader: compute seek and wander

    }
}

#endif


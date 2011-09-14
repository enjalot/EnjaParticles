/****************************************************************************************
* Real-Time Particle System - An OpenCL based Particle system developed to run on modern GPUs. Includes SPH fluid simulations.
* version 1.0, September 14th 2011
* 
* Copyright (C) 2011 Ian Johnson, Andrew Young, Gordon Erlebacher, Myrna Merced, Evan Bollig
* 
* This software is provided 'as-is', without any express or implied
* warranty.  In no event will the authors be held liable for any damages
* arising from the use of this software.
* 
* Permission is granted to anyone to use this software for any purpose,
* including commercial applications, and to alter it and redistribute it
* freely, subject to the following restrictions:
* 
* 1. The origin of this software must not be misrepresented; you must not
* claim that you wrote the original software. If you use this software
* in a product, an acknowledgment in the product documentation would be
* appreciated but is not required.
* 2. Altered source versions must be plainly marked as such, and must not be
* misrepresented as being the original software.
* 3. This notice may not be removed or altered from any source distribution.
****************************************************************************************/


#ifndef _RULES_CL_
#define _RULES_CL_

//These are passed along through cl_neighbors.h only used inside ForNeighbor defined in this file
#define ARGS float4 target, __global float4* pos,  __global float4* vel, __global int4* flockmates,  __global float4* separation, __global float4* alignment, __global float4* cohesion, __global float4* goal, __global float4* avoid 
#define ARGV target, pos, vel, flockmates, separation, alignment, cohesion, goal, avoid

#include "cl_macros.h"
#include "cl_structs.h"

//----------------------------------------------------------------------
inline void ForNeighbor(ARGS,
                        Boid *pt,
				        uint index_i,
				        uint index_j,
				        float4 position_i,
	  			        __constant struct GridParams* gp,
 	  			        __constant struct FLOCKParameters* flockp
                        DEBUG_ARGS)
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

            if(flockp->w_sep > 0.f){
                #include "rule_separation.cl"
            }

            if(flockp->w_align > 0.f){
                #include "rule_alignment.cl"
            }

            if(flockp->w_coh > 0.f){
                #include "rule_cohesion.cl"
            }
        }

    }
}


//Contains Iterate...Cells methods and ZeroPoint
#include "cl_neighbors.h"


//--------------------------------------------------------------
__kernel void rules(ARGS,
        		__global int*    cell_indexes_start,
        		__global int*    cell_indexes_end,
	  			__constant struct GridParams* gp,
				__constant struct FLOCKParameters* flockp 
				DEBUG_ARGS)
{
    // particle index
	int num = flockp->num;

    int index = get_global_id(0);
    if (index >= num) return;

    float4 position_i = pos[index] * flockp->simulation_scale;
    float4 velocity_i = vel[index];

    // Do calculations on particles in neighboring cells
	Boid pt;
	zeroPoint(&pt);

    IterateParticlesInNearbyCells(ARGV, &pt, num, index, position_i, cell_indexes_start, cell_indexes_end, gp, flockp DEBUG_ARGV);

    // average separation
    if(flockp->w_sep > 0.f && pt.num_nearestFlockmates > 0){
        pt.separation /= (float)pt.num_nearestFlockmates;
        pt.separation.w = 0.f;
    }

    // average alignment
    if(flockp->w_align > 0.f && pt.num_flockmates > 0){
	    // dividing by the number of flockmates to get the desired velocity 
	    pt.alignment /= (float)pt.num_flockmates;
        pt.alignment -= velocity_i;
        pt.alignment.w = 0.f;
    }

    // average cohesion
    if(flockp->w_coh > 0.f && pt.num_flockmates > 0){
	    // dividing by the number of flockmates to get the center of mass 
	    pt.cohesion /= (float)pt.num_flockmates;
        pt.cohesion -= position_i;
        pt.cohesion.w = 0.f;
    }
   
    // compute goal
    if(flockp->w_goal > 0.f){
        #include "rule_goal.cl"
    }
    
    // compute avoid
    if(flockp->w_avoid > 0.f){
        #include "rule_avoid.cl"
    }

    //clf[index] = pt.goal;//(float4)(3.,3.,3.,3.); //pt.separation; 
    //cli[index] = (int4)((int)flockp->w_sep,(int)flockp->w_align,(int)flockp->w_coh,(int)flockp->w_goal);
    
    flockmates[index].x = pt.num_flockmates;
    flockmates[index].y = pt.num_nearestFlockmates;
    
    separation[index]   = pt.separation;
    alignment[index]    = pt.alignment;
    cohesion[index]     = pt.cohesion;
    goal[index]         = pt.goal;
    avoid[index]        = pt.avoid;
}

#endif


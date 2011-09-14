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


#ifndef _EULER_INTEGRATION_CL_
#define _EULER_INTEGRATION_CL_

#include "cl_macros.h"
#include "cl_structs.h"
 
        
__kernel void euler_integration(
                    float dt,
                    int two_dimensional,
                   __global float4* pos_u, 
                   __global float4* pos_s, 
                   __global float4* vel_u, 
                   __global float4* vel_s, 
                   __global float4* separation_s, 
                   __global float4* alignment_s, 
                   __global float4* cohesion_s, 
                   __global float4* goal_s, 
                   __global float4* avoid_s, 
                   __global float4* leaderfollowing_s, 
                   __global int* sort_indices,  
                   __constant struct FLOCKParameters* flockp,
                   __constant struct GridParams* gridp
                    DEBUG_ARGS
                    )
                   
{
    unsigned int i = get_global_id(0);
    int num = flockp->num;
    
    if(i >= num) 
	    return;

	// positions
	float4 pi = pos_s[i] * flockp->simulation_scale;

	// velocities
    float4 vi = vel_s[i];

	// veleleration vectors
    float4 vel      = (float4)(0.f, 0.f, 0.f, 1.f);
    float4 vel_sep  = (float4)(0.f, 0.f, 0.f, 1.f);
    float4 vel_aln  = (float4)(0.f, 0.f, 0.f, 1.f);
    float4 vel_coh  = (float4)(0.f, 0.f, 0.f, 1.f);
    float4 vel_goal = (float4)(0.f, 0.f, 0.f, 1.f);
    float4 vel_avoid = (float4)(0.f, 0.f, 0.f, 1.f);
    float4 vel_leadfoll = (float4)(0.f, 0.f, 0.f, 1.f);

    // getting the values of the rules computed in cl_density
	float4 separation = separation_s[i]; 
	float4 alignment = alignment_s[i]; 
	float4 cohesion = cohesion_s[i]; 
    float4 goal = goal_s[i];
    float4 avoid = avoid_s[i];
	float4 leaderfollowing = leaderfollowing_s[i]; 

    // weights for the rules
	float w_sep = flockp->w_sep;    
	float w_aln = flockp->w_align; 
	float w_coh = flockp->w_coh;
    float w_goal = flockp->w_goal;
    float w_avoid = flockp->w_avoid;  
	float w_leadfoll = flockp->w_leadfoll;   
	
    // boundary limits, used to computed boundary conditions    
	float4 bndMax = gridp->bnd_max;
	float4 bndMin = gridp->bnd_min;

	// RULE 1. SEPARATION
	vel_sep = separation * w_sep;
	
	// RULE 2. ALIGNMENT
	vel_aln = alignment * w_aln;

	// RULE 3. COHESION
	vel_coh = cohesion * w_coh;

    // RULE 4. GOAL
    vel_goal = goal * w_goal;

    // RULE 5. AVOID
    vel_avoid = avoid * w_avoid;

    // RULE 6. LEADER FOLLOWING
    vel_leadfoll = leaderfollowing * w_leadfoll;
    
    // compute vel
    vel = vi + vel_sep + vel_aln + vel_coh + vel_goal + vel_avoid + vel_leadfoll;
	vel.w = 0.f;

    // constrain veleleration
    float velspeed = length(vel);
    float4 velnorm = normalize(vel);
    if(velspeed > flockp->max_speed){
        // set magnitude to Max Speed
        vel = velnorm * flockp->max_speed;
    }

    // add circular velocity field
    float4 v = (float4)(-3*pi.y, pi.x, 0.f, 0.f);
    v *= flockp->ang_vel;    

    // add veleleration to velocity
    vi = v + vel;
    vi.w =0.f;

	// INTEGRATION
    pi += dt*vi; 	// averageRules integration, add the velocity times the timestep

#if 1
	// apply periodic boundary conditions
	// assumes particle cannot move by bndMax.x in one iteration
	if(pi.x >= bndMax.x){
		pi.x -= bndMax.x; 
	}
	else if(pi.x <= bndMin.x){
		pi.x += bndMax.x;
	}
	else if(pi.y >= bndMax.y){
		pi.y -= bndMax.y; 
	}
	else if(pi.y <= bndMin.y){
		pi.y += bndMax.y;
	}
	else if(pi.z >= bndMax.z){
		pi.z -= bndMax.z;
	}
	else if(pi.z <= bndMin.z){
		pi.z += bndMax.z;
	}
#endif

    if(two_dimensional)
        pi.z = 0.f;

	// STORE THE NEW POSITION AND NEW VELOCITY 
    uint originalIndex = sort_indices[i];
    vel_u[originalIndex] = vi;	
    pos_u[originalIndex] = (float4)(pi.xyz/flockp->simulation_scale, 1.f);    // changed the last component to 1 for my boids, im not using density
}

#endif

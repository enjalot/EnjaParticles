#ifndef _EULER_INTEGRATION_CL_
#define _EULER_INTEGRATION_CL_

#include "cl_macros.h"
#include "cl_structs.h"
 
        
__kernel void euler_integration(
                    float dt,
                   __global float4* pos_u, 
                   __global float4* pos_s, 
                   __global float4* vel_u, 
                   __global float4* vel_s, 
                   __global float4* separation_s, 
                   __global float4* alignment_s, 
                   __global float4* cohesion_s, 
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

	// acceleration vectors
    float4 acc     = (float4)(0.f, 0.f, 0.f, 1.f);
    float4 acc_sep = (float4)(0.f, 0.f, 0.f, 1.f);
    float4 acc_aln = (float4)(0.f, 0.f, 0.f, 1.f);
    float4 acc_coh = (float4)(0.f, 0.f, 0.f, 1.f);
    float4 acc_leadfoll = (float4)(0.f, 0.f, 0.f, 1.f);

    // getting the values of the rules computed in cl_density
	float4 separation = separation_s[i]; 
	float4 alignment = alignment_s[i]; 
	float4 cohesion = cohesion_s[i]; 
	float4 leaderfollowing = leaderfollowing_s[i]; 

    // weights for the rules
	float w_sep = flockp->w_sep;    //0.10f;  // 0.3f
	float w_aln = flockp->w_align;  //0.001f;
	float w_coh = flockp->w_coh;    //0.0001f;  // 3.f
	float w_leadfoll = flockp->w_leadfoll;   
	
    // boundary limits, used to computed boundary conditions    
	float4 bndMax = gridp->bnd_max;
	float4 bndMin = gridp->bnd_min;

	// RULE 1. SEPARATION
	acc_sep = separation * w_sep;
	
	// RULE 2. ALIGNMENT
	acc_aln = alignment * w_aln;

	// RULE 3. COHESION
	acc_coh = cohesion * w_coh;

    // RULE 4. LEADER FOLLOWING
    acc_leadfoll = leaderfollowing * w_leadfoll;
    
    // compute acc
    acc = vi + acc_sep + acc_aln + acc_coh + acc_leadfoll;
	acc.w = 0.f;

    // constrain acceleration
    float accspeed = length(acc);
    float4 accnorm = normalize(acc);
    if(accspeed > flockp->max_speed){
        // set magnitude to Max Speed
        acc = accnorm * flockp->max_speed;
    }

    // add circular velocity field
    float4 v = (float4)(-pi.z, 0.f, pi.x, 0.f);
    v *= 0.0f;     // TODO: Add this parameter to Blender

    // add acceleration to velocity
    vi = v + acc;
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

	// STORE THE NEW POSITION AND NEW VELOCITY 
    uint originalIndex = sort_indices[i];
    vel_u[originalIndex] = vi;	
    pos_u[originalIndex] = (float4)(pi.xyz/flockp->simulation_scale, 1.f);    // changed the last component to 1 for my boids, im not using density
}

#endif

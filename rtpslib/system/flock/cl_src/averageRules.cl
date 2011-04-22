#include "cl_macros.h"
#include "cl_structs.h"
 
float magnitude(float4 vec)
{
    return sqrt(vec.x*vec.x + vec.y*vec.y + vec.z*vec.z);
}       
        
__kernel void averageRules(
                    float dt,
                   //__global float4* vars_unsorted, 
                   //__global float4* vars_sorted, 
                   //__global float4* positions,  // for VBO 
                   __global float4* pos_u, 
                   __global float4* pos_s, 
                   __global float4* vel_u, 
                   __global float4* vel_s, 
                   __global float4* separation_s, 
                   __global float4* alignment_s, 
                   __global float4* cohesion_s, 
                   __global float4* flockmates_s, 
                   __global int* sort_indices,  
                   //		__global float4* color,
                   __constant struct FLOCKParameters* flockp,
                   __constant struct GridParams* gridp)
                   
{
/*__kernel void averageRules(
        __global int* sort_indices,  
		__global float4* vars_unsorted, 
		__global float4* vars_sorted, 
		__global float4* positions,  // for VBO 
		__constant struct FLOCKParameters* flockp, 
		float dt
		DEBUG_ARGS
        )
{*/
    unsigned int i = get_global_id(0);
    int num = flockp->num;
    
    if(i >= num) 
	    return;

	// positions
	float4 pi = pos_s[i];

	// velocities
    float4 vi = vel_s[i];

	// acceleration vectors
    float4 acc     = (float4)(0.f, 0.f, 0.f, 1.f);
    float4 acc_sep = (float4)(0.f, 0.f, 0.f, 1.f);
    float4 acc_aln = (float4)(0.f, 0.f, 0.f, 1.f);
    float4 acc_coh = (float4)(0.f, 0.f, 0.f, 1.f);

    // getting the values of the rules computed in cl_density
	float4 separation = separation_s[i]; 
	float4 alignment = alignment_s[i]; 
	float4 cohesion = cohesion_s[i]; 
	
    // getting number of flockmates and how many flockmates were within the separation dist
	float numFlockmates = flockmates_s[i].x;
    float count =  flockmates_s[i].y;

    // weights for the rules
	float w_sep = flockp->w_sep;    //0.10f;  // 0.3f
	float w_aln = flockp->w_align;  //0.001f;
	float w_coh = flockp->w_coh;    //0.0001f;  // 3.f
	
    // boundary limits, used to computed boundary conditions    
	float4 bndMax = gridp->grid_max;
	float4 bndMin = gridp->grid_min;


	// RULE 1. SEPARATION
	// already computed in cl_density.h
	// it is stored in pt->force
    if(count > 0){
        separation /=count;
        separation.w =0.f;
        separation = normalize(separation);
    }
	acc_sep = separation * w_sep;
	
	// RULE 2. ALIGNMENT
	// desired velocity computed in cl_density.h
	// it is stored in pt->surf_tens
	// dividing by the number of flockmates to get the actual average
	alignment = numFlockmates > 0 ? alignment/numFlockmates: alignment;

	// steering towards the average velocity 
	alignment -= vi;
    alignment.w = 0.f;
	alignment = normalize(alignment);
	acc_aln = alignment * w_aln;

	// RULE 3. COHESION
	// average position already computed in cl_density.h
	// it is stored in pt->xflock
	// dividing by the number of flockmates to get the actual average
    cohesion = numFlockmates > 0 ? cohesion/numFlockmates: cohesion;

	// steering towards the average position
	cohesion -= pi;
    cohesion.w = 0.f;
	cohesion = normalize(cohesion);
	acc_coh = cohesion * w_coh;

    // compute acc
    acc = vi + acc_sep + acc_aln + acc_coh;
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
    v *= 0.00f;

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


	// SORT STUFF FOR THE NEIGHBOR SEARCH
    uint originalIndex = sort_indices[i];
    vel_u[originalIndex] = vi;	
    pos_u[originalIndex] = (float4)(pi.xyz, 1.f);    // changed the last component to 1 for my boids, im not using density
    //positions[originalIndex] = (float4)(pi.xyz, 1.f);       // for plotting
    
    // debugging vectors
    //int4 iden = (int4)((int)den(i).x, (int)den(i).y, 0, 0);
    //cli[originalIndex] = iden;
    //clf[originalIndex] = pi; 
}

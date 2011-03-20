#include "cl_macros.h"
#include "cl_structs.h"
 
float magnitude(float4 vec)
{
    return sqrt(vec.x*vec.x + vec.y*vec.y + vec.z*vec.z);
}       
        
__kernel void euler(
        __global int* sort_indices,  
		__global float4* vars_unsorted, 
		__global float4* vars_sorted, 
		__global float4* positions,  // for VBO 
//		__global float4* color,
		__constant struct FLOCKParams* params, 
		float dt
		DEBUG_ARGS
        )
{
    unsigned int i = get_global_id(0);
    int num = params->num;
    
    if(i >= num) 
	return;

    // this parameters will be moved to FLOCKparams
	#define	maxspeed	    0.003f	    // .003f
	//#define desiredspeed	0.0025f	// .5f
	//#define MinUrgency      0.0025f	// .05f
	//#define MaxUrgency      0.005f	// .1f

	// positions
	float4 pi = pos(i);

	// velocities
    float4 vi = vel(i);

	// acceleration vectors
    float4 acc     = (float4)(0.f, 0.f, 0.f, 1.f);
    float4 acc_sep = (float4)(0.f, 0.f, 0.f, 1.f);
    float4 acc_aln = (float4)(0.f, 0.f, 0.f, 1.f);
    float4 acc_coh = (float4)(0.f, 0.f, 0.f, 1.f);

    // getting the values of the rules computed in cl_density
	float4 separation = force(i);
	float4 alignment = surface(i);
	float4 cohesion = xflock(i);
	
    // getting number of flockmates and how many flockmates were within the separation dist
	float numFlockmates = den(i).x;
    float count =  den(i).y;

    // weights for the rules
	float w_sep = 1.1f;
	float w_aln = 0.0f;
	float w_coh = 0.00003f;
	
    // boundary limits, used to computed boundary conditions    
	float4 bndMax = params->grid_max;
	float4 bndMin = params->grid_min;


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
	alignment = numFlockmates > 0 ? alignment/numFlockmates : alignment;

	// steering towards the average velocity 
	alignment -= vi;
    alignment.w = 0.f;
	alignment = normalize(alignment);
	acc_aln = alignment * w_aln;


	// RULE 3. COHESION
	
	// average position already computed in cl_density.h
	// it is stored in pt->xflock

	// number of flockmates calculated in cl_density.h
	// it is stored in pt->density.x

	// dividing by the number of flockmates to get the actual average
    cohesion = numFlockmates > 0 ? cohesion/numFlockmates : cohesion;

	// steering towards the average position
	cohesion -= pi;
    cohesion.w = 0.f;
	cohesion = normalize(cohesion);
	acc_coh = cohesion * w_coh;

    // compute acc
    acc = vi + acc_sep + acc_aln + acc_coh;
	acc.w = 0.0f;

	// constrain acceleration
    float accspeed = length(acc);
    float4 accnorm = normalize(acc);
    if(accspeed > maxspeed){
        // set magnitude to Max Speed
        acc = accnorm * maxspeed;
    }

    // add circular velocity field
    float4 v = (float4)(-pi.z, 0.f, pi.x, 0.f);
    v *= 0.00f;

    // add acceleration to velocity
    vi = v + acc;
    vi.w =0.f;


	// INTEGRATION
    pi += dt*vi; 	// euler integration, add the velocity times the timestep

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
    unsorted_vel(originalIndex) = vi;	
    unsorted_pos(originalIndex) = (float4)(pi.xyz, 1.f);    // changed the last component to 1 for my boids, im not using density
	//clf[i].xyz = pi.xyz;
    positions[originalIndex] = (float4)(pi.xyz, 1.f);       // for plotting
    
    // debugging vectors
    int4 iden = (int4)((int)den(i).x, (int)den(i).y, 0, 0);
    cli[originalIndex] = iden;
    clf[originalIndex] = vi; 

}

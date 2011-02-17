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

//    float4 p = pos(i);
//  float4 v = vel(i);  mymese
//    float4 f = force(i);

    //external force is gravity
//  f.z += -9.8f;	mymese

//  float speed = magnitude(f);	mymese
//  if(speed > 600.0f) 		mymese //velocity limit, need to pass in as struct
//  {					mymese
//      f *= 600.0f/speed;		mymese
//  }					mymese

//  v += dt*f;			mymese
    //p += dt*v / params->simulation_scale;
//  p += dt*v;			mymese




#if 1
	#define	separationdist	0.2f
	#define	searchradius	0.5f
	#define	maxspeed	0.3f
	#define MinUrgency      0.05f
	#define MaxUrgency      0.1f

	// index of closest flockmate
	int index_c = (int) den(i).x;

	// positions
	float4 pi = pos(i);
	float4 pc = pos(index_c);

	// velocities
	//float4 vi = force(index_i);
	//float4 vj = force(index_j);
    	float4 v = vel(i);
	float4 vc = vel(index_c);

	// initialize acc to zero
    	float4 acc = (float4)(0.f, 0.f, 0.f, 0.f);

        // Step 1. Update position
	// REMEMBER THAT MY VELOCITIES ARE GOING TO BE STORED IN THE FORCE VECTOR FROM NOW ON
//        pt->position += vi;

        // Step 2. Search for neighbors
        //numFlockmates = SearchFlockmates(i);

	// flockmates
	//int closestFlockmate = pc;
	int numFlockmates = (int) den(i).y;	// TODO: is the y component but Im getting error: make sure that density is a float4

        // Step 3. Compute the three rules
	
	// RULE 1. Separation
	float d = distance(pi,pc);
	float r = d / separationdist;  				// TODO: NEED THE CLOSEST FLOCKMATE (ID OR POSITION), AND THE DISTANTCE FROM IT TO THE CURRENT BOID

    float4 separation = pi - pc;
    separation = normalize(separation);			// TODO: search for normalization in OpenCL Specification

    if(d > separationdist){
            separation *=  r;
    }
    else if(d < separationdist){
            separation *= -r;
    }
    else{
            separation *= 0.f;
    }
    
	acc += separation;
                
	// RULE 2. Alignment
	float4 alignment = vc - v;
    alignment = normalize(alignment);

    acc += alignment;

    // RULE 3. Cohesion
    float4 flockCenter= xflock(i);
	float4 cohesion;

    flockCenter /= numFlockmates;
    cohesion = pi - flockCenter;
    cohesion = normalize(cohesion);

	acc += cohesion;

    // Step 4. Constrain acceleration
    float accspeed = length(acc);
    if(accspeed > maxspeed*MaxUrgency){
            // set magnitude to MaxChangeInAcc
            acc *= (maxspeed*MaxUrgency)/accspeed;
    }

    // Step 5. Add acceleration to velocity
    v += acc;

    // Step 6. Constrain velocity
    float speed = length(v);
    if(speed > maxspeed){
            // set magnitude to MaxSpeed
        v *= maxspeed/speed;
    }

#endif

#if 0
	// positions
	float4 pi = pos(i);

	// velocities
    	float4 v = vel(i);
#endif 

    pi += dt*v; // change it to force for my boids
    //pi.xyz /= params->simulation_scale;

    uint originalIndex = sort_indices[i];

    int4 iden = (int4)((int)den(i).x, (int)den(i).y, 0, 0);
    cli[originalIndex] = iden;
    //clf[originalIndex] = xflock(i);
    clf[originalIndex] = v;



    unsorted_vel(originalIndex) = v;	//mymese
    //unsorted_vel(originalIndex) = (float4)(4., 4., 4., 4.);	//mymese
    //unsorted_veleval(originalIndex) = v;
//  float dens = density(i);		mymese
//  unsorted_pos(originalIndex) = (float4)(p.xyz, dens);	mymese
    unsorted_pos(originalIndex) = (float4)(pi.xyz, 1.f); // change the last component to 1 for my boids, im not using density
    positions[originalIndex] = (float4)(pi.xyz, 1.f);  // for plotting

}

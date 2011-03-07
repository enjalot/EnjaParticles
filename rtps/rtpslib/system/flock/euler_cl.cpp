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





//	#define	separationdist	0.005f	// 3.f
//	#define	searchradius	0.8f
	#define	maxspeed	3.f	// 1.f
	#define desiredspeed	1.5f	// .5f
	#define maxchange	0.1f	// .1f
	#define MinUrgency      0.05f	// .05f
	#define MaxUrgency      0.1f	// .1f

#if 0
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
	float numFlockmates = den(i).y;	// TODO: is the y component but Im getting error: make sure that density is a float4

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

#if 1
	// positions
	float4 pi = pos(i);

	// velocities
    	float4 v = vel(i);

	// acceleration vector
    	float4 acc = (float4)(0.f, 0.f, 0.f, 0.f);

	float4 separation = force(i);
	float4 alignment = surface(i);
	float4 cohesion = xflock(i);
	
	float numFlockmates = den(i).x;
//	if(numFlockmates == 0){
//		numFlockmates = 1;
//	}

	float w_sep = .75f;
	float w_aln = 1.f;
	float w_coh = .01f;
		
	float4 bndMax = params->grid_max;// - params->boundary_distance;
	float4 bndMin = params->grid_min;// + params->boundary_distance;

	// RULE 1. SEPARATION
	
	// already computed in cl_density.h
	// it is stored in pt->force
	//separation = normalize(separation);
	//if(length(separation) < separationdist){
	//	separation *= 2;
	//}	
	acc += separation * w_sep;

	
	// RULE 2. ALIGNMENT

	// desired velocity computed in cl_density.h
	// it is stored in pt->surf_tens

	// steering towards the average velocity
	alignment /= numFlockmates;
	alignment -= v;
	alignment = normalize(alignment);
	acc += alignment * w_aln;


	// RULE 3. COHESION
	
	// average position already computed in cl_density.h
	// it is stored in pt->xflock

	// number of flockmates calculated in cl_density.h
	// it is stored in pt->density.x

	// dividing by the number of flockmates to get the actual average
	cohesion /= numFlockmates;

	// steering towards the average position
	cohesion -= pi;
	cohesion = normalize(cohesion);
	acc += cohesion * w_coh;
    

	// Step 4. Constrain acceleration
    	float accspeed = length(acc);
    	if(accspeed > maxchange){
            	// set magnitude to MaxChangeInAcc
            	acc *= maxchange/accspeed;
    	}

    	// Step 5. Add acceleration to velocity
    	v += acc;

	v.x += MinUrgency;

    	// Step 6. Constrain velocity
    	float speed = length(v);
    	if(speed > maxspeed){
            	// set magnitude to MaxSpeed
        	v *= maxspeed/speed;
    	}

#endif

	// INTEGRATION
//v.x=.1f;
//v.y=.1f;
//v.z=.1f;
 
    	pi += dt*v; 	// euler integration, add the velocity times the timestep
    	//pi.xyz /= params->simulation_scale;

#if 1
	// apply periodic boundary conditions
	if(pi.x >= bndMax.x){
		pi.x = bndMin.x; 
	}
	else if(pi.x <= bndMin.x){
		pi.x = bndMax.x;
	}
	else if(pi.y >= bndMax.y){
		pi.y = bndMin.y; 
	}
	else if(pi.y <= bndMin.y){
		pi.y = bndMax.y;
	}
	else if(pi.z >= bndMax.z){
		pi.z = bndMin.z;
	}
	else if(pi.z <= bndMin.z){
		pi.z = bndMax.z;
	}
#endif

	// SORT STUFF FOR THE NEIGHBOR SEARCH
    	uint originalIndex = sort_indices[i];

    	int4 iden = (int4)((int)den(i).x, (int)den(i).y, 0, 0);
    	cli[originalIndex] = iden;
    	//clf[originalIndex] = xflock(i);
    	clf[originalIndex] = v;

    	unsorted_vel(originalIndex) = v;	//mymese
    	//unsorted_vel(originalIndex) = (float4)(4., 4., 4., 4.);	//mymese
    	//unsorted_veleval(originalIndex) = v;	
	//float dens = density(i);		//mymese
	//unsorted_pos(originalIndex) = (float4)(p.xyz, dens);	//mymese
    	unsorted_pos(originalIndex) = (float4)(pi.xyz, 1.f); // change the last component to 1 for my boids, im not using density
    	positions[originalIndex] = (float4)(pi.xyz, 1.f);  // for plotting
}

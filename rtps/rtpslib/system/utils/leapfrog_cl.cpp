
#include "cl_macros.h"
#include "cl_structures.h"

 
__kernel void ge_leapfrog(
		__global int* sort_indices,  
		__global float4* vars_unsorted, 
		__global float4* vars_sorted, 
		// should not be required since part of vars_unsorted
		__global float4* positions,  // for VBO 
		__constant struct SPHParams* params, 
		float dt)
{
    unsigned int i = get_global_id(0);
	int num = get_global_size(0); // for access functions in cl_macros.h

    float4 p = pos(i);
    float4 v = vel(i);
    float4 f = force(i);

    //external force is gravity
    //f.z += -9.8f;
	f.w = 0.f;

	// REMOVE FOR DEBUGGING
	// THIS IS REALLY A FORCE, NO?
	#if 0
    float speed = length(f);
    if(speed > 600.0f) //velocity limit, need to pass in as struct
    {
        f *= 600.0f/speed;
    }
	#endif

	float4 vnext = v + dt*f;
	//float4 vnext = f;
	// WHY IS MY CORRECTION NEGATIVE and IAN's POSITIVE? 
	//vnext -= 0.005f * xsph(i); // should be param XSPH factor

    p += dt * vnext;
    p.w = 1.0f; //just in case
	float4 veval = 0.5f*(v+vnext);
	v = vnext;


	/ only for debuggin
	veval = f; // SEE IF FORCE IS SYMMETRIC

	// Should not be required, but it is for debugging if I wish to 
	// access the sorted arrays
	vel(i) = v;
	pos(i) = p;
	veleval(i) = veval;


	uint originalIndex = sort_indices[i];

	// writeback to unsorted buffer
	float dens = density(i);
	p.xyz /= params->simulation_scale;
	unsorted_pos(originalIndex) 	= (float4)(p.xyz, dens);
	unsorted_vel(originalIndex) 	= v;
	unsorted_density(originalIndex) = dens; // FOR DEBUGGING ONLY
	unsorted_force(originalIndex) 	= f; // FOR DEBUGGING ONLY
	unsorted_veleval(originalIndex) = veval; 
	positions[originalIndex] 		= (float4)(p.xyz, 1.);  // for plotting
	//positions[originalIndex] 		= (float4)(p.xyz, dens);  // for plotting
}



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
		float dt)
{
    unsigned int i = get_global_id(0);
    int num = params->num;
    
    if(i >= num) 
	return;

    float4 p = pos(i);
//  float4 v = vel(i);  mymese
    float4 f = force(i);

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
    p += dt*f; // change it to force for my boids
    p.w = 1.0f; //just in case
    p.xyz /= params->simulation_scale;

    uint originalIndex = sort_indices[i];

//  unsorted_vel(originalIndex) = v;	mymese
    //unsorted_veleval(originalIndex) = v;
//  float dens = density(i);		mymese
//  unsorted_pos(originalIndex) = (float4)(p.xyz, dens);	mymese
    unsorted_pos(originalIndex) = (float4)(p.xyz, 1.f); // change the last component to 1 for my boids, im not using density
    positions[originalIndex] = (float4)(p.xyz, 1.f);  // for plotting

}

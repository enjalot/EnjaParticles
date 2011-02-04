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
    if(i >= num) return;



    float4 p = pos(i);
    float4 v = vel(i);
    float4 f = force(i);

    //external force is gravity
    f.z += -9.8f;

    float speed = magnitude(f);
    if(speed > 600.0f) //velocity limit, need to pass in as struct
    {
        f *= 600.0f/speed;
    }

    v += dt*f;
    //p += dt*v / params->simulation_scale;
    p += dt*v;
    p.w = 1.0f; //just in case
	p.xyz /= params->simulation_scale;

	uint originalIndex = sort_indices[i];

    unsorted_vel(originalIndex) = v;
    //unsorted_veleval(originalIndex) = v;
    float dens = density(i);
	unsorted_pos(originalIndex) = (float4)(p.xyz, dens);
	positions[originalIndex] = (float4)(p.xyz, 1.);  // for plotting

}

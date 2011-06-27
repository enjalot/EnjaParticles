#include "cl_macros.h"
#include "cl_structs.h"

//float magnitude(float4 vec)
//{
    //return sqrt(vec.x*vec.x + vec.y*vec.y + vec.z*vec.z);
//}       

__kernel void cloudEuler(
				   int num,
                   __global float4* pos_u, 
                   __global float4* pos_s, 
                   __global float4* normal_u, 
                   __global float4* normal_s, 
                   __global float4* velocity_u, 
                   __global float4* velocity_s, 
				   // pos_cg: difference from previous time step
                   float4 pos_cg,  // just the tnraslation from pos_cg0
                   float4 diff_pos_cg,  // just the tnraslation from pos_cg0
                   __global int* sort_indices,  
                   __constant struct SPHParams* sphp, 
                   float dt)
{
    unsigned int i = get_global_id(0);
	float ss = sphp->simulation_scale;
    //int num = sphp->cloud_num;
    if (i >= num) return;

    //float4 p = pos_s[i] * ss;
    float4 p = pos_s[i] * ss;
    float4 vel = velocity_s[i];

    p += dt*vel + diff_pos_cg*ss;
	//p = pos_cg*sphp->simulation_scale + dt*vel;
    //p = dt*vel;
	
    p.w = 1.0f; //just in case
    p.xyz /= sphp->simulation_scale;

    uint originalIndex = sort_indices[i];

    pos_u[originalIndex]    =   (float4)(p.xyz, 1.);  // for plotting
    normal_u[originalIndex] =   normal_s[i];
    velocity_u[originalIndex] = velocity_s[i];
	return;
}


#include "cl_macros.h"
#include "cl_structs.h"

__kernel void kern_cloud_velocity(
				   int num, 		// nb cloud points
				   __global float4* pos_s, // cloud positions
				   __global float4* vel_s, // cloud positions
				   __global float4 pos_cg, // use a constant?
				   __global float4 omega)
{
    unsigned int i = get_global_id(0);
    if (i >= num) return;

	float4 p = pos_s[i] - pos_cg;
	vel_s[i] = (float4)(p.y*omega.z-p.z*omega.y, p.z*omega.x-p.x*omega.z, 
	                 p.x*omega.y-p.y*omega.x);
	vel_s[i] = (1.,0.,0.);
}

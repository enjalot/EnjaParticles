
#include "cl_macros.h"
#include "cl_structs.h"

__kernel void kern_cloud_velocity(
				   int num, 		// nb cloud points
				   float delta_angle,
				   __global float4* pos_s, // cloud positions
				   __global float4* vel_s, // cloud positions
				   float4 pos_cg, // use a constant?
				   float4 omega)
{
    unsigned int i = get_global_id(0);
    if (i >= num) return;

	// SHOULD NOT BE HARDCODED!! :-)
	float simulation_scale = 0.05;  // pass sphp as argument

	float4 p = simulation_scale*(pos_s[i] - pos_cg);
	vel_s[i] = (float4)(p.y*omega.z-p.z*omega.y, p.z*omega.x-p.x*omega.z, 
	                 p.x*omega.y-p.y*omega.x, 1.);
	//vel_s[i] *= omega.w;
	vel_s[i] *= delta_angle*100.f; // 100 is a fudge factor
	vel_s[i].w = 1.;
	//vel_s[i] = (1.,0.,0.,1.);
}

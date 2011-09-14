/****************************************************************************************
* Real-Time Particle System - An OpenCL based Particle system developed to run on modern GPUs. Includes SPH fluid simulations.
* version 1.0, September 14th 2011
* 
* Copyright (C) 2011 Ian Johnson, Andrew Young, Gordon Erlebacher, Myrna Merced, Evan Bollig
* 
* This software is provided 'as-is', without any express or implied
* warranty.  In no event will the authors be held liable for any damages
* arising from the use of this software.
* 
* Permission is granted to anyone to use this software for any purpose,
* including commercial applications, and to alter it and redistribute it
* freely, subject to the following restrictions:
* 
* 1. The origin of this software must not be misrepresented; you must not
* claim that you wrote the original software. If you use this software
* in a product, an acknowledgment in the product documentation would be
* appreciated but is not required.
* 2. Altered source versions must be plainly marked as such, and must not be
* misrepresented as being the original software.
* 3. This notice may not be removed or altered from any source distribution.
****************************************************************************************/



#include "cl_macros.h"
#include "cl_structs.h"

__kernel void kern_cloud_velocity(
				   int num, 		// nb cloud points
				   __global float4* pos_s, // cloud positions
				   __global float4* vel_s, // cloud positions
				   float4 pos_cg, // use a constant?
				   float4 omega)
{
    unsigned int i = get_global_id(0);
    if (i >= num) return;

	float4 simulation_scale = 0.05;  // pass sphp as argument

	float4 p = simulation_scale*(pos_s[i] - pos_cg);
	vel_s[i] = (float4)(p.y*omega.z-p.z*omega.y, p.z*omega.x-p.x*omega.z, 
	                 p.x*omega.y-p.y*omega.x, 1.);
	//vel_s[i] = (1.,0.,0.,1.);
}

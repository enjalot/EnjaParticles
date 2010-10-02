
#include "cl_macros.h"
#include "cl_structures.h"

        
 
__kernel void ge_euler(
		__global int* sort_indices,  
		__global float4* vars_unsorted, 
		__global float4* vars_sorted, 
		__constant struct SPHParams* params, 
		__constant float dt)
{
#if 1
    unsigned int i = get_global_id(0);
	int num = get_global_size(0); // for access functions in cl_macros.h

    float4 p = pos(i);
    float4 v = vel(i);
    float4 f = force(i);

    //external force is gravity
    f.z += -9.8f;

    float speed = length(f);
    if(speed > 600.0f) //velocity limit, need to pass in as struct
    {
        f *= 600.0f/speed;
    }

    v += dt*f / params->simulation_scale;
    p += dt*v;
    p.w = 1.0f; //just in case

    //vel(i) = v;
    //pos(i) = p;

    //if(progress)
    //{
        uint originalIndex = sort_indices[i];

        // writeback to unsorted buffer
		pos(originalIndex) = p;
		vel(originalIndex) = v;
		//force(originalIndex) = f;
        //dParticleData.position[originalIndex]   = make_vec(pos);
        //dParticleData.velocity[originalIndex]   = make_vec(vel);
        //dParticleData.veleval[originalIndex]    = make_vec(vel_eval);

        //float3 color = CalculateColor(coloringGradient, coloringSource, vnext, sph_pressure, sph_force);
        //dParticleData.color[originalIndex]  = make_float4(color, 1);
    //}



#endif
}


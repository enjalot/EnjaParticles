#include "../SPH.h"

namespace rtps {

void SPH::loadEuler()
{
    printf("create euler kernel\n");

    std::string path(SPH_CL_SOURCE_DIR);
    path += "/euler_cl.cl";
    k_euler = Kernel(ps->cli, path, "euler");
  
    int iargs = 0;
    k_euler.setArg(iargs++, cl_sort_indices.getDevicePtr());
    k_euler.setArg(iargs++, cl_vars_unsorted.getDevicePtr());
    k_euler.setArg(iargs++, cl_vars_sorted.getDevicePtr());
    k_euler.setArg(iargs++, cl_position.getDevicePtr());
    k_euler.setArg(iargs++, cl_SPHParams.getDevicePtr());
    k_euler.setArg(iargs++, ps->settings.dt); //time step



} 

void SPH::cpuEuler()
{
    float h = ps->settings.dt;
    for(int i = 0; i < num; i++)
    {
        float4 p = positions[i];
        float4 v = velocities[i];
        float4 f = forces[i];

		if (i == 0) {
			printf("==================================\n");
			printf("Euler: p[%d]= %f, %f, %f, %f\n", i, p.x, p.y, p.z, p.w);
			printf("       v[%d]= %f, %f, %f, %f\n", i, v.x, v.y, v.z, v.w);
		}

        //external force is gravity
        f.z += -9.8f;

        float speed = magnitude(f);
        if(speed > 600.0f) //velocity limit, need to pass in as struct
        {
            f.x *= 600.0f/speed;
            f.y *= 600.0f/speed;
            f.z *= 600.0f/speed;
        }

        float scale = params.simulation_scale;
        v.x += h*f.x / scale;
        v.y += h*f.y / scale;
        v.z += h*f.z / scale;
        
        p.x += h*v.x;
        p.y += h*v.y;
        p.z += h*v.z;
        p.w = 1.0f; //just in case

        velocities[i] = v;
        positions[i] = p;
    }
    //printf("v.z %f p.z %f \n", velocities[0].z, positions[0].z);
}

}

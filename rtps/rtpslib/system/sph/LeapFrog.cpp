#include "../SPH.h"

namespace rtps {

void SPH::loadLeapFrog()
{
    #include "leapfrog.cl"
    //printf("%s\n", euler_program_source.c_str());
    k_leapfrog = Kernel(ps->cli, leapfrog_program_source, "leapfrog");
  
    //TODO: fix the way we are wrapping buffers
    k_leapfrog.setArg(0, cl_position.cl_buffer[0]);
    k_leapfrog.setArg(1, cl_velocity.cl_buffer[0]);
    k_leapfrog.setArg(2, cl_veleval.cl_buffer[0]);
    k_leapfrog.setArg(3, cl_force.cl_buffer[0]);
    k_leapfrog.setArg(4, ps->settings.dt); //time step
    k_leapfrog.setArg(5, cl_params.cl_buffer[0]);

} 

void SPH::cpuLeapFrog()
{
    float h = ps->settings.dt;
    for(int i = 0; i < num; i++)
    {
        float4 p = positions[i];
        float4 v = velocities[i];
        float4 f = forces[i];

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
        float4 vnext = v;
        vnext.x += h*f.x / scale;
        vnext.y += h*f.y / scale;
        vnext.z += h*f.z / scale;
       
        p.x += h*vnext.x;
        p.y += h*vnext.y;
        p.z += h*vnext.z;
        p.w = 1.0f; //just in case

        velocities[i] = vnext;
        positions[i] = p;
         
        veleval[i].x = (v.x + vnext.x) *.5f;
        veleval[i].y = (v.y + vnext.y) *.5f;
        veleval[i].z = (v.z + vnext.z) *.5f;

    }
    //printf("v.z %f p.z %f \n", velocities[0].z, positions[0].z);
}

}

#include "../SPH.h"

namespace rtps {

/*
void SPH::loadEuler()
{
    #include "euler.cl"
    //printf("%s\n", euler_program_source.c_str());
    k_euler = Kernel(ps->cli, euler_program_source, "euler");
  
    //TODO: fix the way we are wrapping buffers
    k_euler.setArg(0, cl_position.cl_buffer[0]);
    k_euler.setArg(1, cl_velocity.cl_buffer[0]);
    k_euler.setArg(2, cl_force.cl_buffer[0]);
    k_euler.setArg(3, ps->settings.dt); //time step

} 
*/

void SPH::cpuLeapFrogStep1()
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

        float4 vnext = v;
        v.x += h*f.x/2.0f;
        v.y += h*f.y/2.0f;
        v.z += h*f.z/2.0f;
        
        p.x += h*v.x;
        p.y += h*v.y;
        p.z += h*v.z;
        p.w = 1.0f; //just in case

        velocities[i] = v;
        positions[i] = p;
    }
    //printf("v.z %f p.z %f \n", velocities[0].z, positions[0].z);
}
void SPH::cpuLeapFrogStep2()
{
    //v.x += xpsh[i] * xpsh_factor;
}

}

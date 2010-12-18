#include "cl_structs.h"
 
float magnitude(float4 vec)
{
    return sqrt(vec.x*vec.x + vec.y*vec.y + vec.z*vec.z);
}       
        
__kernel void euler(__global float4* pos, __global float4* vel, __global float4* force, float h, __constant struct SPHParams* params )
{
    unsigned int i = get_global_id(0);

    float4 p = pos[i];
    float4 v = vel[i];
    float4 f = force[i];

    //external force is gravity
    f.z += -9.8f;

    float speed = magnitude(f);
    if(speed > 600.0f) //velocity limit, need to pass in as struct
    {
        f *= 600.0f/speed;
    }

    v += h*f;
    p += h*v / params->simulation_scale;
    p.w = 1.0f; //just in case

    vel[i] = v;
    pos[i] = p;

}

#define STRINGIFY(A) #A

std::string viscosity_program_source = STRINGIFY(
//do the SPH viscosity calculations and update the velocity
__kernel void viscosity(__global float4* pos, __global float4* vel)
{
    unsigned int i = get_global_id(0);

}
);


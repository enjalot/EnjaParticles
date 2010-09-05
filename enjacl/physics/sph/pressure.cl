#define STRINGIFY(A) #A

std::string pressure_program_source = STRINGIFY(
//do the SPH pressure calculations and update the force
__kernel void pressure(__global float4* pos, __global float* density, __global float4* force)
{
    unsigned int i = get_global_id(0);

}
);


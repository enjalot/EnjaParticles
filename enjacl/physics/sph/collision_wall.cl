#define STRINGIFY(A) #A

std::string collision_wall_program_source = STRINGIFY(
//do the SPH pressure calculations and update the force
__kernel void collision_wall(__global float4* pos, __global float4* force, __global float4* walls)
{
    unsigned int i = get_global_id(0);

}
);

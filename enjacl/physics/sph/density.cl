#define STRINGIFY(A) #A

std::string density_program_source = STRINGIFY(
//update the SPH density
__kernel void density(__global float4* pos, __global float* density)
{
    unsigned int i = get_global_id(0);

}
);


#define STRINGIFY(A) #A

std::string euler_program_source = STRINGIFY(
__kernel void euler(__global float4* pos, __global float4* vel, __global float4* force, float h)
{
    unsigned int i = get_global_id(0);

    float4 p = pos[i];
    float4 v = vel[i];
    float4 f = force[i];

    f.z = -9.8f;

    v += h*f;
    p += h*v;
    p.w = 1.0f; //just in case

    vel[i] = v;
    pos[i] = p;
}
);


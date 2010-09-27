// Put in cl_macros.h
#define DENS 0
#define POS 1
#define VEL 2
#define FOR 3

#define numParticles num

#define FETCH(t, i) t[i]
#define FETCH_VEL(t, i) 	t[i+VEL*numParticles]
#define FETCH_POS(t, i) 	t[i+POS*numParticles]
#define FETCH_DENS(t, i) 	t[i+DENS*numParticles]
#define FETCH_FOR(t, i) 	t[i+FOR*numParticles]
#define pos(i) 		vars_sorted[i+POS*numParticles]
#define vel(i) 		vars_sorted[i+VEL*numParticles]
#define density(i) 	vars_sorted[i+DENS*numParticles].x
#define force(i) 	vars_sorted[i+FOR*numParticles]

typedef struct GE_SPHParams
{
    float3 grid_min;            //float3s are really float4 in opencl 1.0 & 1.1
    float3 grid_max;            //so we have padding in C++ definition
    float mass;
    float rest_distance;
    float smoothing_distance;
    float simulation_scale;
    float boundary_stiffness;
    float boundary_dampening;
    float boundary_distance;
    float EPSILON;
    float PI;       //delicious
    float K;        //speed of sound
} GE_SPHParams;


float magnitude(float4 vec)
{
    return sqrt(vec.x*vec.x + vec.y*vec.y + vec.z*vec.z);
}
float dist_squared(float4 vec)
{
    return vec.x*vec.x + vec.y*vec.y + vec.z*vec.z;
}

//----------------------------------------------------------------------
// ***** TODO *****
// Replace pos by vars_sorted[
//----------------------------------------------------------------------
//__kernel void density(__global float4* pos, __global float* density, __constant struct GE_SPHParams* params, __global float4* error)
__kernel void ge_density(__constant int nb_vars, __global float4* vars_sorted, __constant struct GE_SPHParams* params, __global float4* error)
{
    unsigned int i = get_global_id(0);
	int num = get_global_size(0);
    //int num = 1024;

    float h = params->smoothing_distance;
    //stuff from Tim's code (need to match #s to papers)
    float alpha = 315.f/208.f/params->PI/h/h/h;
    //float h9 = h*h*h * h*h*h * h*h*h;
    //float alpha = 315.f/64.0f/params->PI/h9;

    float4 p = pos(i) * params->simulation_scale;
    //density[i] = 0.0f;
    density(i) = 0.0f;

    //super slow way, we need to use grid + sort method to get nearest neighbors
    //this code should never see the light of day on a GPU... just sayin
    for(int j = 0; j < num; j++)
    {
        if(j == i) continue;
        //float4 pj = pos[j] * params->simulation_scale;
        float4 pj = pos(j)* params->simulation_scale;
        float4 r = p - pj;
        error[i] = r;
        float rlen = magnitude(r);
        if(rlen < h)
        {
            float r2 = dist_squared(r);
            float re2 = h*h;
            if(r2/re2 <= 4.f)
            {
                float R = sqrt(r2/re2);
                float Wij = alpha*(2.f/3.f - 9.f*R*R/8.f + 19.f*R*R*R/24.f - 5.f*R*R*R*R/32.f);
                //float hr2 = (h*h - r2);
                //float Wij = alpha * hr2*hr2*hr2;
                //density[i] += params->mass * Wij;
                density(i) += params->mass * Wij;
            }
        }
    }
}
//);

//----------------------------------------------------------------------
//----------------------------------------------------------------------
#if 0
__kernel void density(__global float4* pos, __global float* density, __constant struct GE_SPHParams* params, __global float4* error)
{
    unsigned int i = get_global_id(0);
    int num = 1024;

    float h = params->smoothing_distance;
    //stuff from Tim's code (need to match #s to papers)
    float alpha = 315.f/208.f/params->PI/h/h/h;
    //float h9 = h*h*h * h*h*h * h*h*h;
    //float alpha = 315.f/64.0f/params->PI/h9;

    float4 p = pos[i] * params->simulation_scale;
    density[i] = 0.0f;

    //super slow way, we need to use grid + sort method to get nearest neighbors
    //this code should never see the light of day on a GPU... just sayin
    for(int j = 0; j < num; j++)
    {
        if(j == i) continue;
        float4 pj = pos[j] * params->simulation_scale;
        float4 r = p - pj;
        error[i] = r;
        float rlen = magnitude(r);
        if(rlen < h)
        {
            float r2 = dist_squared(r);
            float re2 = h*h;
            if(r2/re2 <= 4.f)
            {
                float R = sqrt(r2/re2);
                float Wij = alpha*(2.f/3.f - 9.f*R*R/8.f + 19.f*R*R*R/24.f - 5.f*R*R*R*R/32.f);
                //float hr2 = (h*h - r2);
                //float Wij = alpha * hr2*hr2*hr2;
                density[i] += params->mass * Wij;
            }
        }
    }
}
);
//----------------------------------------------------------------------
#endif

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
    float4 grid_min;            //float3s are really float4 in opencl 1.0 & 1.1
    float4 grid_max;            //so we have padding in C++ definition
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
__kernel void ge_pressure(__constant int nb_vars, __global float4* vars_sorted, __constant struct GE_SPHParams* params)
{
    unsigned int i = get_global_id(0);
    float4 f = (float4)(0.0f, 0.0f, 0.0f, 0.0f);

    //obviously this is going to be passed in as a parameter
    //int num = 1024;
	int num = get_global_size(0);

    //float4 p = pos[i] * params->simulation_scale;
    float4 p = pos(i) * params->simulation_scale;


    float h = params->smoothing_distance;

    //stuff from Tim's code (need to match #s to papers)
    //float alpha = 315.f/208.f/params->PI/h/h/h;
    float h6 = h*h*h * h*h*h;
    float alpha = 45.f/params->PI/h6;

    //super slow way, we need to use grid + sort method to get nearest neighbors
    //this code should never see the light of day on a GPU... just sayin
    for(int j = 0; j < num; j++)
    {
        if(j == i) continue;
        //float4 pj = pos[j] * params->simulation_scale;
        float4 pj = pos(j) * params->simulation_scale;
        float4 r = p - pj;
        float rlen = magnitude(r);
        if(rlen < h)
        {
            float r2 = rlen*rlen;
            float re2 = h*h;
            if(r2/re2 <= 4.f)
            {
                //float R = sqrt(r2/re2);
                //float Wij = alpha*(-2.25f + 2.375f*R - .625f*R*R);
                float hr2 = (h - rlen);
                float Wij = alpha * hr2*hr2*hr2/rlen;
                //from tim's code
                /*
                float Pi = 1.013E5*(pow(density[i]/1000.0f, 7.0f) - 1.0f);
                float Pj = 1.013E5*(pow(density[j]/1000.0f, 7.0f) - 1.0f);
                float kern = params->mass * Wij * (Pi + Pj) / (density[i] * density[j]);
                */
                //form simple GE_SPH in Krog's thesis
                //float Pi = params->K*(density[i] - 1000.0f); //rest density
                //float Pj = params->K*(density[j] - 1000.0f); //rest density
                float Pi = params->K*(density(i) - 1000.0f); //rest density
                float Pj = params->K*(density(j) - 1000.0f); //rest density
                float kern = params->mass * -1.0f * Wij * (Pi + Pj) / (2.0f * density(j));
                //float kern = params->mass * -1.0f * Wij * (Pi + Pj) / (density[i] * density[j]);
                f += kern * r;
            }

        }
    }
    //force[i] -= f;
    force(i) -= f;
}

//----------------------------------------------------------------------
#if 0
__kernel void pressure(__global float4* pos, __global float* density, __global float4* force, __constant struct GE_SPHParams* params)
{
    unsigned int i = get_global_id(0);
    float4 f = (float4)(0.0f, 0.0f, 0.0f, 0.0f);

    float4 p = pos[i] * params->simulation_scale;

    //obviously this is going to be passed in as a parameter
    int num = 1024;
    float h = params->smoothing_distance;

    //stuff from Tim's code (need to match #s to papers)
    //float alpha = 315.f/208.f/params->PI/h/h/h;
    float h6 = h*h*h * h*h*h;
    float alpha = 45.f/params->PI/h6;


    //super slow way, we need to use grid + sort method to get nearest neighbors
    //this code should never see the light of day on a GPU... just sayin
    for(int j = 0; j < num; j++)
    {
        if(j == i) continue;
        float4 pj = pos[j] * params->simulation_scale;
        float4 r = p - pj;
        float rlen = magnitude(r);
        if(rlen < h)
        {
            float r2 = rlen*rlen;
            float re2 = h*h;
            if(r2/re2 <= 4.f)
            {
                //float R = sqrt(r2/re2);
                //float Wij = alpha*(-2.25f + 2.375f*R - .625f*R*R);
                float hr2 = (h - rlen);
                float Wij = alpha * hr2*hr2*hr2/rlen;
                //from tim's code
                /*
                float Pi = 1.013E5*(pow(density[i]/1000.0f, 7.0f) - 1.0f);
                float Pj = 1.013E5*(pow(density[j]/1000.0f, 7.0f) - 1.0f);
                float kern = params->mass * Wij * (Pi + Pj) / (density[i] * density[j]);
                */
                //form simple GE_SPH in Krog's thesis
                float Pi = params->K*(density[i] - 1000.0f); //rest density
                float Pj = params->K*(density[j] - 1000.0f); //rest density
                float kern = params->mass * -1.0f * Wij * (Pi + Pj) / (2.0f * density[j]);
                //float kern = params->mass * -1.0f * Wij * (Pi + Pj) / (density[i] * density[j]);
                f += kern * r;
            }

        }
    }
    force[i] -= f;
}
#endif



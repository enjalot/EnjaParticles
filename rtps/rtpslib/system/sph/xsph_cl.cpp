#include "cl_structs.h"

float magnitude(float4 vec)
{
    return sqrt(vec.x*vec.x + vec.y*vec.y + vec.z*vec.z);
}
float dist_squared(float4 vec)
{
    return vec.x*vec.x + vec.y*vec.y + vec.z*vec.z;
}

       
__kernel void xsph(__global float4* pos, __global float4* veleval, __global float* density, __global float4* force, __global float4* xsph, __constant struct SPHParams* params)
{
    unsigned int i = get_global_id(0);
    int num = params->num;
    if(i > num) return;

    float sadf = 5;
    float4 p = pos[i] * params->simulation_scale;
    float4 v = veleval[i];
    float di = density[i];
 
    float h = params->smoothing_distance;

    //stuff from Tim's code (need to match #s to papers)
    //float alpha = 315.f/208.f/params->PI/h/h/h;
    float h9 = h*h*h * h*h*h * h*h*h;
    float alpha = 315.f / 64.0f / params->PI / h9;

    float4 f = (float4)(0.0f, 0.0f, 0.0f, 0.0f);

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
                float4 vj = veleval[j];
                float dj = density[j];

                float hr2 = (h*h - dist_squared(r));
                float Wij = alpha * hr2*hr2*hr2;
                float fc = 2.0 * params->mass * Wij / (di + dj);
                f += fc * (vj - v);
                //f = (float4)(fc, fc, fc, 0.0f);
            }

        }
    }
    xsph[i] = f;

}

#include "cl_structs.h"

float magnitude(float4 vec)
{
    return sqrt(vec.x*vec.x + vec.y*vec.y + vec.z*vec.z);
}
float dist_squared(float4 vec)
{
    return vec.x*vec.x + vec.y*vec.y + vec.z*vec.z;
}


__kernel void viscosity(__global float4* pos, __global float4* veleval, __global float* density, __global float4* force, __constant struct SPHParams* params)
{
    unsigned int i = get_global_id(0);
    int num = params->num;
    if (i > num) return;


    float4 p = pos[i] * params->simulation_scale;
    float4 v = veleval[i];// * params->simulation_scale;
    //float4 v = vel[i] * params->simulation_scale;
    float4 f = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
    //float mu = 1.0f;
    float mu = 1.001f;

    float h = params->smoothing_distance;

    //stuff from Tim's code (need to match #s to papers)
    //float alpha = 315.f/208.f/params->PI/h/h/h;
    float h6 = h*h*h * h*h*h;
    float alpha = 45.f/params->PI/h6;

    float di = density[i];

    //super slow way, we need to use grid + sort method to get nearest neighbors
    //this code should never see the light of day on a GPU... just sayin
    for (int j = 0; j < num; j++)
    {
        if (j == i) continue;
        float4 pj = pos[j] * params->simulation_scale;
        float4 r = p - pj;
        float rlen = magnitude(r);
        if (rlen < h)
        {
            float r2 = rlen*rlen;
            float re2 = h*h;
            if (r2/re2 <= 4.f)
            {

                float dj = density[j];
                float4 vj = veleval[j];// * params->simulation_scale;
                //float4 vj = vel[j] * params->simulation_scale;

                //float R = sqrt(r2/re2);
                //float Wij = alpha*(-2.25f + 2.375f*R - .625f*R*R);
                float Wij = alpha * (h - rlen);
                //from tim's code
                //form simple SPH in Krog's thesis
                f += mu * params->mass * Wij * (vj - v) / (di * dj);
                //f = (float4)(0.5f, 0.5f, 0.5f, 0.0f);
            }

        }
    }
    force[i] += f;

}

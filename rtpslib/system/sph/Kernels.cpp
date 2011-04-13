#include "../SPH.h"

namespace rtps
{

    float SPH::Wpoly6(float4 r, float h)
    {
        float h9 = h*h*h * h*h*h * h*h*h;
        float alpha = 315.f/64.0f/params.PI/h9;
        float r2 = dist_squared(r);
        float hr2 = (h*h - r2);
        float Wij = alpha * hr2*hr2*hr2;
        return Wij;
    }


    float SPH::Wspiky(float4 r, float h)
    {
        float h6 = h*h*h * h*h*h;
        float alpha = -45.f/params.PI/h6;
        float rlen = magnitude(r);
        float hr2 = (h - rlen);
        float Wij = alpha * hr2*hr2/rlen;
        return Wij;
    }



}

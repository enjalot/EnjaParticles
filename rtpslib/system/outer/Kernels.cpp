/****************************************************************************************
* Real-Time Particle System - An OpenCL based Particle system developed to run on modern GPUs. Includes SPH fluid simulations.
* version 1.0, September 14th 2011
* 
* Copyright (C) 2011 Ian Johnson, Andrew Young, Gordon Erlebacher, Myrna Merced, Evan Bollig
* 
* This software is provided 'as-is', without any express or implied
* warranty.  In no event will the authors be held liable for any damages
* arising from the use of this software.
* 
* Permission is granted to anyone to use this software for any purpose,
* including commercial applications, and to alter it and redistribute it
* freely, subject to the following restrictions:
* 
* 1. The origin of this software must not be misrepresented; you must not
* claim that you wrote the original software. If you use this software
* in a product, an acknowledgment in the product documentation would be
* appreciated but is not required.
* 2. Altered source versions must be plainly marked as such, and must not be
* misrepresented as being the original software.
* 3. This notice may not be removed or altered from any source distribution.
****************************************************************************************/


#include "../OUTER.h"

namespace rtps
{
namespace outer
{

    float OUTER::Wpoly6(float4 r, float h)
    {
        float h9 = h*h*h * h*h*h * h*h*h;
        float alpha = 315.f/64.0f/params.PI/h9;
        float r2 = dist_squared(r);
        float hr2 = (h*h - r2);
        float Wij = alpha * hr2*hr2*hr2;
        return Wij;
    }


    float OUTER::Wspiky(float4 r, float h)
    {
        float h6 = h*h*h * h*h*h;
        float alpha = -45.f/params.PI/h6;
        float rlen = magnitude(r);
        float hr2 = (h - rlen);
        float Wij = alpha * hr2*hr2/rlen;
        return Wij;
    }



}
}

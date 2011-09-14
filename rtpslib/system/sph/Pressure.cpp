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


#include "SPH.h"
#include <math.h>



namespace rtps
{

    float SPH::Wspiky(float4 r, float h)
    {
        float h6 = h*h*h * h*h*h;
        float alpha = -45.f/sphp.PI/h6;
        float rlen = magnitude(r);
        float hr2 = (h - rlen);
        float Wij = alpha * hr2*hr2/rlen;
        return Wij;
    }
    /*
        //stuff from Tim's code (need to match #s to papers)
        //float alpha = 315.f/208.f/sphp->PI/h/h/h;
        //float R = sqrt(r2/re2);
        //float Wij = alpha*(-2.25f + 2.375f*R - .625f*R*R);
    
     */

    void SPH::cpuPressure()
    {

        float scale = sphp.simulation_scale;
        float h = sphp.smoothing_distance;

        for (int i = 0; i < num; i++)
        {

            float4 p = positions[i];
            p = float4(p.x * scale, p.y * scale, p.z * scale, p.w * scale);

            float4 f = float4(0.0f, 0.0f, 0.0f, 0.0f);

            //super slow way, we need to use grid + sort method to get nearest neighbors
            //this code should never see the light of day on a GPU... just sayin
            for (int j = 0; j < num; j++)
            {
                if (j == i) continue;
                float4 pj = positions[j];

                pj = float4(pj.x * scale, pj.y * scale, pj.z * scale, pj.w * scale);
                float4 r = float4(p.x - pj.x, p.y - pj.y, p.z - pj.z, p.w - pj.w);

                float rlen = magnitude(r);
                if (rlen < h)
                {
                    float r2 = rlen*rlen;
                    float re2 = h*h;
                    if (r2/re2 <= 4.f)
                    {
                        //from tim's code
                        /*
                        float Pi = 1.013E5*(pow(density[i]/1000.0f, 7.0f) - 1.0f);
                        float Pj = 1.013E5*(pow(density[j]/1000.0f, 7.0f) - 1.0f);
                        float kern = sphp->mass * Wij * (Pi + Pj) / (density[i] * density[j]);
                        */
                        //from simple SPH in Krog's thesis
                        float Pi = sphp.K*(densities[i] - 1000.0f); //rest density
                        float Pj = sphp.K*(densities[j] - 1000.0f); //rest density
                        //float kern = sphp->mass * -1.0f * Wij * (Pi + Pj) / (2.0f * density[j]);
                        float Wij = Wspiky(r, h);
                        float kern = sphp.mass * -.5f * Wij * (Pi + Pj) / (densities[i] * densities[j]);
                        //float kern = sphp.mass * -.5f * Wij * (Pi/(densities[i]*densities[i]) + Pj/(densities[j]*densities[j]));
                        f.x += kern * r.x;
                        f.y += kern * r.y;
                        f.z += kern * r.z;
                    }

                }
            }
            //printf("forces[%d] = %f %f %f\n", i, f.x, f.y, f.z);
            forces[i] = f;

        }
    }




}

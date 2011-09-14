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

    float OUTER::Wviscosity(float4 r, float h)
    {
        //from simple OUTER in Krog's thesis
        float h6 = h*h*h * h*h*h;
        float alpha = 45.f/sphp.PI/h6;
        float rlen = magnitude(r);
        float Wij = alpha * (h - rlen);
        return Wij;

    }

    void OUTER::cpuViscosity()
    {

		#if 0
        float scale = sphp.simulation_scale;
        float h = sphp.smoothing_distance;
        float mu = 1.001f; //viscosity coefficient (how to select?)
        float4* vel;
        if (integrator == EULER)
        {
            vel = &velocities[0];
        }
        else if (integrator == LEAPFROG)
        {
            vel = &veleval[0];
        }

        for (int i = 0; i < num; i++)
        {

            float4 p = positions[i];
            float4 v = vel[i];
            p = float4(p.x * scale, p.y * scale, p.z * scale, p.w * scale);
            //v = float4(v.x * scale, v.y * scale, v.z * scale, v.w * scale);

            float4 f = float4(0.0f, 0.0f, 0.0f, 0.0f);

            //stuff from Tim's code (need to match #s to papers)
            //float alpha = 315.f/208.f/sphp->PI/h/h/h;

            for (int j = 0; j < num; j++)
            {
                if (j == i) continue;
                float4 pj = positions[j];
                float4 vj = vel[j];
                pj = float4(pj.x * scale, pj.y * scale, pj.z * scale, pj.w * scale);
                //vj = float4(vj.x * scale, vj.y * scale, vj.z * scale, vj.w * scale);
                float4 r = float4(p.x - pj.x, p.y - pj.y, p.z - pj.z, p.w - pj.w);

                float rlen = magnitude(r);
                if (rlen < h)
                {
                    float r2 = rlen*rlen;
                    float re2 = h*h;
                    if (r2/re2 <= 4.f)
                    {
                        //float R = sqrt(r2/re2);
                        //float Wij = alpha*(-2.25f + 2.375f*R - .625f*R*R);
                        float Wij = Wviscosity(r, h);
                        float fcoeff = mu * sphp.mass * Wij / (densities[j] * densities[i]);
                        f.x += fcoeff * (vj.x - v.x); 
                        f.y += fcoeff * (vj.y - v.y); 
                        f.z += fcoeff * (vj.z - v.z); 
                    }

                }
            }
            forces[i].x += f.x;
            forces[i].y += f.y;
            forces[i].z += f.z;
        }
		#endif

    }

}

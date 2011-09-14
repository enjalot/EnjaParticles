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


#include "../Simple.h"
#include <math.h>

namespace rtps
{

    void Simple::loadEuler()
    {
        std::string path(SIMPLE_CL_SOURCE_DIR);
        path += "/euler_cl.cl";
        k_euler = Kernel(ps->cli, path, "euler");

        k_euler.setArg(0, cl_position.getDevicePtr());
        k_euler.setArg(1, cl_velocity.getDevicePtr());
        k_euler.setArg(2, cl_force.getDevicePtr());
        k_euler.setArg(3, cl_color.getDevicePtr());
        k_euler.setArg(4, ps->settings->dt); //time step

    } 
    void Simple::cpuEuler()
    {
        //printf("in cpuEuler\n");
        float h = ps->settings->dt;
        //printf("h: %f\n", h);


        for (int i = 0; i < num; i++)
        {
            float4 p = positions[i];
            float4 v = velocities[i];
            float4 f = forces[i];

            v.x += h*f.x;
            v.y += h*f.y;
            v.z += h*f.z;


            p.x += h*v.x;
            p.y += h*v.y;
            p.z += h*v.z;
            p.w = 1.0f; //just in case

            velocities[i] = v;
            positions[i] = p;


            float colx = v.x;
            float coly = v.y;
            float colz = v.z;
            if (colx < 0)
            {
                colx = -1.0f*colx;
            }
            if (colx > 1)
            {
                colx = 1.0f;
            }
            if (coly < 0)
            {
                coly = -1.0f*coly;
            }
            if (coly > 1)
            {
                coly = 1.0f;
            }
            if (colz < 0)
            {
                colz = -1.0f*colz;
            }
            if (colz > 1)
            {
                colz = 1.0f;
            }

            colors[i].x = colx;
            colors[i].y = coly;
            colors[i].z = colz;
            colors[i].w = 1.0f;

        }
        //printf("v.z %f p.z %f \n", velocities[0].z, positions[0].z);
    }

}

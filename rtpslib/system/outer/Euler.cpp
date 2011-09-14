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
    Euler::Euler(std::string path, CL* cli_, EB::Timer* timer_)
    {
        cli = cli_;
        timer = timer_;
 
        printf("create euler kernel\n");
        path += "/euler.cl";
        k_euler = Kernel(cli, path, "euler");
    } 
    
    void Euler::execute(int num,
                    float dt,
                    Buffer<float4>& pos_u,
                    Buffer<float4>& pos_s,
                    Buffer<float4>& vel_u,
                    Buffer<float4>& vel_s,
                    Buffer<float4>& force_s,
                    //Buffer<float4>& uvars, 
                    //Buffer<float4>& svars, 
                    Buffer<unsigned int>& indices,
                    //params
                    Buffer<OUTERParams>& sphp,
                    //debug params
                    Buffer<float4>& clf_debug,
                    Buffer<int4>& cli_debug)
    {

        int iargs = 0;
        //k_euler.setArg(iargs++, uvars.getDevicePtr());
        //k_euler.setArg(iargs++, svars.getDevicePtr());
        k_euler.setArg(iargs++, pos_u.getDevicePtr());
        k_euler.setArg(iargs++, pos_s.getDevicePtr());
        k_euler.setArg(iargs++, vel_u.getDevicePtr());
        k_euler.setArg(iargs++, vel_s.getDevicePtr());
        k_euler.setArg(iargs++, force_s.getDevicePtr());
        k_euler.setArg(iargs++, indices.getDevicePtr());
        k_euler.setArg(iargs++, sphp.getDevicePtr());
        k_euler.setArg(iargs++, dt); //time step

        int local_size = 128;
        k_euler.execute(num, local_size);

    }
} // Namespace

    void OUTER::cpuEuler()
    {
		#if 0
        float h = ps->settings->dt;
        for (int i = 0; i < num; i++)
        {
            float4 p = positions[i];
            float4 v = velocities[i];
            float4 f = forces[i];

            if (i == 0)
            {
                printf("==================================\n");
                printf("Euler: p[%d]= %f, %f, %f, %f\n", i, p.x, p.y, p.z, p.w);
                printf("       v[%d]= %f, %f, %f, %f\n", i, v.x, v.y, v.z, v.w);
            }

            //external force is gravity
            f.z += -9.8f;

            float speed = magnitude(f);
            if (speed > 600.0f) //velocity limit, need to pass in as struct
            {
                f.x *= 600.0f/speed;
                f.y *= 600.0f/speed;
                f.z *= 600.0f/speed;
            }

            float scale = sphp.simulation_scale;
            v.x += h*f.x / scale;
            v.y += h*f.y / scale;
            v.z += h*f.z / scale;

            p.x += h*v.x;
            p.y += h*v.y;
            p.z += h*v.z;
            p.w = 1.0f; //just in case

            velocities[i] = v;
            positions[i] = p;
        }
        //printf("v.z %f p.z %f \n", velocities[0].z, positions[0].z);
		#endif
    }

}

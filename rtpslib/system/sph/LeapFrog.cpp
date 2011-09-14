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


#include "../SPH.h"

namespace rtps
{
namespace sph
{

    LeapFrog::LeapFrog(std::string path, CL* cli_, EB::Timer* timer_)
    {
        cli = cli_;
        timer = timer_;
 
        printf("create leapfrog kernel\n");
        path += "/leapfrog.cl";
        k_leapfrog = Kernel(cli, path, "leapfrog");

    } 
    void LeapFrog::execute(int num,
                    float dt,
                    Buffer<float4>& pos_u,
                    Buffer<float4>& pos_s,
                    Buffer<float4>& vel_u,
                    Buffer<float4>& vel_s,
                    Buffer<float4>& veleval_u,
                    Buffer<float4>& force_s,
                    Buffer<float4>& xsph_s,
                    //Buffer<float4>& uvars, 
                    //Buffer<float4>& svars, 
                    Buffer<unsigned int>& indices,
                    //params
                    Buffer<SPHParams>& sphp,
                    //debug params
                    Buffer<float4>& clf_debug,
                    Buffer<int4>& cli_debug)
    {

        int iargs = 0;
        //k_leapfrog.setArg(iargs++, uvars.getDevicePtr());
        //k_leapfrog.setArg(iargs++, svars.getDevicePtr());
        k_leapfrog.setArg(iargs++, pos_u.getDevicePtr());
        k_leapfrog.setArg(iargs++, pos_s.getDevicePtr());
        k_leapfrog.setArg(iargs++, vel_u.getDevicePtr());
        k_leapfrog.setArg(iargs++, vel_s.getDevicePtr());
        k_leapfrog.setArg(iargs++, veleval_u.getDevicePtr());
        k_leapfrog.setArg(iargs++, force_s.getDevicePtr());
        k_leapfrog.setArg(iargs++, xsph_s.getDevicePtr());
        k_leapfrog.setArg(iargs++, indices.getDevicePtr());
        //leapfrog.setArg(iargs++, color.getDevicePtr());
        k_leapfrog.setArg(iargs++, sphp.getDevicePtr());
        k_leapfrog.setArg(iargs++, dt); //time step

        int local_size = 128;
        float gputime = k_leapfrog.execute(num, local_size);
        if(gputime > 0)
            timer->set(gputime);


} //namespace sph

#if 0
#define DENS 0
#define POS 1
#define VEL 2

        printf("************ LeapFrog **************\n");
            int nbc = num+5;
            std::vector<float4> poss(nbc);
            std::vector<float4> uposs(nbc);
            std::vector<float4> dens(nbc);

            //cl_vars_sorted.copyToHost(dens, DENS*sphp.max_num);
            cl_vars_sorted.copyToHost(poss, POS*sphp.max_num);
            cl_vars_unsorted.copyToHost(uposs, POS*sphp.max_num);

            for (int i=0; i < nbc; i++)
            //for (int i=0; i < 10; i++) 
            {
                poss[i] = poss[i] / sphp.simulation_scale;
                //printf("-----\n");
                //printf("clf_debug: %f, %f, %f, %f\n", clf[i].x, clf[i].y, clf[i].z, clf[i].w);
                printf("pos sorted: %f, %f, %f, %f\n", poss[i].x, poss[i].y, poss[i].z, poss[i].w);
                printf("pos unsorted: %f, %f, %f, %f\n", uposs[i].x, uposs[i].y, uposs[i].z, uposs[i].w);
                //printf("dens sorted: %f, %f, %f, %f\n", dens[i].x, dens[i].y, dens[i].z, dens[i].w);
            }

#endif

        /*
         * enables us to cut off after a couple iterations
         * by setting cut = 1 from some other function
        if(cut >= 1)
        {
            if (cut == 2) {exit(0);}
            cut++;
        }
        */


    }

    void SPH::cpuLeapFrog()
    {
        float h = ps->settings->dt;
        for (int i = 0; i < num; i++)
        {
            float4 p = positions[i];
            float4 v = velocities[i];
            float4 f = forces[i];

            //external force is gravity
            f.z += -9.8f;

            float speed = magnitude(f);
            if (speed > 600.0f) //velocity limit, need to pass in as struct
            {
                f.x *= 600.0f/speed;
                f.y *= 600.0f/speed;
                f.z *= 600.0f/speed;
            }

            float4 vnext = v;
            vnext.x += h*f.x;
            vnext.y += h*f.y;
            vnext.z += h*f.z;

            float xsphfactor = .1f;
            vnext.x += xsphfactor * xsphs[i].x;
            vnext.y += xsphfactor * xsphs[i].y;
            vnext.z += xsphfactor * xsphs[i].z;

            float scale = sphp.simulation_scale;
            p.x += h*vnext.x / scale;
            p.y += h*vnext.y / scale;
            p.z += h*vnext.z / scale;
            p.w = 1.0f; //just in case

            veleval[i].x = (v.x + vnext.x) *.5f;
            veleval[i].y = (v.y + vnext.y) *.5f;
            veleval[i].z = (v.z + vnext.z) *.5f;

            velocities[i] = vnext;
            positions[i] = p;

        }
        //printf("v.z %f p.z %f \n", velocities[0].z, positions[0].z);
    }

} //namespace rtps

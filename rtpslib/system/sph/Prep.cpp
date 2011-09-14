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

#include <string>

namespace rtps
{
namespace sph
{

    Prep::Prep(CL* cli_, EB::Timer* timer_)
    {
        cli = cli_;
        timer = timer_;
        printf("create prep kernel\n");
        std::string path(SPH_CL_SOURCE_DIR);
        path = path + "/prep.cl";
        k_prep = Kernel(cli, path, "prep");
    }

    void Prep::execute(int num,
                    int stage,
                    Buffer<float4>& pos_u,
                    Buffer<float4>& pos_s,
                    Buffer<float4>& vel_u,
                    Buffer<float4>& vel_s,
                    Buffer<float4>& veleval_u,
                    Buffer<float4>& veleval_s,
                    Buffer<float4>& color_u,
                    Buffer<float4>& color_s,
                    //Buffer<float4>& uvars, 
                    //Buffer<float4>& svars, 
                    Buffer<unsigned int>& indices,
                    //params
                    Buffer<SPHParams>& sphp,
                    //Buffer<GridParams>& gp,
                    //debug params
                    Buffer<float4>& clf_debug,
                    Buffer<int4>& cli_debug)
    {
        /**
         * sometimes we only want to copy positions
         * this should probably be replaced with Scopy
         * i don't think straight copy is most efficient...
         */

        printf("num: %d, stage: %d\n", num, stage);
        int args = 0;
        k_prep.setArg(args++, stage);
        k_prep.setArg(args++, pos_u.getDevicePtr());
        k_prep.setArg(args++, pos_s.getDevicePtr());
        k_prep.setArg(args++, vel_u.getDevicePtr());
        k_prep.setArg(args++, vel_s.getDevicePtr());
        k_prep.setArg(args++, veleval_u.getDevicePtr());
        k_prep.setArg(args++, veleval_s.getDevicePtr());
        //k_prep.setArg(args++, uvars.getDevicePtr());
        //k_prep.setArg(args++, svars.getDevicePtr()); 
        k_prep.setArg(args++, color_u.getDevicePtr());
        k_prep.setArg(args++, color_s.getDevicePtr());
        k_prep.setArg(args++, indices.getDevicePtr());
        k_prep.setArg(args++, sphp.getDevicePtr());


        int ctaSize = 128; // work group size
        // Hash based on unscaled data
        try
        {
            k_prep.execute(num, ctaSize);
        }
        catch (cl::Error er)
        {
            printf("ERROR(prep): %s(%s)\n", er.what(), oclErrorString(er.err()));
            exit(1);
        }

    }

}//namespace sph
}//namespace rtps

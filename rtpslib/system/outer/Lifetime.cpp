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

    Lifetime::Lifetime(std::string path, CL* cli_, EB::Timer* timer_, std::string filename)
    {
        cli = cli_;
        timer = timer_;
 
        printf("create liftime kernel\n");
        path += "/" + filename;
        k_lifetime = Kernel(cli, path, "lifetime");

    } 
    void Lifetime::execute(int num,
                    float increment,
                    Buffer<float4>& pos_u,
                    Buffer<float4>& color_u, 
                    Buffer<float4>& color_s, 
                    Buffer<unsigned int>& indices,
                    Buffer<float4>& clf_debug,
                    Buffer<int4>& cli_debug)
    {

        int iargs = 0;
        k_lifetime.setArg(iargs++, num); //time step
        k_lifetime.setArg(iargs++, increment); //time step
        k_lifetime.setArg(iargs++, pos_u.getDevicePtr());
        k_lifetime.setArg(iargs++, color_u.getDevicePtr());
        k_lifetime.setArg(iargs++, color_s.getDevicePtr());
        k_lifetime.setArg(iargs++, indices.getDevicePtr());
        //lifetime.setArg(iargs++, color.getDevicePtr());
        k_lifetime.setArg(iargs++, clf_debug.getDevicePtr());
        k_lifetime.setArg(iargs++, cli_debug.getDevicePtr());



        int local_size = 128;
        float gputime = k_lifetime.execute(num, local_size);
        if(gputime > 0)
            timer->set(gputime);



#if 0

        if(num > 0)
        {
            printf("************ Lifetime**************\n");
            int nbc = num + 5;
            std::vector<int4> cli(nbc);
            std::vector<float4> clf(nbc);

            cli_debug.copyToHost(cli);
            clf_debug.copyToHost(clf);

            for (int i=0; i < nbc; i++)
            {
                //printf("-----\n");
                printf("clf_debug: %f, %f, %f, %f\n", clf[i].x, clf[i].y, clf[i].z, clf[i].w);
                //printf("cli_debug: %d, %d, %d, %d\n", cli[i].x, cli[i].y, cli[i].z, cli[i].w);
            }
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


}
}

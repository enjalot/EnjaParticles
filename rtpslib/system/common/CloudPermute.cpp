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


#include "CloudPermute.h"

#include <string>

namespace rtps
{

	//----------------------------------------------------------------------
    CloudPermute::CloudPermute(std::string path, CL* cli_, EB::Timer* timer_)
    {
        cli = cli_;
        timer = timer_;
        printf("create permute kernel\n");
        path = path + "/cloud_permute.cl";
        k_permute = Kernel(cli, path, "cloud_permute");
        
    }

    void CloudPermute::execute(int num,
                    //input
                    Buffer<float4>& pos_u,
                    Buffer<float4>& pos_s,
                    Buffer<float4>& normal_u,
                    Buffer<float4>& normal_s,
                    Buffer<unsigned int>& indices,
                    //params
                    Buffer<GridParams>& gp,
                    //debug params
                    Buffer<float4>& clf_debug,
                    Buffer<int4>& cli_debug)
    {
		//printf("CloudPermute: num= %d\n", num);
        
        int iarg = 0;
        k_permute.setArg(iarg++, num);
        k_permute.setArg(iarg++, pos_u.getDevicePtr());
        k_permute.setArg(iarg++, pos_s.getDevicePtr());
        k_permute.setArg(iarg++, normal_u.getDevicePtr());
        k_permute.setArg(iarg++, normal_s.getDevicePtr());
        k_permute.setArg(iarg++, indices.getDevicePtr());

        int workSize = 64;
        
        //printf("cloudPermute, before kernel exec, num=  %d\n", num);
        try
        {
			float gputime;
            gputime = k_permute.execute(num, workSize);
            if(gputime > 0)
                timer->set(gputime);

        }
        catch (cl::Error er)
        {
            printf("ERROR(data structures): %s(%s)\n", er.what(), oclErrorString(er.err()));
        }

        
#if 0
        //printCloudPermuteDiagnostics();

        printf("**************** CloudPermute Diagnostics ****************\n");
        int nbc = num;
        printf("(cloud) num: %d\n", num);
        printf("num particles: %d\n", num);

        std::vector<unsigned int> is(nbc);
        std::vector<unsigned int> ie(nbc);
        
        ci_end.copyToHost(ie);
        ci_start.copyToHost(is);


        for(int i = 0; i < nbc; i++)
        {
            if (is[i] != -1)// && ie[i] != 0)
            {
                //nb = ie[i] - is[i];
                //nb_particles += nb;
                printf("cell: %d indices start: %d indices stop: %d\n", i, is[i], ie[i]);
            }
        }

#endif

#if 0
        //print out elements from the sorted arrays
#define DENS 0
#define POS 1
#define VEL 2

			printf("gordon\n");
            int nbc = num+5;
            std::vector<float4> hpos_u(nbc);
            std::vector<float4> hpos_s(nbc);
            std::vector<unsigned int> hindices(nbc);

            //svars.copyToHost(dens, DENS*sphp.max_num);
            //svars.copyToHost(poss, POS*sphp.max_num);

			pos_u.copyToHost(hpos_u);
			pos_s.copyToHost(hpos_s);
			indices.copyToHost(hindices);

			printf("**** INSIDE CLOUDPERMUTE ****\n");

			printf("**** UNSORTED POSITIONS *****\n");
            for (int i=0; i < num; i++)
            {
                //printf("clf_debug: %f, %f, %f, %f\n", clf[i].x, clf[i].y, clf[i].z, clf[i].w);
                printf("pos unsorted: %f, %f, %f, %f\n", hpos_u[i].x, hpos_u[i].y, hpos_u[i].z, hpos_u[i].w);
            }

			printf("**** SORTED POSITIONS *****\n");
            for (int i=0; i < num; i++)
            {
                printf("pos sorted: %f, %f, %f, %f\n", hpos_s[i].x, hpos_s[i].y, hpos_s[i].z, hpos_s[i].w);
            }

			printf("**** SORTED INDICES *****\n");
            for (int i=0; i < num; i++)
            {
                printf("indices: %d\n", hindices[i]);
            }
#endif


        //return nc;
    }
	//----------------------------------------------------------------------
}

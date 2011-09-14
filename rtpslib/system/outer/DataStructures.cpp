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


#include "OUTER.h"

#include <string>

namespace rtps
{
namespace outer
{

    DataStructures::DataStructures(CL* cli_, EB::Timer* timer_)
    {
        cli = cli_;
        timer = timer_;
        printf("create datastructures kernel\n");
        std::string path(OUTER_CL_SOURCE_DIR);
        path = path + "/datastructures.cl";
        k_datastructures = Kernel(cli, path, "datastructures");
        
    }

    int DataStructures::execute(int num,
                    //input
                    Buffer<float4>& pos_u,
                    Buffer<float4>& pos_s,
                    Buffer<float4>& vel_u,
                    Buffer<float4>& vel_s,
                    Buffer<float4>& veleval_u,
                    Buffer<float4>& veleval_s,
                    //Buffer<float4>& uvars, 
                    Buffer<float4>& color_u,
                    //Buffer<float4>& svars, 
                    Buffer<float4>& color_s,
                    //output
                    Buffer<unsigned int>& hashes,
                    Buffer<unsigned int>& indices,
                    Buffer<unsigned int>& ci_start,
                    Buffer<unsigned int>& ci_end,
                    //params
                    Buffer<OUTERParams>& sphp,
                    Buffer<GridParams>& gp,
                    int nb_cells,               //we should be able to get this from the gp buffer
                    //debug params
                    Buffer<float4>& clf_debug,
                    Buffer<int4>& cli_debug)
    {

        //-------------------
        // Set cl_cell indices to -1
        int minus = 0xffffffff;
        std::vector<unsigned int> ci_start_v(nb_cells+1);
        std::fill(ci_start_v.begin(), ci_start_v.end(), minus);
        ci_start.copyToDevice(ci_start_v);


        int iarg = 0;
        k_datastructures.setArg(iarg++, pos_u.getDevicePtr());
        k_datastructures.setArg(iarg++, pos_s.getDevicePtr());
        k_datastructures.setArg(iarg++, vel_u.getDevicePtr());
        k_datastructures.setArg(iarg++, vel_s.getDevicePtr());
        k_datastructures.setArg(iarg++, veleval_u.getDevicePtr());
        k_datastructures.setArg(iarg++, veleval_s.getDevicePtr());
        //k_datastructures.setArg(iarg++, uvars.getDevicePtr());
        k_datastructures.setArg(iarg++, color_u.getDevicePtr());
        //k_datastructures.setArg(iarg++, svars.getDevicePtr());
        k_datastructures.setArg(iarg++, color_s.getDevicePtr());
        k_datastructures.setArg(iarg++, hashes.getDevicePtr());
        k_datastructures.setArg(iarg++, indices.getDevicePtr());
        k_datastructures.setArg(iarg++, ci_start.getDevicePtr());
        k_datastructures.setArg(iarg++, ci_end.getDevicePtr());
        //k_datastructures.setArg(iarg++, cl_num_changed.getDevicePtr());
        k_datastructures.setArg(iarg++, sphp.getDevicePtr());
        k_datastructures.setArg(iarg++, gp.getDevicePtr());

        int workSize = 64;
        int nb_bytes = (workSize+1)*sizeof(int);
        k_datastructures.setArgShared(iarg++, nb_bytes);

        
        //printf("about to data structures\n");
        try
        {
            float gputime = k_datastructures.execute(num, workSize);
            if(gputime > 0)
                timer->set(gputime);

        }
        catch (cl::Error er)
        {
            printf("ERROR(data structures): %s(%s)\n", er.what(), oclErrorString(er.err()));
        }

        //ps->cli->queue.finish();

        std::vector<unsigned int> num_changed(1);
        ci_start.copyToHost(num_changed, nb_cells);
        //ci_end.copyToHost(num_changed, nb_cells);
       
        int nc = num_changed[0];
        //printf("Num Changed: %d\n", nc);

        //if(num > 0 && nc < 0) { exit(0); }
        
#if 0
        //printDataStructuresDiagnostics();

        printf("**************** DataStructures Diagnostics ****************\n");
        int nbc = nb_cells + 1;
        printf("nb_cells: %d\n", nbc);
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

            nbc = num+5;
            std::vector<float4> poss(nbc);
            std::vector<float4> dens(nbc);

            //svars.copyToHost(dens, DENS*sphp.max_num);
            svars.copyToHost(poss, POS*sphp.max_num);

            for (int i=0; i < nbc; i++)
            //for (int i=0; i < 10; i++) 
            {
                poss[i] = poss[i] / sphp.simulation_scale;
                //printf("-----\n");
                //printf("clf_debug: %f, %f, %f, %f\n", clf[i].x, clf[i].y, clf[i].z, clf[i].w);
                printf("pos sorted: %f, %f, %f, %f\n", poss[i].x, poss[i].y, poss[i].z, poss[i].w);
                //printf("dens sorted: %f, %f, %f, %f\n", dens[i].x, dens[i].y, dens[i].z, dens[i].w);
            }

#endif


        return nc;
    }

}
}

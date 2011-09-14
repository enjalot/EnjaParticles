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


#include "Hash.h"
#include <string>

namespace rtps
{

	//----------------------------------------------------------------------
    Hash::Hash(std::string path, CL* cli_, EB::Timer* timer_)
    {
        cli = cli_;
        timer = timer_;
        printf("create hash kernel\n");
        path = path + "/hash.cl";
        k_hash = Kernel(cli, path, "hash");
    }

	//----------------------------------------------------------------------
    void Hash::execute(int num,
                    //input
                    Buffer<float4>& pos_u, 
                    //output
                    Buffer<unsigned int>& hashes,
                    Buffer<unsigned int>& indices,
                    //params
                    Buffer<GridParams>& gp,
                    //debug params
                    Buffer<float4>& clf_debug,
                    Buffer<int4>& cli_debug)
    {

        int args = 0;
        //k_hash.setArg(args++, uvars.getDevicePtr()); // positions + other variables
        k_hash.setArg(args++, num);
        k_hash.setArg(args++, pos_u.getDevicePtr()); 
        k_hash.setArg(args++, hashes.getDevicePtr());
        k_hash.setArg(args++, indices.getDevicePtr());
        k_hash.setArg(args++, gp.getDevicePtr());
        /*
        printf("about to make debug buffers\n");
        Buffer<float4> clf_debug = Buffer<float4>(cli, std::vector<float4>(num));
        Buffer<int4> cli_debug = Buffer<int4>(cli, std::vector<int4>(num));
        */
        k_hash.setArg(args++, clf_debug.getDevicePtr());
        k_hash.setArg(args++, cli_debug.getDevicePtr());

		
        float gputime;
        int ctaSize = 128; // work group size
        // Hash based on unscaled data
        //printf("num in hash %d\n", num);
        gputime = k_hash.execute(num, ctaSize);

        if(gputime > 0)
            timer->set(gputime);
            //timers["hash_gpu"]->set(gputime);
        //k_hash.execute(max_num, ctaSize); //makes the out of bounds particles "stick".. or not
        // set cell_indicies_start to -1

        //ps->cli->queue.finish();

        //-------------------

        //diagnostics
#if 0
        if(num > 0)
        {
            printf("***** PRINT hash diagnostics ******\n");
            int nbc = num + 5;
            nbc = 8;
            std::vector<unsigned int> sh = hashes.copyToHost(nbc);
            std::vector<unsigned int> si = indices.copyToHost(nbc);
            //cl_cells->copyToHost();
            std::vector<int4> cli = cli_debug.copyToHost(nbc);
            std::vector<float4> clf = clf_debug.copyToHost(nbc);
            //cl_GridParams.copyToHost();

            //GridParams& gp = *cl_GridParams->getHostPtr();
            //gp.print();

            //cli_debug->copyToHost();

            //sphp.print();
            //settings.printSettings();
            for (int i=0; i < nbc; i++)
            {

                printf("cl_sort_hash[%d] %u, cl_sort_indices[%d]: %u\n", i, sh[i], i, si[i]);
                //printf("cli_debug: %d, %d, %d, %d\n", cli[i].x, cli[i].y, cli[i].z, cli[i].w);
                //printf("clf_debug: %f, %f, %f, %f\n", clf[i].x, clf[i].y, clf[i].z, clf[i].w);
                //printf("-----\n");

#if 0
                int gx = (cl_cells[i].x - gp.grid_min.x) * gp.grid_inv_delta.x ;
                int gy = (cl_cells[i].y - gp.grid_min.y) * gp.grid_inv_delta.y ;
                int gz = (cl_cells[i].z - gp.grid_min.z) * gp.grid_inv_delta.z ;
                //printf("cl_cells,cl_cells,cl_cells= %f, %f, %f\n", cl_cells[i].x, cl_cells[i].y, cl_cells[i].z);
                //gp.grid_min.print("grid min");
                //printf("gx,gy,gz= %d, %d, %d\n", gx, gy, gz);
                unsigned int idx = (gz*gp.grid_res.y + gy) * gp.grid_res.x + gx; 
                if (idx != cl_sort_hashes[i])
                {
                    printf("hash indices (exact vs GPU do not match)\n");
                }
                printf("cli_debug: %d, %d, %d\n", cli_debug[i].x, cli_debug[i].y, cli_debug[i].z);
                //printf("---------------------------\n");
#endif
            }
        }
#endif
    }

    //----------------------------------------------------------------------

}

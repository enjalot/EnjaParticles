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
using namespace std;

namespace rtps
{
namespace outer
{
#if 0 
    //----------------------------------------------------------------------

    //----------------------------------------------------------------------
    void OUTER::loadNeighbors()
    {
        printf("enter neighbor\n");

        try
        {
            string path(OUTER_CL_SOURCE_DIR);
            path = path + "/neighbors.cl";
            k_neighbors = Kernel(ps->cli, path, "neighbors");
            printf("bigger problem\n");
        }
        catch (cl::Error er)
        {
            printf("ERROR(neighborSearch): %s(%s)\n", er.what(), oclErrorString(er.err()));
            exit(1);
        }


    }
    //----------------------------------------------------------------------

    void OUTER::neighborSearch(int choice)
    {
        int iarg = 0;
        k_neighbors.setArg(iarg++, cl_vars_sorted.getDevicePtr());
        k_neighbors.setArg(iarg++, cl_cell_indices_start.getDevicePtr());
        k_neighbors.setArg(iarg++, cl_cell_indices_end.getDevicePtr());
        k_neighbors.setArg(iarg++, cl_GridParamsScaled.getDevicePtr());
        //k_neighbors.setArg(iarg++, cl_FluidParams->getDevicePtr());
        k_neighbors.setArg(iarg++, cl_sphp.getDevicePtr());

        // ONLY IF DEBUGGING
        k_neighbors.setArg(iarg++, clf_debug.getDevicePtr());
        k_neighbors.setArg(iarg++, cli_debug.getDevicePtr());

        // which == 0 : density update
        // which == 1 : force update

        /*
        if (which == 0) ts_cl[TI_DENS]->start();
        if (which == 1) ts_cl[TI_PRES]->start();
        if (which == 2) ts_cl[TI_COL]->start();
        if (which == 3) ts_cl[TI_COL_NORM]->start();
        */

        //Copy choice to OUTERParams
        settings->SetSetting("Choice", choice);
        //sphp.choice = choice;
        updateOUTERP();
        /*
        std::vector<OUTERParams> vsphp(0);
        vsphp.push_back(sphp);
        cl_OUTERParams.copyToDevice(vsphp);
        */

#if 0
        std::vector<int4> cli = cli_debug.copyToHost(2);
        for (int i=0; i < 2; i++)
        {
            printf("cli_debug: %d\n", cli[i].w);
        }
#endif

        int local = 64;
        try
        {
            k_neighbors.execute(num, local);
        }

        catch (cl::Error er)
        {
            printf("ERROR(neighbor %d): %s(%s)\n", choice, er.what(), oclErrorString(er.err()));
        }
        ps->cli->queue.finish();

#if 0 //printouts    
        //DEBUGING
        
        if(num > 0)// && choice == 0)
        {
            printf("============================================\n");
            printf("which == %d *** \n", choice);
            printf("***** PRINT neighbors diagnostics ******\n");
            printf("num %d\n", num);

            std::vector<int4> cli(num);
            std::vector<float4> clf(num);
            
            cli_debug.copyToHost(cli);
            clf_debug.copyToHost(clf);

            std::vector<float4> poss(num);
            std::vector<float4> dens(num);

            for (int i=0; i < num; i++)
            //for (int i=0; i < 10; i++) 
            {
                //printf("-----\n");
                printf("clf_debug: %f, %f, %f, %f\n", clf[i].x, clf[i].y, clf[i].z, clf[i].w);
                //if(clf[i].w == 0.0) exit(0);
                //printf("cli_debug: %d, %d, %d, %d\n", cli[i].x, cli[i].y, cli[i].z, cli[i].w);
                //		printf("pos : %f, %f, %f, %f\n", pos[i].x, pos[i].y, pos[i].z, pos[i].w);
            }
        }
#endif

    }
#endif

} // namespace
} // namespace

#include "SPH.h"

#include <string>

namespace rtps
{

    DataStructures::DataStructures(CL* cli_, EB::Timer* timer_)
    {
        cli = cli_;
        timer = timer_;
        printf("create datastructures kernel\n");
        std::string path(SPH_CL_SOURCE_DIR);
        path = path + "/datastructures.cl";
        k_datastructures = Kernel(cli, path, "datastructures");
        
    }

    int DataStructures::execute(int num,
                    //input
                    Buffer<float4>& uvars, 
                    Buffer<float4>& color_u,
                    Buffer<float4>& svars, 
                    Buffer<float4>& color_s,
                    //output
                    Buffer<unsigned int>& hashes,
                    Buffer<unsigned int>& indices,
                    Buffer<unsigned int>& ci_start,
                    Buffer<unsigned int>& ci_end,
                    //params
                    Buffer<SPHParams>& sphp,
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
        k_datastructures.setArg(iarg++, uvars.getDevicePtr());
        k_datastructures.setArg(iarg++, color_u.getDevicePtr());
        k_datastructures.setArg(iarg++, svars.getDevicePtr());
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
            k_datastructures.execute(num, workSize);
        }
        catch (cl::Error er)
        {
            printf("ERROR(data structures): %s(%s)\n", er.what(), oclErrorString(er.err()));
        }

        //ps->cli->queue.finish();

        std::vector<unsigned int> num_changed(1);
        ci_start.copyToHost(num_changed, nb_cells);
       
        int nc = num_changed[0];
        return nc;
        //printf("Num Changed: %d\n", nc);

        //if(num > 0 && nc < 0) { exit(0); }
        
#if 0 
//this has been moved to SPH.cpp
        if (nc < num && nc > 0)
        //if(num > 0)
        {
            num = nc;
            settings->SetSetting("Number of Particles", num);
            //sphp.num = num;
            updateSPHP();
            renderer->setNum(sphp.num);
            //need to copy sorted positions into unsorted + position array
            
            prep(2);
            //hash();
            hash.execute(   num,
                    uvars,
                    hashes,
                    indices,
                    cl_sphp,
                    cl_GridParams,
                    clf_debug,
                    cli_debug);

            bitonic_sort();
            
        }
#endif

#if 0
        //printDataStructuresDiagnostics();

        printf("**************** DataStructures Diagnostics ****************\n");
        int nbc = grid_params.nb_cells + 1;
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


    }

}

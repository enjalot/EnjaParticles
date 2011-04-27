#include "CellIndices.h"

#include <string>

namespace rtps
{

    CellIndices::CellIndices(std::string path, CL* cli_, EB::Timer* timer_)
    {
        cli = cli_;
        timer = timer_;
        printf("create cellindices kernel\n");
        path = path + "/cellindices.cl";
        k_cellindices = Kernel(cli, path, "cellindices");
        
    }

    int CellIndices::execute(int num,
                    Buffer<unsigned int>& hashes,
                    Buffer<unsigned int>& indices,
                    Buffer<unsigned int>& ci_start,
                    Buffer<unsigned int>& ci_end,
                    //params
                    //Buffer<SPHParams>& sphp,
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
        k_cellindices.setArg(iarg++, num);
        k_cellindices.setArg(iarg++, hashes.getDevicePtr());
        k_cellindices.setArg(iarg++, indices.getDevicePtr());
        k_cellindices.setArg(iarg++, ci_start.getDevicePtr());
        k_cellindices.setArg(iarg++, ci_end.getDevicePtr());
        //k_cellindices.setArg(iarg++, cl_num_changed.getDevicePtr());
        //k_cellindices.setArg(iarg++, sphp.getDevicePtr());
        k_cellindices.setArg(iarg++, gp.getDevicePtr());

        int workSize = 64;
        int nb_bytes = (workSize+1)*sizeof(int);
        k_cellindices.setArgShared(iarg++, nb_bytes);

        
        //printf("about to data structures\n");
        try
        {
            float gputime = k_cellindices.execute(num, workSize);
            if(gputime > 0)
                timer->set(gputime);

        }
        catch (cl::Error er)
        {
            printf("ERROR(cellindices): %s(%s)\n", er.what(), oclErrorString(er.err()));
        }

        //ps->cli->queue.finish();

        std::vector<unsigned int> num_changed(1);
        ci_start.copyToHost(num_changed, nb_cells);
        //ci_end.copyToHost(num_changed, nb_cells);
       
        int nc = num_changed[0];
        //printf("Num Changed: %d\n", nc);

        //if(num > 0 && nc < 0) { exit(0); }
        
#if 0
        //printCellIndicesDiagnostics();

        printf("**************** CellIndices Diagnostics ****************\n");
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
                if(is[i] < 8000 || ie[i] > 0)
                {
                    printf("cell: %d indices start: %d indices stop: %d\n", i, is[i], ie[i]);
                }
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

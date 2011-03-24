#include "SPH.h"

#include <string>

namespace rtps
{

    void SPH::loadDataStructures()
    {
        printf("create datastructures kernel\n");
        std::string path(SPH_CL_SOURCE_DIR);
        path = path + "/datastructures.cl";
        //std::string filepath = path + "/datastructures.cl";
        //k_datastructures = Kernel(ps->cli, path, filepath, "datastructures");
        k_datastructures = Kernel(ps->cli, path, "datastructures");
        
    }

    void SPH::testCut()
    {
        num = 10;
        sphp.num = num;
        updateSPHP();
        renderer->setNum(sphp.num);
        cut = true;

    }

    void SPH::buildDataStructures()
    // Generate hash list: stored in cl_sort_hashes
    {

        int iarg = 0;
        k_datastructures.setArg(iarg++, cl_vars_unsorted.getDevicePtr());
        k_datastructures.setArg(iarg++, cl_vars_sorted.getDevicePtr());
        k_datastructures.setArg(iarg++, cl_sort_hashes.getDevicePtr());
        k_datastructures.setArg(iarg++, cl_sort_indices.getDevicePtr());
        k_datastructures.setArg(iarg++, cl_cell_indices_start.getDevicePtr());
        k_datastructures.setArg(iarg++, cl_cell_indices_end.getDevicePtr());
        //k_datastructures.setArg(iarg++, cl_num_changed.getDevicePtr());
        k_datastructures.setArg(iarg++, cl_SPHParams.getDevicePtr());
        k_datastructures.setArg(iarg++, cl_GridParamsScaled.getDevicePtr());

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

        ps->cli->queue.finish();

        std::vector<unsigned int> num_changed(1);
        cl_cell_indices_start.copyToHost(num_changed, grid_params.nb_cells);
       
        int nc = num_changed[0];
        printf("Num Changed: %d\n", nc);

        //if(num > 0 && nc < 0) { exit(0); }
        
        if (nc < num && nc > 0)
        //if(num > 0)
        {
            //sphp.num = nc-1;
            //num = 10;
            //seems like hashes are getting messed up
            //probably because we are hashing on unsorted particles, cutting off only works on sorted
            //num = 10;
            num = nc;
            sphp.num = num;
            updateSPHP();
            renderer->setNum(sphp.num);
            //need to copy sorted positions into unsorted + position array
            prep(2);
        }

        printDataStructuresDiagnostics();

    }

    void SPH::printDataStructuresDiagnostics()
    {
        printf("**************** DataStructures Diagnostics ****************\n");
        int nbc = grid_params.nb_cells + 1;
        printf("nb_cells: %d\n", nbc);
        printf("num particles: %d\n", num);

        std::vector<unsigned int> is(nbc);
        std::vector<unsigned int> ie(nbc);
        
        cl_cell_indices_end.copyToHost(ie);
        cl_cell_indices_start.copyToHost(is);


        for(int i = 0; i < nbc; i++)
        {
            if (is[i] != -1)// && ie[i] != 0)
            {
                //nb = ie[i] - is[i];
                //nb_particles += nb;
                printf("cell: %d indices start: %d indices stop: %d\n", i, is[i], ie[i]);
            }
        }


#if 0
        //print out elements from the sorted arrays
#define DENS 0
#define POS 1
#define VEL 2

            nbc = num+5;
            std::vector<float4> poss(nbc);
            std::vector<float4> dens(nbc);

            //cl_vars_sorted.copyToHost(dens, DENS*sphp.max_num);
            cl_vars_sorted.copyToHost(poss, POS*sphp.max_num);

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

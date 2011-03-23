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

        /*
        std::vector<unsigned int> num_changed(20);
        std::fill(num_changed.begin(), num_changed.end(), 0);
        num_changed[0] = 0;
        cl_num_changed = Buffer<unsigned int>(ps->cli, num_changed);
        */

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

    }

    void SPH::buildDataStructures()
    // Generate hash list: stored in cl_sort_hashes
    {


        /*
        int nbc = 20;
        std::vector<int> sh = cl_sort_hashes.copyToHost(nbc);
        //std::vector<int> eci = cl_cell_indices_end.copyToHost(nbc);
    
        for(int i = 0; i < nbc; i++)
        {
            printf("sh[%d] %d; ", i, sh[i]);
        }
        printf("\n");
        */

        /*
        //std::vector<unsigned int> num_changed(1);
        std::vector<unsigned int> num_changed(20);
        std::fill(num_changed.begin(), num_changed.end(), 0);

        num_changed[0] = 0;
        cl_num_changed.copyToDevice(num_changed);
        */

        //printf("about to data structures\n");
        int workSize = 64; // work group size
        try
        {
            k_datastructures.execute(num, workSize);
        }
        catch (cl::Error er)
        {
            printf("ERROR(data structures): %s(%s)\n", er.what(), oclErrorString(er.err()));
        }

        ps->cli->queue.finish();

        /*
        cl_num_changed.copyToHost(num_changed);
        for(int i = 0; i < num; i++)
        {
            printf("nc[%d] = %d\n", i, num_changed[i]);
        }
        int nc = num_changed[0];
        printf("Num Changed: %d\n", nc);
        
        if(num != 0 && num != 20)
        { 
            printf("QUITER!\n");
            exit(0);}

        if (nc < num && nc > 0)
        //if(num > 0)
        {
            //sphp.num = nc-1;
            //num = 10;
            //seems like hashes are getting messed up
            //num = nc-1;
            sphp.num = num;
            updateSPHP();
            renderer->setNum(sphp.num);
        }
        */

        printDataStructuresDiagnostics();

#if 0
    //printouts
    int nbc = 10;
    //printf("start cell indices\n");
    //printf("end cell indices\n");
    //nbc = grid_params.nb_cells;
    //std::vector<int> is = cl_cell_indices_start.copyToHost(nbc);
    //std::vector<int> ie = cl_cell_indices_end.copyToHost(nbc);

    std::vector<int> is(nbc);
    std::vector<int> ie(nbc);
    
    cl_cell_indices_end.copyToHost(is);
    cl_cell_indices_start.copyToHost(ie, 100);


    /*
    for(int i = 0; i < nbc; i++)
    {
        printf("sci[%d] %d eci[%d] %d\n", i, is[i], i, ie[i]);
    }
    */

    int nb_particles = 0;
    int nb;
    int asdf = 0;
    for (int i=0; i < nbc; i++) {
    //for (int i=0; i < 100; i++) {
        printf("is,ie[%d]= %d, %d\n", i, is[i], ie[i]);
        // ie[i] SHOULD NEVER BE ZERO 
        //printf("is[%d] %d ie[%d] %d\n", i, is[i], i, ie[i]);
        if (is[i] != -1 && ie[i] != 0) {
            nb = ie[i] - is[i];
            nb_particles += nb;
        }
        */

        int nb_particles = 0;
        int nb;
        int asdf = 0;
        for (int i=0; i < grid_params.nb_cells; i++)
        {
            //for (int i=0; i < 100; i++) {
            //printf("is,ie[%d]= %d, %d\n", i, is[i], ie[i]);
            // ie[i] SHOULD NEVER BE ZERO 
            //printf("is[%d] %d ie[%d] %d\n", i, is[i], i, ie[i]);
            if (is[i] != -1 && ie[i] != 0)
            {
                nb = ie[i] - is[i];
                nb_particles += nb;
            }
            if (is[i] != -1 && ie[i] != 0 && i > 600 && i < 1000)
            {
                asdf++;
                //printf("(GPU) [%d]: indices_start: %d, indices_end: %d, nb pts: %d\n", i, is[i], ie[i], nb);
            }
        }
        printf("asdf: %d\n", asdf);
        printf("done with data structures\n");
#endif


    }

    void SPH::printDataStructuresDiagnostics()
    {
        printf("**************** DataStructures Diagnostics ****************\n");
        int nbc = grid_params.nb_cells;
        printf("nb_cells: %d\n", nbc);

        std::vector<unsigned int> is(nbc);
        std::vector<unsigned int> ie(nbc);
        
        cl_cell_indices_end.copyToHost(ie);
        cl_cell_indices_start.copyToHost(is);


        for(int i = 0; i < nbc+1; i++)
        {
            if (is[i] != -1)// && ie[i] != 0)
            {
                //nb = ie[i] - is[i];
                //nb_particles += nb;
                printf("cell: %d indices start: %d indices stop: %d\n", i, is[i], ie[i]);
            }
        }

    }

}

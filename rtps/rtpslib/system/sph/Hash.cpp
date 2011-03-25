#include "SPH.h"

#include <string>

namespace rtps
{

    void SPH::loadHash()
    {
        printf("create hash kernel\n");
        std::string path(SPH_CL_SOURCE_DIR);
        path = path + "/hash.cl";
        k_hash = Kernel(ps->cli, path, "hash");
    }

    void SPH::hash()
    // Generate hash list: stored in cl_sort_hashes
    {

        int args = 0;
        k_hash.setArg(args++, cl_vars_unsorted.getDevicePtr()); // positions + other variables
        k_hash.setArg(args++, cl_sort_hashes.getDevicePtr());
        k_hash.setArg(args++, cl_sort_indices.getDevicePtr());
        k_hash.setArg(args++, cl_SPHParams.getDevicePtr());
        k_hash.setArg(args++, cl_GridParams.getDevicePtr());
        k_hash.setArg(args++, clf_debug.getDevicePtr());
        k_hash.setArg(args++, cli_debug.getDevicePtr());


        int ctaSize = 128; // work group size
        // Hash based on unscaled data
        //printf("num in hash %d\n", num);
        k_hash.execute(num, ctaSize);
        //k_hash.execute(max_num, ctaSize); //makes the out of bounds particles "stick".. or not
        // set cell_indicies_start to -1
        int minus = 0xffffffff;

        ps->cli->queue.finish();

        //-------------------
        // Set cl_cell indices to -1
        std::vector<unsigned int> cells_indices_start(grid_params.nb_cells+1);
        std::fill(cells_indices_start.begin(), cells_indices_start.end(), minus);
        cl_cell_indices_start.copyToDevice(cells_indices_start);

        //-------------------


        if(num > 0)
        {
            //printHashDiagnostics();
        }

    }

    //----------------------------------------------------------------------

    void SPH::printHashDiagnostics()
    {
#if 1
        printf("***** PRINT hash diagnostics ******\n");
        int nbc = num + 5;
        std::vector<unsigned int> sh = cl_sort_hashes.copyToHost(nbc);
        std::vector<unsigned int> si = cl_sort_indices.copyToHost(nbc);
        //cl_cells->copyToHost();
        std::vector<int4> cli = cli_debug.copyToHost(nbc);
        std::vector<float4> clf = clf_debug.copyToHost(nbc);
        //cl_GridParams.copyToHost();

        //GridParams& gp = *cl_GridParams->getHostPtr();
        //gp.print();

        //cli_debug->copyToHost();

        //for (int i=0; i < num; i++) {  
        for (int i=0; i < nbc; i++)
        {
            printf("cl_sort_hash[%d] %u, cl_sort_indices[%d]: %u\n", i, sh[i], i, si[i]);
            printf("cli_debug: %d, %d, %d, %d\n", cli[i].x, cli[i].y, cli[i].z, cli[i].w);
            printf("clf_debug: %f, %f, %f, %f\n", clf[i].x, clf[i].y, clf[i].z, clf[i].w);
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
#endif
    }

    //----------------------------------------------------------------------

}

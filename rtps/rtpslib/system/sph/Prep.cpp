#include "SPH.h"

#include <string>

namespace rtps
{

    void SPH::loadPrep()
    {
        printf("create prep kernel\n");
        std::string path(SPH_CL_SOURCE_DIR);
        path = path + "/prep.cl";
        k_prep = Kernel(ps->cli, path, "prep");

    }

    void SPH::prep(int stage)
    {
        /**
         * sometimes we only want to copy positions
         * this should probably be replaced with Scopy
         * i don't think straight copy is most efficient...
         */

        printf("num: %d, stage: %d\n", num, stage);
        int args = 0;
        k_prep.setArg(args++, num);
        k_prep.setArg(args++, stage);
        k_prep.setArg(args++, cl_position.getDevicePtr());
        k_prep.setArg(args++, cl_velocity.getDevicePtr());
        k_prep.setArg(args++, cl_vars_unsorted.getDevicePtr());
        k_prep.setArg(args++, cl_vars_sorted.getDevicePtr()); 
        k_prep.setArg(args++, cl_sort_indices.getDevicePtr());


        int ctaSize = 128; // work group size
        // Hash based on unscaled data
        try
        {
            k_prep.execute(num, ctaSize);
        }
        catch (cl::Error er)
        {
            printf("ERROR(prep): %s(%s)\n", er.what(), oclErrorString(er.err()));
            exit(1);
        }

        ps->cli->queue.finish();
    }

}

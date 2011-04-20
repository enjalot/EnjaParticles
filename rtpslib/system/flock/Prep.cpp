#include "FLOCK.h"

#include <string>

namespace rtps 
{
namespace flock 
{

    Prep::Prep(CL* cli_, EB::Timer* timer_)
    {
        cli = cli_;
        timer = timer_;
        printf("create prep kernel\n");
        std::string path(FLOCK_CL_SOURCE_DIR);
        path = path + "/prep.cl";
        k_prep = Kernel(cli, path, "prep");
    }

    void Prep::execute(int num,
                    int stage,
                    Buffer<float4>& pos_u,
                    Buffer<float4>& pos_s,
                    Buffer<float4>& vel_u,
                    Buffer<float4>& vel_s,
                    Buffer<float4>& veleval_u,
                    Buffer<float4>& veleval_s,
                    Buffer<float4>& color_u,
                    Buffer<float4>& color_s,
                    //Buffer<float4>& uvars, 
                    //Buffer<float4>& svars, 
                    Buffer<unsigned int>& indices,
                    //params
                    Buffer<FLOCKParameters>& flockp,
                    //Buffer<GridParams>& gp,
                    //debug params
                    Buffer<float4>& clf_debug,
                    Buffer<int4>& cli_debug)
    {
        /**
         * sometimes we only want to copy positions
         * this should probably be replaced with Scopy
         * i don't think straight copy is most efficient...
         */

        printf("num: %d, stage: %d\n", num, stage);
        int args = 0;
        k_prep.setArg(args++, stage);
        k_prep.setArg(args++, pos_u.getDevicePtr());
        k_prep.setArg(args++, pos_s.getDevicePtr());
        k_prep.setArg(args++, vel_u.getDevicePtr());
        k_prep.setArg(args++, vel_s.getDevicePtr());
        k_prep.setArg(args++, veleval_u.getDevicePtr());
        k_prep.setArg(args++, veleval_s.getDevicePtr());
        //k_prep.setArg(args++, uvars.getDevicePtr());
        //k_prep.setArg(args++, svars.getDevicePtr()); 
        k_prep.setArg(args++, color_u.getDevicePtr());
        k_prep.setArg(args++, color_s.getDevicePtr());
        k_prep.setArg(args++, indices.getDevicePtr());
        k_prep.setArg(args++, flockp.getDevicePtr());


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

    }

} // namespace flock

}

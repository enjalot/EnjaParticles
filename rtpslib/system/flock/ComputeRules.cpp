#include <FLOCK.h>
#include<math.h>

namespace rtps 
{
    //----------------------------------------------------------------------
    ComputeRules::ComputeRules(CL* cli_, EB::Timer* timer_)
    {
        cli = cli_;
        timer = timer_;
     
        printf("load computeRules\n");

        try
        {
            string path(FLOCK_CL_SOURCE_DIR);
            path = path + "/computeRules.cl";
            k_computeRules= Kernel(cli, path, "computeRules_update");
        }
        catch (cl::Error er)
        {
            printf("ERROR(computeRules): %s(%s)\n", er.what(), oclErrorString(er.err()));
        }


    }
    //----------------------------------------------------------------------

    void ComputeRules::execute(int num,
                    //input
                    Buffer<float4>& pos_s,
                    Buffer<float4>& sep_s,
                    //output
                    Buffer<unsigned int>& ci_start,
                    Buffer<unsigned int>& ci_end,
                    //params
                    Buffer<FLOCKParameters>& flockp,
                    Buffer<GridParams>& gp,
                    //debug params
                    Buffer<float4>& clf_debug,
                    Buffer<int4>& cli_debug)
    { 
        int iarg = 0;
        k_computeRules.setArg(iarg++, pos_s.getDevicePtr());
        k_computeRules.setArg(iarg++, sep_s.getDevicePtr());
        k_computeRules.setArg(iarg++, ci_start.getDevicePtr());
        k_computeRules.setArg(iarg++, ci_end.getDevicePtr());
        k_computeRules.setArg(iarg++, gp.getDevicePtr());
        k_computeRules.setArg(iarg++, flockp.getDevicePtr());

        // ONLY IF DEBUGGING
        k_computeRules.setArg(iarg++, clf_debug.getDevicePtr());
        k_computeRules.setArg(iarg++, cli_debug.getDevicePtr());

        int local = 64;
        try
        {
            float gputime = k_computeRules.execute(num, local);
            if(gputime > 0)
                timer->set(gputime);

        }

        catch (cl::Error er)
        {
            printf("ERROR(computeRules): %s(%s)\n", er.what(), oclErrorString(er.err()));
        }
    }

    void FLOCK::cpuComputeRules()
    {

    }

} 

#include <FLOCK.h>
#include<math.h>

namespace rtps
{
    //----------------------------------------------------------------------
    Rules::Rules(CL* cli_, EB::Timer* timer_)
    {
        cli = cli_;
        timer = timer_;
     
        printf("load rules\n");

        try
        {
            string path(FLOCK_CL_SOURCE_DIR);
            path = path + "/rules.cl";
            k_rules= Kernel(cli, path, "rules_update");
        }
        catch (cl::Error er)
        {
            printf("ERROR(rules): %s(%s)\n", er.what(), oclErrorString(er.err()));
        }


    }
    //----------------------------------------------------------------------

    void Rules::execute(int num,
                    //input
                    Buffer<float4>& pos_s,
                    Buffer<float4>& sep_s,
                    //output
                    Buffer<unsigned int>& ci_start,
                    Buffer<unsigned int>& ci_end,
                    //params
                    Buffer<FLOCKParams>& flockp,
                    Buffer<GridParams>& gp,
                    //debug params
                    Buffer<float4>& clf_debug,
                    Buffer<int4>& cli_debug)
    { 
        int iarg = 0;
        k_rules.setArg(iarg++, pos_s.getDevicePtr());
        k_rules.setArg(iarg++, sep_s.getDevicePtr());
        k_rules.setArg(iarg++, ci_start.getDevicePtr());
        k_rules.setArg(iarg++, ci_end.getDevicePtr());
        k_rules.setArg(iarg++, gp.getDevicePtr());
        k_rules.setArg(iarg++, flockp.getDevicePtr());

        // ONLY IF DEBUGGING
        k_rules.setArg(iarg++, clf_debug.getDevicePtr());
        k_rules.setArg(iarg++, cli_debug.getDevicePtr());

        int local = 64;
        try
        {
            float gputime = k_rules.execute(num, local);
            if(gputime > 0)
                timer->set(gputime);

        }

        catch (cl::Error er)
        {
            printf("ERROR(rules): %s(%s)\n", er.what(), oclErrorString(er.err()));
        }
    }

    void FLOCK::cpuRules()
    {

    }

} 

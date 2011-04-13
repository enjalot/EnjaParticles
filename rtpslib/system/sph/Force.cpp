#include <SPH.h>

namespace rtps 
{

    //----------------------------------------------------------------------
    Force::Force(CL* cli_, EB::Timer* timer_)
    {
        cli = cli_;
        timer = timer_;
     
        printf("load force\n");

        try
        {
            string path(SPH_CL_SOURCE_DIR);
            path = path + "/force.cl";
            k_force = Kernel(cli, path, "force_update");
        }
        catch (cl::Error er)
        {
            printf("ERROR(Force): %s(%s)\n", er.what(), oclErrorString(er.err()));
        }


    }
    //----------------------------------------------------------------------

    void Force::execute(int num,
                    Buffer<float4>& pos_s,
                    Buffer<float>& dens_s,
                    Buffer<float4>& veleval_s,
                    Buffer<float4>& force_s,
                    Buffer<float4>& xsph_s,
                    Buffer<unsigned int>& ci_start,
                    Buffer<unsigned int>& ci_end,
                    //params
                    Buffer<SPHParams>& sphp,
                    Buffer<GridParams>& gp,
                    //debug params
                    Buffer<float4>& clf_debug,
                    Buffer<int4>& cli_debug)
    { 
        int iarg = 0;
        k_force.setArg(iarg++, pos_s.getDevicePtr());
        k_force.setArg(iarg++, dens_s.getDevicePtr());
        k_force.setArg(iarg++, veleval_s.getDevicePtr());
        k_force.setArg(iarg++, force_s.getDevicePtr());
        k_force.setArg(iarg++, xsph_s.getDevicePtr());
        k_force.setArg(iarg++, ci_start.getDevicePtr());
        k_force.setArg(iarg++, ci_end.getDevicePtr());
        k_force.setArg(iarg++, gp.getDevicePtr());
        k_force.setArg(iarg++, sphp.getDevicePtr());

        // ONLY IF DEBUGGING
        k_force.setArg(iarg++, clf_debug.getDevicePtr());
        k_force.setArg(iarg++, cli_debug.getDevicePtr());

        int local = 64;
        try
        {
            float gputime = k_force.execute(num, local);
            if(gputime > 0)
                timer->set(gputime);

        }

        catch (cl::Error er)
        {
            printf("ERROR(force ): %s(%s)\n", er.what(), oclErrorString(er.err()));
        }

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


}


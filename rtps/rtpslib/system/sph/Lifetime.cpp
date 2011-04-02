#include "../SPH.h"

namespace rtps
{

    Lifetime::Lifetime(CL* cli_, EB::Timer* timer_)
    {
        cli = cli_;
        timer = timer_;
 
        printf("create liftime kernel\n");
        std::string path(SPH_CL_SOURCE_DIR);
        path += "/lifetime.cl";
        k_lifetime = Kernel(cli, path, "lifetime");

    } 
    void Lifetime::execute(int num,
                    float increment,
                    Buffer<float4>& pos,
                    Buffer<float4>& color_u, 
                    Buffer<float4>& color_s, 
                    Buffer<unsigned int>& indices
                    //Buffer<float4>& clf_debug,
                    //Buffer<int4>& cli_debug)
        )
    {

        int iargs = 0;
        k_lifetime.setArg(iargs++, pos.getDevicePtr());
        k_lifetime.setArg(iargs++, color_u.getDevicePtr());
        k_lifetime.setArg(iargs++, color_s.getDevicePtr());
        k_lifetime.setArg(iargs++, indices.getDevicePtr());
        //lifetime.setArg(iargs++, color.getDevicePtr());
        k_lifetime.setArg(iargs++, increment); //time step

        int local_size = 128;
        float gputime = k_lifetime.execute(num, local_size);
        if(gputime > 0)
            timer->set(gputime);



#if 0
#define DENS 0
#define POS 1
#define VEL 2

        printf("************ LeapFrog **************\n");
            int nbc = num+5;
            std::vector<float4> poss(nbc);
            std::vector<float4> uposs(nbc);
            std::vector<float4> dens(nbc);

            //cl_vars_sorted.copyToHost(dens, DENS*sphp.max_num);
            cl_vars_sorted.copyToHost(poss, POS*sphp.max_num);
            cl_vars_unsorted.copyToHost(uposs, POS*sphp.max_num);

            for (int i=0; i < nbc; i++)
            //for (int i=0; i < 10; i++) 
            {
                poss[i] = poss[i] / sphp.simulation_scale;
                //printf("-----\n");
                //printf("clf_debug: %f, %f, %f, %f\n", clf[i].x, clf[i].y, clf[i].z, clf[i].w);
                printf("pos sorted: %f, %f, %f, %f\n", poss[i].x, poss[i].y, poss[i].z, poss[i].w);
                printf("pos unsorted: %f, %f, %f, %f\n", uposs[i].x, uposs[i].y, uposs[i].z, uposs[i].w);
                //printf("dens sorted: %f, %f, %f, %f\n", dens[i].x, dens[i].y, dens[i].z, dens[i].w);
            }

#endif

        /*
         * enables us to cut off after a couple iterations
         * by setting cut = 1 from some other function
        if(cut >= 1)
        {
            if (cut == 2) {exit(0);}
            cut++;
        }
        */


    }


}

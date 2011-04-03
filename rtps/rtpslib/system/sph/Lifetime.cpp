#include "../SPH.h"

namespace rtps
{

    Lifetime::Lifetime(CL* cli_, EB::Timer* timer_, std::string filename)
    {
        cli = cli_;
        timer = timer_;
 
        printf("create liftime kernel\n");
        std::string path(SPH_CL_SOURCE_DIR);
        path += "/" + filename;
        k_lifetime = Kernel(cli, path, "lifetime");

    } 
    void Lifetime::execute(int num,
                    float increment,
                    Buffer<float4>& pos,
                    Buffer<float4>& color_u, 
                    Buffer<float4>& color_s, 
                    Buffer<unsigned int>& indices,
                    Buffer<float4>& clf_debug,
                    Buffer<int4>& cli_debug)
    {

        int iargs = 0;
        k_lifetime.setArg(iargs++, num); //time step
        k_lifetime.setArg(iargs++, increment); //time step
        k_lifetime.setArg(iargs++, pos.getDevicePtr());
        k_lifetime.setArg(iargs++, color_u.getDevicePtr());
        k_lifetime.setArg(iargs++, color_s.getDevicePtr());
        k_lifetime.setArg(iargs++, indices.getDevicePtr());
        //lifetime.setArg(iargs++, color.getDevicePtr());
        k_lifetime.setArg(iargs++, clf_debug.getDevicePtr());
        k_lifetime.setArg(iargs++, cli_debug.getDevicePtr());



        int local_size = 128;
        float gputime = k_lifetime.execute(num, local_size);
        if(gputime > 0)
            timer->set(gputime);



#if 0

        if(num > 0)
        {
            printf("************ Lifetime**************\n");
            int nbc = num + 5;
            std::vector<int4> cli(nbc);
            std::vector<float4> clf(nbc);

            cli_debug.copyToHost(cli);
            clf_debug.copyToHost(clf);

            for (int i=0; i < nbc; i++)
            {
                //printf("-----\n");
                printf("clf_debug: %f, %f, %f, %f\n", clf[i].x, clf[i].y, clf[i].z, clf[i].w);
                //printf("cli_debug: %d, %d, %d, %d\n", cli[i].x, cli[i].y, cli[i].z, cli[i].w);
            }
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

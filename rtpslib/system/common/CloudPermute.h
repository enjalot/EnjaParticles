#ifndef RTPS_CLOUDPERMUTE_H_INCLUDED
#define RTPS_CLOUDPERMUTE_H_INCLUDED


#include <RTPS.h>
//#include <CLL.h>
#include <Buffer.h>


namespace rtps
{
    class CloudPermute 
    {
        public:
            CloudPermute() { cli = NULL; timer = NULL; };
            CloudPermute(std::string path, CL* cli, EB::Timer* timer);
            void execute(int num, // nb cloud points
                    //input
                    Buffer<float4>& pos_u,
                    Buffer<float4>& pos_s,
                    Buffer<float4>& normal_u,
                    Buffer<float4>& normal_s,
                    //Buffer<float4>& vel_u,
                    //Buffer<float4>& vel_s,
                    //Buffer<float4>& veleval_u,
                    //Buffer<float4>& veleval_s,
                    //Buffer<float4>& color_u,
                    //Buffer<float4>& color_s,
                    Buffer<unsigned int>& indices,
                    //params
                    //Buffer<SPHParams>& sphp,
                    Buffer<GridParams>& gp,
                    //debug params
                    Buffer<float4>& clf_debug,
                    Buffer<int4>& cli_debug);

        private:
            CL* cli;
            Kernel k_permute;
            EB::Timer* timer;
    };
}

#endif

#ifndef RTPS_PREP_H_INCLUDED
#define RTPS_PREP_H_INCLUDED


#include <CLL.h>
#include <Buffer.h>


namespace rtps
{
    class Prep 
    {
        public:
            Prep() { cli = NULL; timer = NULL; };
            Prep(CL* cli, EB::Timer* timer);
            void execute(int num,
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
                    Buffer<SPHParams>& sphp,
                    //Buffer<GridParams>& gp,
                    //debug params
                    Buffer<float4>& clf_debug,
                    Buffer<int4>& cli_debug);

        private:
            CL* cli;
            Kernel k_prep;
            EB::Timer* timer;
    };
}

#endif

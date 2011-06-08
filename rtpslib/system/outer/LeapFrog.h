#ifndef RTPS_OUTER_LEAPFROG_H_INCLUDED
#define RTPS_OUTER_LEAPFROG_H_INCLUDED


#include <CLL.h>
#include <Buffer.h>


namespace rtps
{
namespace outer
{

    class LeapFrog
    {
        public:
            LeapFrog() { cli = NULL; timer = NULL; };
            LeapFrog(std::string path, CL* cli, EB::Timer* timer);
            void execute(int num,
                        float dt,
                        //input
                        Buffer<float4>& pos_u,
                        Buffer<float4>& pos_s,
                        Buffer<float4>& vel_u,
                        Buffer<float4>& vel_s,
                        Buffer<float4>& veleval_u,
                        Buffer<float4>& force_s,
                        Buffer<float4>& xsph_s,
                        //Buffer<float4>& uvars, 
                        //Buffer<float4>& svars, 
                        Buffer<unsigned int>& indices,
                        //params
                        Buffer<OUTERParams>& sphp,
                        //debug
                        Buffer<float4>& clf_debug,
                        Buffer<int4>& cli_debug);

        private:
            CL* cli;
            Kernel k_leapfrog;
            EB::Timer* timer;
    };
}
}

#endif

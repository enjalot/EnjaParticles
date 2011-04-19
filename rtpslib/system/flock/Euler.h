#ifndef RTPS_EULER_H_INCLUDED
#define RTPS_EULER_H_INCLUDED


#include <CLL.h>
#include <Buffer.h>


namespace rtps
{
    class Euler
    {
        public:
            Euler() { cli = NULL; timer = NULL; };
            Euler(CL* cli, EB::Timer* timer);
            void execute(int num,
                        float dt,
                        Buffer<float4>& pos_u,
                        Buffer<float4>& pos_s,
                        Buffer<float4>& vel_u,
                        Buffer<float4>& vel_s,
                        Buffer<float4>& separation_s,
                        Buffer<float4>& alignment_s,
                        Buffer<float4>& cohesion_s,
                        Buffer<unsigned int>& indices,
                        //params
                        Buffer<FLOCKParameters>& flockp,
                        //debug
                        Buffer<float4>& clf_debug,
                        Buffer<int4>& cli_debug);
            
           

        private:
            CL* cli;
            Kernel k_euler;
            EB::Timer* timer;
    };
}

#endif

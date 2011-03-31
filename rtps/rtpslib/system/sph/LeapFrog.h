#ifndef RTPS_LEAPFROG_H_INCLUDED
#define RTPS_LEAPFROG_H_INCLUDED


#include <CLL.h>
#include <Buffer.h>


namespace rtps
{
    class LeapFrog
    {
        public:
            LeapFrog() { cli = NULL; timer = NULL; };
            LeapFrog(CL* cli, EB::Timer* timer);
            void execute(int num,
                        float dt,
                        //input
                        Buffer<float4>& pos,
                        Buffer<float4>& uvars, 
                        Buffer<float4>& svars, 
                        //output
                        Buffer<unsigned int>& indices,
                        //params
                        Buffer<SPHParams>& sphp,
                        //debug
                        Buffer<float4>& clf_debug,
                        Buffer<int4>& cli_debug);
            
           

        private:
            CL* cli;
            Kernel k_leapfrog;
            EB::Timer* timer;
    };
}

#endif

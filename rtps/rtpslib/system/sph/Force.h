#ifndef RTPS_FORCE_H_INCLUDED
#define RTPS_FORCE_H_INCLUDED


#include <CLL.h>
#include <Buffer.h>


namespace rtps
{
    class Force
    {
        public:
            Force() { cli = NULL; timer = NULL; };
            Force(CL* cli, EB::Timer* timer);
            void execute(int num,
                    //input
                    Buffer<float4>& svars, 
                    //output
                    Buffer<unsigned int>& ci_start,
                    Buffer<unsigned int>& ci_end,
                    //params
                    Buffer<SPHParams>& sphp,
                    Buffer<GridParams>& gp,
                    //debug params
                    Buffer<float4>& clf_debug,
                    Buffer<int4>& cli_debug);

        private:
            CL* cli;
            Kernel k_force;
            EB::Timer* timer;
    };
}

#endif

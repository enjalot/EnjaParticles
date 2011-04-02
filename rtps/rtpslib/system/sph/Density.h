#ifndef RTPS_DENSITY_H_INCLUDED
#define RTPS_DENSITY_H_INCLUDED


#include <CLL.h>
#include <Buffer.h>


namespace rtps
{
    class Density
    {
        public:
            Density() { cli = NULL; timer = NULL; };
            Density(CL* cli, EB::Timer* timer);
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
            Kernel k_density;
            EB::Timer* timer;
    };
}

#endif
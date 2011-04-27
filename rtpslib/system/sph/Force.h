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
            Force(std::string path, CL* cli, EB::Timer* timer);
            void execute(int num,
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
                    Buffer<int4>& cli_debug);

        private:
            CL* cli;
            Kernel k_force;
            EB::Timer* timer;
    };
}

#endif

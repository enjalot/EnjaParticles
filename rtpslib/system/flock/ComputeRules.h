#ifndef RTPS_COMPUTERULES_H_INCLUDED
#define RTPS_COMPUTERULES_H_INCLUDED


#include <CLL.h>
#include <Buffer.h>


namespace rtps 
{
    class ComputeRules
    {
        public:
            ComputeRules() { cli = NULL; timer = NULL; };
            ComputeRules(CL* cli, EB::Timer* timer);
            void execute(int num,
                    //input
                    //Buffer<float4>& svars, 
                    Buffer<float4>& pos_s, 
                    Buffer<float4>& sep_s, 
                    //output
                    Buffer<unsigned int>& ci_start,
                    Buffer<unsigned int>& ci_end,
                    //params
                    Buffer<FLOCKParameters>& flockp,
                    Buffer<GridParams>& gp,
                    //debug params
                    Buffer<float4>& clf_debug,
                    Buffer<int4>& cli_debug);

        private:
            CL* cli;
            Kernel k_computeRules;
            EB::Timer* timer;
    };
}

#endif

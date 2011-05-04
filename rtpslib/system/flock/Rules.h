#ifndef RTPS_RULES_H_INCLUDED
#define RTPS_RULES_H_INCLUDED


#include <CLL.h>
#include <Buffer.h>


namespace rtps 
{
    class Rules
    {
        public:
            Rules() { cli = NULL; timer = NULL; };
            Rules(std::string path, CL* cli, EB::Timer* timer);
            void executeFlockmates(int num,
                    //input
                    Buffer<int4>& neigh_s, 
                    //output
                    Buffer<unsigned int>& ci_start,
                    Buffer<unsigned int>& ci_end,
                    //params
                    Buffer<GridParams>& gp,
                    Buffer<FLOCKParameters>& flockp,
                    //debug params
                    Buffer<float4>& clf_debug,
                    Buffer<int4>& cli_debug);
            void executeSeparation(int num,
                    //input
                    Buffer<float4>& pos_s, 
                    Buffer<float4>& sep_s, 
                    Buffer<int4>& neigh_s, 
                    //output
                    Buffer<unsigned int>& ci_start,
                    Buffer<unsigned int>& ci_end,
                    //params
                    Buffer<GridParams>& gp,
                    Buffer<FLOCKParameters>& flockp,
                    //debug params
                    Buffer<float4>& clf_debug,
                    Buffer<int4>& cli_debug);
            void executeAlignment(int num,
                    //input
                    Buffer<float4>& vel_s, 
                    Buffer<float4>& align_s, 
                    Buffer<int4>& neigh_s, 
                    //output
                    Buffer<unsigned int>& ci_start,
                    Buffer<unsigned int>& ci_end,
                    //params
                    Buffer<GridParams>& gp,
                    Buffer<FLOCKParameters>& flockp,
                    //debug params
                    Buffer<float4>& clf_debug,
                    Buffer<int4>& cli_debug);
            void executeCohesion(int num,
                    //input
                    Buffer<float4>& pos_s, 
                    Buffer<float4>& coh_s, 
                    Buffer<int4>& neigh_s, 
                    //output
                    Buffer<unsigned int>& ci_start,
                    Buffer<unsigned int>& ci_end,
                    //params
                    Buffer<GridParams>& gp,
                    Buffer<FLOCKParameters>& flockp,
                    //debug params
                    Buffer<float4>& clf_debug,
                    Buffer<int4>& cli_debug);

        private:
            CL* cli;
            Kernel k_flockmates;
            Kernel k_rule_separation;
            Kernel k_rule_alignment;
            Kernel k_rule_cohesion;
            EB::Timer* timer;
    };
}

#endif

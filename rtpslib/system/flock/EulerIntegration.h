#ifndef RTPS_EULER_INTEGRATION_H_
#define RTPS_EULER_INTEGRATION_H_


#include <CLL.h>
#include <Buffer.h>


namespace rtps 
{
    class EulerIntegration
    {
        public:
            EulerIntegration() { cli = NULL; timer = NULL; };
            EulerIntegration(std::string path, CL* cli, EB::Timer* timer);
            void execute(int num,
                        float dt,
                        Buffer<float4>& pos_u,
                        Buffer<float4>& pos_s,
                        Buffer<float4>& vel_u,
                        Buffer<float4>& vel_s,
                        Buffer<float4>& separation_s,
                        Buffer<float4>& alignment_s,
                        Buffer<float4>& cohesion_s,
                        Buffer<float4>& goal_s,
                        Buffer<float4>& avoid_s,
                        Buffer<float4>& leaderfollowing_s,
                        Buffer<unsigned int>& indices,
                        //params
                        Buffer<FLOCKParameters>& flockp,
                        Buffer<GridParams>& gridp,
                        //debug
                        Buffer<float4>& clf_debug,
                        Buffer<int4>& cli_debug);
            
           

        private:
            CL* cli;
            Kernel k_euler_integration;
            EB::Timer* timer;
    };
}

#endif

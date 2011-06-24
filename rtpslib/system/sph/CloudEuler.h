#ifndef RTPS_CLOUD_EULER_H_INCLUDED
#define RTPS_CLOUD_EULER_H_INCLUDED


#include <CLL.h>
#include <Buffer.h>

namespace rtps
{
    class CloudEuler
    {
        public:
            CloudEuler() { cli = NULL; timer = NULL; };
            CloudEuler(std::string path, CL* cli, EB::Timer* timer);
            void execute(int num,
                        float dt,
                        Buffer<float4>& pos_u,
                        Buffer<float4>& pos_s,
                        Buffer<float4>& normal_u,
                        Buffer<float4>& normal_s,
                        Buffer<float4>& velocity_u,
                        Buffer<float4>& velocity_s,
						//float4 vel,
                        Buffer<unsigned int>& indices,
                        //params
                        Buffer<SPHParams>& sphp,
                        //debug
                        Buffer<float4>& clf_debug,
                        Buffer<int4>& cli_debug);
            
           

        private:
            CL* cli;
            Kernel k_cloud_euler;
            EB::Timer* timer;
    };
}

#endif

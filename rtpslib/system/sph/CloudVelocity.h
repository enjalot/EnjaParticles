#ifndef RTPS_CLOUD_VELOCITY_H_INCLUDED
#define RTPS_CLOUD_VELOCITY_H_INCLUDED


#include <CLL.h>
#include <Buffer.h>

#include "CloudVelocity.h"


namespace rtps
{
    class CloudVelocity
    {
        public:
            CloudVelocity() { cli = NULL; timer = NULL; };
            CloudVelocity(std::string path, CL* cli, EB::Timer* timer);
            void execute(int num,
						float time,
						float delta_angle,
                        Buffer<float4>& pos_s,
                        Buffer<float4>& vel_s,
						float4& pos_cg,
						float4& omega);
            
           

        private:
            CL* cli;
            Kernel k_cloud_velocity;
            EB::Timer* timer;
    };
}

#endif

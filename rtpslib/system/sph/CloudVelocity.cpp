#include "../CLOUD.h"

namespace rtps
{
    CloudVelocity::CloudVelocity(std::string path, CL* cli_, EB::Timer* timer_)
    {
        cli = cli_;
        timer = timer_;
 
        printf("create cloud velocity kernel\n");
        path += "/cloud_velocity.cl";
        k_cloud_velocity = Kernel(cli, path, "kern_cloud_velocity");
    } 
    
	//----------------------------------------------------------------------
    void CloudVelocity::execute(int num,
                    float time, // or dt?
					float delta_angle,
                    Buffer<float4>& pos_s,
                    Buffer<float4>& vel_s,
                    float4& pos_cg,
                    float4& omega)
    {
        int iargs = 0;
        k_cloud_velocity.setArg(iargs++, num);
        k_cloud_velocity.setArg(iargs++, delta_angle);
        k_cloud_velocity.setArg(iargs++, pos_s.getDevicePtr());
        k_cloud_velocity.setArg(iargs++, vel_s.getDevicePtr());
        k_cloud_velocity.setArg(iargs++, pos_cg);
        k_cloud_velocity.setArg(iargs++, omega);

        int local_size = 128;
        k_cloud_velocity.execute(num, local_size);
    }
}
//----------------------------------------------------------------------

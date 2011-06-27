#include "../SPH.h"

namespace rtps
{
    CloudEuler::CloudEuler(std::string path, CL* cli_, EB::Timer* timer_)
    {
        cli = cli_;
        timer = timer_;
 
        printf("create euler kernel\n");
        path += "/cloud_euler.cl";
        k_cloud_euler = Kernel(cli, path, "cloudEuler");
    } 

    void CloudEuler::execute(int num,
                    float dt,
                    Buffer<float4>& pos_u,
                    Buffer<float4>& pos_s,
                    Buffer<float4>& normal_u,
                    Buffer<float4>& normal_s,
                    Buffer<float4>& velocity_u,
                    Buffer<float4>& velocity_s,
					float4& pos_cg,
					float4& diff_pos_cg,
                    //float4 vel,
                    Buffer<unsigned int>& indices,
                    //params
                    Buffer<SPHParams>& sphp,
                    //debug params
                    Buffer<float4>& clf_debug,
                    Buffer<int4>& cli_debug)
    {
        int iargs = 0;
        k_cloud_euler.setArg(iargs++, num);
        k_cloud_euler.setArg(iargs++, pos_u.getDevicePtr());
        k_cloud_euler.setArg(iargs++, pos_s.getDevicePtr());
        k_cloud_euler.setArg(iargs++, normal_u.getDevicePtr());
        k_cloud_euler.setArg(iargs++, normal_s.getDevicePtr());
        k_cloud_euler.setArg(iargs++, velocity_u.getDevicePtr());
        k_cloud_euler.setArg(iargs++, velocity_s.getDevicePtr());
        k_cloud_euler.setArg(iargs++, pos_cg);
        k_cloud_euler.setArg(iargs++, diff_pos_cg);
        //k_cloud_euler.setArg(iargs++, vel);
        k_cloud_euler.setArg(iargs++, indices.getDevicePtr());
        k_cloud_euler.setArg(iargs++, sphp.getDevicePtr());
        k_cloud_euler.setArg(iargs++, dt); //time step

		pos_cg.print("*** pos_cg ***");


		//printf("BEFORE k_cloud_euler.execute\n");

        int local_size = 128;
        k_cloud_euler.execute(num, local_size);
    }

	// NO CPU IMPLEMENTATION
}

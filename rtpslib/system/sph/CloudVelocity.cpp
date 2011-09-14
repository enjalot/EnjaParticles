/****************************************************************************************
* Real-Time Particle System - An OpenCL based Particle system developed to run on modern GPUs. Includes SPH fluid simulations.
* version 1.0, September 14th 2011
* 
* Copyright (C) 2011 Ian Johnson, Andrew Young, Gordon Erlebacher, Myrna Merced, Evan Bollig
* 
* This software is provided 'as-is', without any express or implied
* warranty.  In no event will the authors be held liable for any damages
* arising from the use of this software.
* 
* Permission is granted to anyone to use this software for any purpose,
* including commercial applications, and to alter it and redistribute it
* freely, subject to the following restrictions:
* 
* 1. The origin of this software must not be misrepresented; you must not
* claim that you wrote the original software. If you use this software
* in a product, an acknowledgment in the product documentation would be
* appreciated but is not required.
* 2. Altered source versions must be plainly marked as such, and must not be
* misrepresented as being the original software.
* 3. This notice may not be removed or altered from any source distribution.
****************************************************************************************/


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

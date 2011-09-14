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


#include "../FLOCK.h"

namespace rtps 
{

    EulerIntegration::EulerIntegration(std::string path, CL* cli_, EB::Timer* timer_)
    {
        cli = cli_;
        timer = timer_;
 
        printf("create euler_integration kernel\n");
        path += "/euler_integration.cl";
        k_euler_integration = Kernel(cli, path, "euler_integration");
    } 
    
    void EulerIntegration::execute(int num,
                    float dt,
                    bool two_dimensional,
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
                    //debug params
                    Buffer<float4>& clf_debug,
                    Buffer<int4>& cli_debug)
    {

        int iargs = 0;
        k_euler_integration.setArg(iargs++, dt); //time step
        k_euler_integration.setArg(iargs++, two_dimensional); //2D or 3D?
        k_euler_integration.setArg(iargs++, pos_u.getDevicePtr());
        k_euler_integration.setArg(iargs++, pos_s.getDevicePtr());
        k_euler_integration.setArg(iargs++, vel_u.getDevicePtr());
        k_euler_integration.setArg(iargs++, vel_s.getDevicePtr());
        k_euler_integration.setArg(iargs++, separation_s.getDevicePtr());
        k_euler_integration.setArg(iargs++, alignment_s.getDevicePtr());
        k_euler_integration.setArg(iargs++, cohesion_s.getDevicePtr());
        k_euler_integration.setArg(iargs++, goal_s.getDevicePtr());
        k_euler_integration.setArg(iargs++, avoid_s.getDevicePtr());
        k_euler_integration.setArg(iargs++, leaderfollowing_s.getDevicePtr());
        k_euler_integration.setArg(iargs++, indices.getDevicePtr());
        k_euler_integration.setArg(iargs++, flockp.getDevicePtr());
        k_euler_integration.setArg(iargs++, gridp.getDevicePtr());
        
        // ONLY IF DEBUGGING
        k_euler_integration.setArg(iargs++, clf_debug.getDevicePtr());
        k_euler_integration.setArg(iargs++, cli_debug.getDevicePtr());


        int local_size = 128;
        k_euler_integration.execute(num, local_size);

    }

    void FLOCK::cpuEulerIntegration()
    {
        float w_sep = flock_params.w_sep;   
        float w_aln = flock_params.w_align;
        float w_coh = flock_params.w_coh;  
        float w_goal = flock_params.w_goal;
        float w_avoid = flock_params.w_avoid;
        float w_wander = flock_params.w_wander;

        float4 bndMax = grid_params.bnd_max;
        float4 bndMin = grid_params.bnd_min;
        
        for(int i = 0; i < num; i++)
        { 
            float4 pi = positions[i]; //* flock_params.simulation_scale;

           // Step 4. Weight the steering behaviors
	        separation[i] *=  w_sep;
	        alignment[i]  *=  w_aln;
	        cohesion[i]   *=  w_coh;
            goal[i]       *=  w_goal;
            avoid[i]      *=  w_avoid;
            wander[i]     *=  w_wander;

            // Step 5. Set the final velocity
	        velocities[i] += (separation[i] + alignment[i] + cohesion[i] + goal[i] + avoid[i] + wander[i]);
        
            // Step 6. Constrain velocity
            float  vel_mag  = velocities[i].length();
	        float4 vel_norm = normalize3(velocities[i]);
            if(vel_mag > flock_params.max_speed){
                // set magnitude to max speed 
                 velocities[i] = vel_norm * flock_params.max_speed;
            }

            // (Optional Step) Add circular velocity
            float4 v = float4(-pi.y, pi.x, 0.f,  0.f);
            v *= flock_params.ang_vel;
            velocities[i] += v;

            // Step 7. Integration 
            pi += ps->settings->dt*velocities[i];
            pi.w = 1.0f; //just in case

    	    // Step 8. Check boundary conditions
            if(pi.x >= bndMax.x){
                pi.x -= bndMax.x;
            }   
            else if(pi.x <= bndMin.x){
                pi.x += bndMax.x;
            }
            else if(pi.y >= bndMax.y){
                pi.y -= bndMax.y;
            }   
            else if(pi.y <= bndMin.y){
                pi.y += bndMax.y;
            }
            else if(pi.z >= bndMax.z){
                pi.z -= bndMax.z;
            }
            else if(pi.z <= bndMin.z){
                pi.z += bndMax.z;
            }

            // Step 9. 2D or 3D?
            if(ps->settings->two_dimensional)
                pi.z =0.f;

            // Copy positions to official vector
            positions[i] = pi; // flock_params.simulation_scale;
            positions[i].w = 1.0f;	
        }
    }

}

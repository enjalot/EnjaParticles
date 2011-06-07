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
                    Buffer<float4>& pos_u,
                    Buffer<float4>& pos_s,
                    Buffer<float4>& vel_u,
                    Buffer<float4>& vel_s,
                    Buffer<float4>& separation_s,
                    Buffer<float4>& alignment_s, 
                    Buffer<float4>& cohesion_s, 
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
        k_euler_integration.setArg(iargs++, pos_u.getDevicePtr());
        k_euler_integration.setArg(iargs++, pos_s.getDevicePtr());
        k_euler_integration.setArg(iargs++, vel_u.getDevicePtr());
        k_euler_integration.setArg(iargs++, vel_s.getDevicePtr());
        k_euler_integration.setArg(iargs++, separation_s.getDevicePtr());
        k_euler_integration.setArg(iargs++, alignment_s.getDevicePtr());
        k_euler_integration.setArg(iargs++, cohesion_s.getDevicePtr());
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

        float4 bndMax = grid_params.bnd_max;
        float4 bndMin = grid_params.bnd_min;
        
        for(int i = 0; i < num; i++)
        { 
            float4 pi = positions[i]; //* flock_params.simulation_scale;

           // Step 4. Weight the steering behaviors
	        separation[i] *=  w_sep;
	        alignment[i]  *=  w_aln;
	        cohesion[i]   *=  w_coh;

            // Step 5. Set the final acceleration
	        velocities[i] += (separation[i] + alignment[i] + cohesion[i]);
        
            // Step 6. Constrain acceleration
            float  vel_mag  = velocities[i].length();
	        float4 vel_norm = normalize3(velocities[i]);
            if(vel_mag > flock_params.max_speed){
                // set magnitude to max speed 
                 velocities[i] = vel_norm * flock_params.max_speed;
            }

            // (Optional Step) Add circular velocity
            float4 v = float4(-pi.z, 0., pi.x, 0.);
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

            positions[i] = pi; // flock_params.simulation_scale;
            positions[i].w = 1.0f;	
        }
    }

}

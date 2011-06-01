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
        float4 acc;
        float dist;

        float4 vel_sep, vel_aln, vel_coh;
        float4 acc_separation, acc_alignment, acc_cohesion;

        float w_sep = flock_params.w_sep;   //0.0001f;  //.0003f;
        float w_aln = flock_params.w_align;  //0.0001f;  //0.0001f;
        float w_coh = flock_params.w_coh;   //0.00003f;  //0.00003f;

        float4 bndMax = grid_params.bnd_max;
        float4 bndMin = grid_params.bnd_min;

	    float spacing = spacing;
	    spacing *= 2;

        int nb_cells = (int)((bndMax.x-bndMin.x)/spacing) * (int)((bndMax.y-bndMin.y)/spacing) * (int)((bndMax.z-bndMin.z)/spacing);
        vector<int>* flockmates = new vector<int>[nb_cells];

        // Step 1. Loop over all boids
        for(int i = 0; i < num; i++)
        {
	        // initialize acc to zero
            acc = float4(0.f, 0.f, 0.f, 1.f);

            // Step 2. Search for neighbors
            for(int j=0; j < num; j++){
                if(j == i){
			        continue;
                }
                float4 d = positions[i] - positions[j];
		        dist = d.length();

                // is boid j a flockmate?
                if(dist <= flock_params.search_radius){
                    (*flockmates).push_back(j);
                }
            }   
	
            // Step 3. If they are flockmates, compute the three rules
            if((*flockmates).size() > 0){
		        acc_separation = float4(0.f, 0.f, 0.f, 1.f);
		        acc_alignment  = float4(0.f, 0.f, 0.f, 1.f);
		        acc_cohesion   = float4(0.f, 0.f, 0.f, 1.f);

		        int count = 0;

                // 3.1 Separation
		        for(int j=0; j < (*flockmates).size(); j++){
        		    float4 separation =  positions[i] - positions[(*flockmates)[j]];
			    
                    float d = separation.length();
        		    if(d < flock_params.min_dist){
                		separation = normalize3(separation);
				        separation = separation / d;
				        acc_separation = acc_separation + separation;
				        count++;
        		    }   
		        }

		        if(count > 0){
			        acc_separation = acc_separation / count;
			        acc_separation = normalize3(acc_separation);
		        }

		        // 3.2 Alignment
		        float4 avg_a = float4(0.f, 0.f, 0.f, 0.f);
		        for(int j=0; j < (*flockmates).size(); j++){
                    avg_a = avg_a + velocities[(*flockmates)[j]];
	    	    }   
		        avg_a = (*flockmates).size() > 0 ? avg_a / (*flockmates).size() : avg_a;
		        acc_alignment = avg_a - velocities[i];
		        acc_alignment = normalize3(acc_alignment); 


		        // 3.3 Cohesion
                float4 avg_c = float4(0.f, 0.f, 0.f, 0.f);
                for(int j=0; j < (*flockmates).size(); j++){
                    avg_c = avg_c + positions[(*flockmates)[j]];
                }
                avg_c = (*flockmates).size() > 0 ? avg_c / (*flockmates).size() : avg_c;
                acc_cohesion = avg_c - positions[i];
                acc_cohesion = normalize3(acc_cohesion);

                (*flockmates).clear();
	        }   

            // Step 4. Weight the steering behaviors
	        vel_sep = acc_separation * w_sep;
	        vel_aln = acc_alignment  * w_aln;
	        vel_coh = acc_cohesion   * w_coh;

            // Step 5. Set the final acceleration
	        acc = velocities[i] + vel_sep + vel_aln + vel_coh;
        
            // Step 6. Constrain acceleration
            float  acc_mag  = acc.length();
	        float4 acc_norm = normalize3(acc);
            if(acc_mag > flock_params.max_speed){
                // set magnitude to max speed 
                acc = acc_norm * flock_params.max_speed;
            }


            // (Optional Step) Add circular velocity
            float4 v = float4(-positions[i].z, 0., positions[i].x, 0.);
            v = v*flock_params.ang_vel; // 0.0005
            velocities[i] = v + acc;

    	    // Step 7. Integration 
	        positions[i] = positions[i] + ps->settings->dt*velocities[i];
            positions[i].w = 1.0f; //just in case

    	    // Step 8. Check boundary conditions
            if(positions[i].x >= bndMax.x){
                positions[i].x -= bndMax.x;
            }   
            else if(positions[i].x <= bndMin.x){
                positions[i].x += bndMax.x;
            }
            else if(positions[i].y >= bndMax.y){
                positions[i].y -= bndMax.y;
            }   
            else if(positions[i].y <= bndMin.y){
                positions[i].y += bndMax.y;
            }
            else if(positions[i].z >= bndMax.z){
                positions[i].z -= bndMax.z;
            }
            else if(positions[i].z <= bndMin.z){
                positions[i].z += bndMax.z;
            }
		
        }
        delete [] flockmates;
    }

}

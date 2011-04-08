
//----------------------------------------------------------------------
void FLOCK::ge_loadEuler()
{
    printf("create euler kernel\n");

    std::string path(FLOCK_CL_SOURCE_DIR);
    path += "/euler_cl.cl";
    k_euler = Kernel(ps->cli, path, "euler");
  
    int iargs = 0;
    k_euler.setArg(iargs++, cl_sort_indices.getDevicePtr());
    k_euler.setArg(iargs++, cl_vars_unsorted.getDevicePtr());
    k_euler.setArg(iargs++, cl_vars_sorted.getDevicePtr());
    k_euler.setArg(iargs++, cl_position.getDevicePtr());
    k_euler.setArg(iargs++, cl_FLOCKParameters.getDevicePtr());
    k_euler.setArg(iargs++, ps->settings.dt);           //time step
    k_euler.setArg(iargs++, ps->settings.min_dist);     //desired separation distance
    k_euler.setArg(iargs++, ps->settings.search_radius);//searching radius for neighbors
    k_euler.setArg(iargs++, ps->settings.max_speed);    //max speed for the boids

	// ONLY IF DEBUGGING
	k_euler.setArg(iargs++, clf_debug.getDevicePtr());
	k_euler.setArg(iargs++, cli_debug.getDevicePtr());



} 

//----------------------------------------------------------------------
void FLOCK::ge_cpuEuler()
{
    float4 acc;
    float dist;

    float4 vel_sep, vel_aln, vel_coh;
    float4 acc_separation, acc_alignment, acc_cohesion;

    float w_sep = 0.0001f;  //.0003f;
    float w_aln = 0.0001f;  //0.0001f;
    float w_coh = 0.00003f;  //0.00003f;

    float4 bndMax = flock_params.grid_max;
    float4 bndMin = flock_params.grid_min;

	float spacing = flock_settings.spacing;
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
            if(dist <= ps->settings.search_radius){
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
        		if(d < ps->settings.min_dist){
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
        if(acc_mag > ps->settings.max_speed){
                // set magnitude to max speed 
                acc = acc_norm * ps->settings.max_speed;
        }


        // (Optional Step) Add circular velocity
        float4 v = float4(-positions[i].z, 0., positions[i].x, 0.);
        v = v*.000; // 0.0005
        velocities[i] = v + acc;

    	// Step 7. Integrate        
	    positions[i] = positions[i] + ps->settings.dt*velocities[i];
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


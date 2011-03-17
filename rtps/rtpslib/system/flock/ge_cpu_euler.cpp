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
    k_euler.setArg(iargs++, cl_FLOCKParams.getDevicePtr());
    k_euler.setArg(iargs++, ps->settings.dt); //time step

	// ONLY IF DEBUGGING
	k_euler.setArg(iargs++, clf_debug.getDevicePtr());
	k_euler.setArg(iargs++, cli_debug.getDevicePtr());



} 

//----------------------------------------------------------------------
void FLOCK::ge_cpuEuler()
{
    #define searchradius 	    0.5f         // 0.3
    #define separationdist  	0.25f       // 0.08
    #define maxspeed        	0.03f      // 0.003f
    #define desiredspeed    	0.025f     // 0.0025f
    #define maxchange       	0.05f      // 0.005f
    #define MinUrgency      	0.025f     // 0.0025f
    #define MaxUrgency      	0.05f      // 0.005f


    //static int count = 0;

    float h = ps->settings.dt;

    printf("enter CPUEuler\n");
    //printf("enter CPUEuler: count= %d\n", count);
    //count++;

    float4 acc;
    float dist;

    float4 vel_sep, vel_aln, vel_coh;
    float4 acc_separation, acc_alignment, acc_cohesion;

    float w_sep = 0.0003f;  //.0005f;
    float w_aln = 0.0001f;  //0.0005f;
    float w_coh = 0.00003f;  //0.00001f;
//printf("10\n");//GE

    float4 bndMax = params.grid_max;// - params->boundary_distance;
    float4 bndMin = params.grid_min;// + params->boundary_distance;
//printf("11\n");//GE
bndMax.print("bndMax");
bndMin.print("bndMin");
printf("h= %f\n", h);

	float hh = flock_settings.spacing;
	hh *= 2;

	// ARE YOU SURE nb_CELLS WILL BE CORRECT? 
	// what about rounding errors because we dealing with floats? 
    int nb_cells = (int)((bndMax.x-bndMin.x)/hh) * (int)((bndMax.y-bndMin.y)/hh) * (int)((bndMax.z-bndMin.z)/hh);
//printf("12\n");//GE 
printf("nb_cells= %d\n", nb_cells);
    vector<int>* flockmates = new vector<int>[nb_cells];
//printf("13\n");//GE 
//   int MaxFlockmates = num / 2;
//   int flockmates[MaxFlockmates];
//   int numFlockmates = 0;

printf("loop over all boids\n");
printf("nb_cells: %d\n", nb_cells);
printf("flockmates size: %d\n", (*flockmates).size()); 
//exit(0);
printf("num: %d\n\n", num);
    // loop over all boids
    for(int i = 0; i < num; i++)
    {
//	printf("boid %d\n", i);
    //positions[i].print("position of the boid");
	// boid in case
	//pi = positions[i];
        //vi = velocities[i];

#if 0
	    // boid in case
	    p1 = positions[i];
		if (p1.x != p1.x || p1.y != p1.y || p1.z != p1.z || p1.w != p1.w) {
			printf("BEFORE, p1 = nan\n");
			p1.print("BEFORE p1");
			exit(0);
		}
        v1 = velocities[i];
#endif
	    // initialize acc to zero
        acc = float4(0.f, 0.f, 0.f, 1.f);
        //numFlockmates = 0;

//printf("11\n");//GE
	#if 0 
	// print boid info
		if (i == 0 || i == 1) {
			printf("================= Position boid %d ==============\n", i);
			printf("Euler: p[%d]= %f, %f, %f, %f\n", i, p1.x, p1.y, p1.z, p1.w);
			printf("       v[%d]= %f, %f, %f, %f\n", i, v1.x, v1.y, v1.z, v1.w);
			printf("       h    = %f\n",h);
		}
	#endif

        // Step 2. Search for neighbors
	// loop over all boids
        for(int j=0; j < num; j++){
             	if(j == i){
                    printf("boid %d and %d are the same\n", i, j);
			        continue;
                }

              	float4 d = positions[i] - positions[j];
		        dist = d.length();

                // is boid j a flockmate?
                if(dist <= searchradius){
    //                    printf("boid %d is neighbor of boid %d\n", i, j);
                    	(*flockmates).push_back(j);
      //                  printf("neigh index: %d\n", (*flockmates)[(*flockmates).size()-1]);
                       // positions[(*flockmates)[(*flockmates).size()-1]].print("neigh position");
             	}
        }
//printf("search for neighbors done\n");
//printf("============== numFlockmates: %d ==============\n", numFlockmates); 
	
        // Step 3. If they are flockmates, compute the three rules
        if((*flockmates).size() > 0){
		acc_separation = float4(0.f, 0.f, 0.f, 1.f);
		acc_alignment  = float4(0.f, 0.f, 0.f, 1.f);
		acc_cohesion   = float4(0.f, 0.f, 0.f, 1.f);

		int count = 0;

                // Separation
//printf("12\n");//GE
//printf("flockmates size: %d\n", (*flockmates).size()); 
		for(int j=0; j < (*flockmates).size(); j++){
                	//pj = positions[flockmates[j]];
			//vj = velocities[flockmates[j]];

        		float4 separation =  positions[i] - positions[(*flockmates)[j]];
        //    positions[i].print("positions[i]: ");
        //    positions[(*flockmates)[j]].print("positions[j]: ");
        //    separation.print("separation: ");

		if (separation.x != separation.x || separation.y != separation.y || separation.z != separation.z || separation.w != separation.w) {
            separation.print("separation: ");
            printf("boid: %d separation\n", i);
            exit(0);
        }

			    float d = separation.length();
                	//r = d / separationdist;  

        		//separation = normalize3(separation);

        		//if(d >= separationdist){
                	//	separation = separation * r;
        		//}
        		if(d < separationdist){
                		//separation = separation * -r;
                		separation = normalize3(separation);
				        separation = separation / d;
				        acc_separation = acc_separation + separation;
				        count++;

		if (separation.x != separation.x || separation.y != separation.y || separation.z != separation.z || separation.w != separation.w) {
            separation.print("separation: ");
            printf("boid: %d inside separation dist statement\n", i);
            exit(0);
        }

		if (acc_separation.x != acc_separation.x || acc_separation.y != acc_separation.y || acc_separation.z != acc_separation.z || acc_separation.w != acc_separation.w) {
            acc_separation.print("acc_separation: ");
            printf("boid: %d inside separation dist statement\n", i);
            exit(0);
        }

        		}
		}

//printf("13\n");//GE
		if(count > 0){
			acc_separation = acc_separation / count;
			acc_separation = normalize3(acc_separation);
		}
		//acc_separation += separation;
		//acc_separation.w = 1.f;

		if (acc_separation.x != acc_separation.x || acc_separation.y != acc_separation.y || acc_separation.z != acc_separation.z || acc_separation.w != acc_separation.w) {
            acc_separation.print("acc_separation: ");
            printf("boid: %d after computing acc_separation\n", i);
            exit(0);
        }

		// Alignment
		float4 avg_a = float4(0.f, 0.f, 0.f, 0.f);
		for(int j=0; j < (*flockmates).size(); j++){
                	avg_a = avg_a + velocities[(*flockmates)[j]];
	    	}
        //avg_a.print("average before");
		avg_a = (*flockmates).size() > 0 ? avg_a / (*flockmates).size() : avg_a;
        //avg_a.print("average after");
		acc_alignment = avg_a - velocities[i];
        //acc_alignment.print("acc aln before");
		acc_alignment = normalize3(acc_alignment); 
        //acc_alignment.print("acc aln after");
		//acc_alignment += v2;
	   	//acc_alignment.w = 1.f;

		if (acc_alignment.x != acc_alignment.x || acc_alignment.y != acc_alignment.y || acc_alignment.z != acc_alignment.z || acc_alignment.w != acc_alignment.w) {
            acc_alignment.print("acc_alignment: ");
            printf("boid: %d\n", i);
            exit(0);
        }

		// Cohesion
                float4 avg_c = float4(0.f, 0.f, 0.f, 0.f);
                for(int j=0; j < (*flockmates).size(); j++){
                        avg_c = avg_c + positions[(*flockmates)[j]];
                }
                avg_c = (*flockmates).size() > 0 ? avg_c / (*flockmates).size() : avg_c;
                acc_cohesion = avg_c - positions[i];
                acc_cohesion = normalize3(acc_cohesion);
		//acc_cohesion += p2;
		//acc_cohesion.w = 1.f;



		if (acc_cohesion.x != acc_cohesion.x || acc_cohesion.y != acc_cohesion.y || acc_cohesion.z != acc_cohesion.z || acc_cohesion.w != acc_cohesion.w) {
            acc_cohesion.print("acc_cohesion: ");
            printf("boid: %d\n", i);
            exit(0);
        }

(*flockmates).clear();
//p1.print("p1: ");
//p2.print("p2: ");
//exit(0);
	}   


//printf("\n");//GE
	vel_sep = acc_separation * w_sep;
	vel_aln = acc_alignment  * w_aln;
	vel_coh = acc_cohesion   * w_coh;
//vel_sep.print("separation");
//vel_aln.print("alignment");
//vel_coh.print("cohesion");

//printf("acc: %f, %f, %f, %f\n", )	
//exit(0);
#if 0
            // adding the rules to the acceleration vector

		    // Separation
			acc_separation.print("acc_separation");
		    acc += acc_separation * w_sep;
//acc_separation.print("separation: ");
		    // Alignment
			acc_alignment.print("1 acc_alignment"); // nan
		    acc_alignment = acc_alignment / numFlockmates;
			acc_alignment.print("2 acc_alignment"); // nan
			printf("numFlockmakes= %d\n", numFlockmates);
			//v1.print("v1");
		    acc_alignment = acc_alignment - v1;
			acc_alignment.print("2.5 acc_alignment"); // nan
		    acc_alignment = normalize3(acc_alignment);
			acc_alignment.print("3 acc_alignment"); // nan
		
		    acc += acc_alignment * w_aln;
//acc_alignment.print("alignment: ");

		    // Cohesion
		    acc_cohesion = acc_cohesion / numFlockmates;
		    acc_cohesion = acc_cohesion - p1;
		    acc_cohesion = normalize3(acc_cohesion);
			acc_cohesion.print("acc_cohesion");

		    acc += acc_cohesion * w_coh;
//acc_cohesion.print("cohesion: ");
#endif
//	    }


	acc = velocities[i] + vel_sep + vel_aln + vel_coh;
//acc.print("acceleration before constraint");

//acc.z = 0.f;
//acc.w = 1.f;

//printf("4\n");//GE
        // Step 4. Constrain acceleration
        float  acc_mag  = acc.length();
	    float4 acc_norm = normalize3(acc);
 
        if(acc_mag > maxchange){
                // set magnitude to MaxChangeInAcc
                acc = acc_norm * maxchange;
        }

//acc.print("acceleration after constraint");

        // Step 5. Add acceleration to velocity
        float4 v = float4(-positions[i].y, positions[i].x, 0., 0.);
        v = v*.00;
        velocities[i] = v + acc;
//velocities[i].print("final velocity");
        //velocities[i] = acc;

	// remove for debugging
        //v1.x += MinUrgency;

#if 0
        // Step 6. Constrain velocity
        float speed = magnitude3(v1);
        if(speed > maxspeed){
                // set magnitude to MaxSpeed
                v1 = v1 * (maxspeed/speed);
        }
printf("3\n");//GE
v1.print("velocity: "); 
#endif
        //float scale = params.simulation_scale;
        //v.x += h*f.x / scale;
        //v.y += h*f.y / scale;
        //v.z += h*f.z / scale;
	
    	// Step 7. Integrate        
        //p1.x += h*v1.x;
        //p1.y += h*v1.y;
        //p1.z += h*v1.z;
	    positions[i] = positions[i] + velocities[i];
        positions[i].w = 1.0f; //just in case

//printf("2\n");//GE
#if 0
		if (p1.x != p1.x || p1.y != p1.y || p1.z != p1.z || p1.w != p1.w) {
			printf("h= %f\n", (float) h);
			v1.print("v1");
		}
#endif
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

		
#if 0
        if (i == 0) {
			printf("================= Final position ==============\n");
			printf("Euler: p[%d]= %f, %f, %f, %f\n", i, p1.x, p1.y, p1.z, p1.w);
			printf("       v[%d]= %f, %f, %f, %f\n", i, v1.x, v1.y, v1.z, v1.w);
			printf("       h    = %f\n",h);
		}
#endif

//printf("%d\n",i);
//printf("vs: %d\n",velocities.size());
//printf("ps: %d\n",positions.size());

#if 0
        // write values
        velocities[i] = v1;
        positions[i] = p1;
		if (p1.x != p1.x || p1.y != p1.y || p1.z != p1.z || p1.w != p1.w) {
			printf("GE i= %d\n", i);
			printf("AFTER nan\n");
			p1.print("AFTER p1");
			exit(1);
		}
#endif
//printf("1\n");exit(0);  //GE
    }
delete [] flockmates;
    //printf("v.z %f p.z %f \n", velocities[0].z, positions[0].z);
}


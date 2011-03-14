#include "../FLOCK.h"

namespace rtps {

void FLOCK::loadEuler()
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

void FLOCK::cpuEuler()
{
    #define searchradius 	    0.08f
    #define separationdist  	0.03f
    #define maxspeed        	3.f     // 1.f
    #define desiredspeed    	1.5f    // .5f
    #define maxchange       	0.1f    // .1f
    #define MinUrgency      	0.05f   // .05f
    #define MaxUrgency      	0.1f    // .1f

    float h = ps->settings.dt;

    float4 acc;
    int numFlockmates=0;
    float4 p1, p2, v1, v2;
    float d, r;

    int MaxFlockmates = num/2;
    int flockmates[MaxFlockmates];

    float4 separation, alignment, cohesion;
    float4 acc_separation, acc_alignment, acc_cohesion;

    float w_sep = 0.0f;
    float w_aln = .000f;
    float w_coh = 0.0f;

    float4 bndMax = params.grid_max;// - params->boundary_distance;
    float4 bndMin = params.grid_min;// + params->boundary_distance;

    // loop over all boids
    for(int i = 0; i < num; i++)
    {
	    // boid in case
	    p1 = positions[i];
        v1 = velocities[i];

	    // initialize acc to zero
        acc = float4(0.f, 0.f, 0.f, 0.f);
        numFlockmates = 0;

	#if 1 
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
             if(j != i){
                    p2 = positions[j];
//p2.print("p2");
                    d = sqrt((p2.x-p1.x)*(p2.x-p1.x) + (p2.y-p1.y)*(p2.y-p1.y) + (p2.z-p1.z)*(p2.z-p1.z));

                    // is boid j a flockmate?
                    if(d < searchradius){
                            flockmates[numFlockmates] = j;
                            numFlockmates++;

                            // did I find the max num of flockmates already?
                            if(numFlockmates == MaxFlockmates) break;
                    }
             }
        }
//printf("search for neighbors done\n");
//printf("============== numFlockmates: %d ==============\n", numFlockmates); 
	
        // Step 3. If they are flockmates, compute the three rules
        if(numFlockmates > 0){
		    acc_separation = float4(0.f, 0.f, 0.f, 0.f);
		    acc_alignment  = float4(0.f, 0.f, 0.f, 0.f);
		    acc_cohesion   = float4(0.f, 0.f, 0.f, 0.f);

		    for(int j=0; j < numFlockmates; j++){
                p2 = positions[flockmates[j]];
			    v2 = velocities[flockmates[j]];
        		
                // Separation
        		separation = p2- p1;
               	d = sqrt((p2.x-p1.x)*(p2.x-p1.x) + (p2.y-p1.y)*(p2.y-p1.y) + (p2.z-p1.z)*(p2.z-p1.z));
                r = d / separationdist;  

        		separation = normalize(separation);

        		if(d >= separationdist){
                		separation = separation * r;
        		}
        		else if(d <= separationdist){
                		separation = separation * -r;
        		}
        		else{
                		separation = separation * 0.f;
        		}

			    acc_separation += separation;
			    acc_separation.w = 1.f;

			    // Alignment
			    acc_alignment += v2;
			    acc_alignment.w = 1.f;

			    // Cohesion
			    acc_cohesion += p2;
			    acc_cohesion.w = 1.f;
//p1.print("p1: ");
//p2.print("p2: ");
//exit(0);
		    }   

//printf("acc: %f, %f, %f, %f\n", )	
//exit(0);
		
            // adding the rules to the acceleration vector

		    // Separation
		    acc += acc_separation * w_sep;

		    // Alignment
		    acc_alignment = acc_alignment / numFlockmates;
		    acc_alignment = acc_alignment - v1;
		    acc_alignment = normalize(acc_alignment);
		
		    acc += acc_alignment * w_aln;

		    // Cohesion
		    acc_cohesion = acc_cohesion / numFlockmates;
		    acc_cohesion = acc_cohesion - p1;
		    acc_cohesion = normalize(acc_cohesion);

		    acc += acc_cohesion * w_coh;
	    }

        // Step 4. Constrain acceleration
        float accspeed = magnitude(acc);
        if(accspeed > maxchange){
                // set magnitude to MaxChangeInAcc
                acc = acc * (maxchange/accspeed);
        }

        // Step 5. Add acceleration to velocity
        v1 += acc;

        v1.x += MinUrgency;

        // Step 6. Constrain velocity
        float speed = magnitude(v1);
        if(speed > maxspeed){
                // set magnitude to MaxSpeed
                v1 = v1 * (maxspeed/speed);
        }
	 
        //float scale = params.simulation_scale;
        //v.x += h*f.x / scale;
        //v.y += h*f.y / scale;
        //v.z += h*f.z / scale;
	
	    // Step 7. Integrate        
        p1.x += h*v1.x;
        p1.y += h*v1.y;
        p1.z += h*v1.z;
        p1.w = 1.0f; //just in case

	    // Step 8. Check boundary conditions
        if(p1.x >= bndMax.x){
                p1.x = bndMin.x;
        }
        else if(p1.x <= bndMin.x){
                p1.x = bndMax.x;
        }
        else if(p1.y >= bndMax.y){
                p1.y = bndMin.y;
        }
        else if(p1.y <= bndMin.y){
                p1.y = bndMax.y;
        }
        else if(p1.z >= bndMax.z){
                p1.z = bndMin.z;
        }
        else if(p1.z <= bndMin.z){
                p1.z = bndMax.z;
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

        // write values
        velocities[i] = v1;
        positions[i] = p1;
    }

    //printf("v.z %f p.z %f \n", velocities[0].z, positions[0].z);
}

}

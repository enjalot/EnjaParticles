#include "../FLOCK.h"
#include <vector>
using namespace std;
namespace rtps {


//----------------------------------------------------------------------
#include "ge_cpu_euler.cpp"
//----------------------------------------------------------------------
void FLOCK::loadEuler()
{
    printf("create euler kernel\n");

    std::string path(FLOCK_CL_SOURCE_DIR);
    path += "/euler.cl";
    k_euler = Kernel(ps->cli, path, "euler");
  
    int iargs = 0;
    k_euler.setArg(iargs++, cl_sort_indices.getDevicePtr());
    k_euler.setArg(iargs++, cl_vars_unsorted.getDevicePtr());
    k_euler.setArg(iargs++, cl_vars_sorted.getDevicePtr());
    k_euler.setArg(iargs++, cl_position.getDevicePtr());
    k_euler.setArg(iargs++, cl_FLOCKParameters.getDevicePtr());
    k_euler.setArg(iargs++, ps->settings.dt); //time step

	// ONLY IF DEBUGGING
	k_euler.setArg(iargs++, clf_debug.getDevicePtr());
	k_euler.setArg(iargs++, cli_debug.getDevicePtr());



} 

//----------------------------------------------------------------------
void FLOCK::cpuEuler()
{
    #define searchradius 	    .08f
    #define separationdist  	.03f
    #define maxspeed        	1.f     // 1.f
    #define desiredspeed    	.5f    // .5f
    #define maxchange       	0.1f    // .1f
    #define MinUrgency      	0.05f   // .05f
    #define MaxUrgency      	0.1f    // .1f


    float h = ps->settings.dt;

    float4 acc;
    float dist;

    float4 vel_sep, vel_aln, vel_coh;
    float4 acc_separation, acc_alignment, acc_cohesion;

    float w_sep = 0.0f;
    float w_aln = 0.0f;
    float w_coh = 0.0f;

    float4 bndMax = flock_params.grid_max;
    float4 bndMin = flock_params.grid_min;

	float hh = flock_settings.spacing;
	hh *= 2;

    int nb_cells = (int)((bndMax.x-bndMin.x)/hh) * (int)((bndMax.y-bndMin.y)/hh) * (int)((bndMax.z-bndMin.z)/hh);
    vector<int>* flockmates = new vector<int>[nb_cells];

    // loop over all boids
    for(int i = 0; i < num; i++)
    {
	    // initialize acc to zero
        acc = float4(0.f, 0.f, 0.f, 0.f);

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
		        continue;
            }

            float4 d = positions[i] - positions[j];
		    dist = d.length();

            // is boid j a flockmate?
            if(dist <= searchradius){
                (*flockmates).push_back(j);
            }
        }
        
        // Step 3. If they are flockmates, compute the three rules
        if((*flockmates).size() > 0){
		    acc_separation = float4(0.f, 0.f, 0.f, 1.f);
		    acc_alignment  = float4(0.f, 0.f, 0.f, 1.f);
		    acc_cohesion   = float4(0.f, 0.f, 0.f, 1.f);

		    int count = 0;

            // Separation
		    for(int j=0; j < (*flockmates).size(); j++){
                float4 separation =  positions[i] - positions[(*flockmates)[j]];

                // mymese degugging
                #if 0
		            if (separation.x != separation.x || separation.y != separation.y || separation.z != separation.z || separation.w != separation.w) {
                        separation.print("separation: ");
                        printf("boid: %d separation\n", i);
                        exit(0);
                    }
                #endif

			    float d = separation.length();
        	    if(d < separationdist){
                    separation = normalize3(separation);
				    separation = separation / d;
				    acc_separation = acc_separation + separation;
				    count++;
       
                    // mymese degugging
                    #if 0
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

                        #endif
                }
		    }   

		    if(count > 0){
			    acc_separation = acc_separation / count;
			    acc_separation = normalize3(acc_separation);
		    }
		    acc_separation.w = 1.f;

            // mymese degugging
            #if 0
		        if (acc_separation.x != acc_separation.x || acc_separation.y != acc_separation.y || acc_separation.z != acc_separation.z || acc_separation.w != acc_separation.w) {
                    acc_separation.print("acc_separation: ");
                    printf("boid: %d after computing acc_separation\n", i);
                    exit(0);
                }
            #endif

		    // Alignment
		    float4 avg_a = float4(0.f, 0.f, 0.f, 0.f);
		    for(int j=0; j < (*flockmates).size(); j++){
                avg_a = avg_a + positions[(*flockmates)[j]];
	        }
		    avg_a = (*flockmates).size() > 0 ? avg_a / (*flockmates).size() : avg_a;
		    acc_alignment = avg_a - velocities[i];
		    acc_alignment = normalize(acc_alignment); 
	   	    acc_alignment.w = 1.f;

            // mymese degugging
            #if 0
		        if (acc_alignment.x != acc_alignment.x || acc_alignment.y != acc_alignment.y || acc_alignment.z != acc_alignment.z || acc_alignment.w != acc_alignment.w) {
                    acc_alignment.print("acc_alignment: ");
                    printf("boid: %d\n", i);
                    exit(0);
                }
            #endif

		    // Cohesion
            float4 avg_c = float4(0.f, 0.f, 0.f, 0.f);
            for(int j=0; j < (*flockmates).size(); j++){
                avg_c = avg_c + positions[(*flockmates)[j]];
            }   
            avg_c = (*flockmates).size() > 0 ? avg_c / (*flockmates).size() : avg_c;
            acc_cohesion = avg_c - positions[i];
            acc_cohesion = normalize(acc_cohesion);
		    acc_cohesion.w = 1.f;

            // mymese degugging
            #if 0
		        if (acc_cohesion.x != acc_cohesion.x || acc_cohesion.y != acc_cohesion.y || acc_cohesion.z != acc_cohesion.z || acc_cohesion.w != acc_cohesion.w) {
                acc_cohesion.print("acc_cohesion: ");
                printf("boid: %d\n", i);
                exit(0);
            }
            #endif
        
            (*flockmates).clear();
	    }   

        // weight the steers 
	    vel_sep = acc_separation * w_sep;
	    vel_aln = acc_alignment  * w_aln;
	    vel_coh = acc_cohesion   * w_coh;

        // add the steers to previous velocity
	    acc = velocities[i] + vel_sep + vel_aln + vel_coh;

        float  acc_mag  = acc.length();
	    float4 acc_norm = normalize3(acc);
        if(acc_mag > maxchange){
            // set magnitude to MaxChangeInAcc
            acc = acc_norm * maxchange;
        }
       
        // Step 5. Add acceleration to velocity
        velocities[i] = acc;

        // Step 6. Integrate
	    positions[i] = positions[i] + h * velocities[i];
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

#include "../FLOCK.h"
#include <vector>
using namespace std;
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
    #define searchradius 	20.f
    #define separationdist  	30.f
    #define maxspeed        	3.f     // 1.f
    #define desiredspeed    	1.5f    // .5f
    #define maxchange       	0.1f    // .1f
    #define MinUrgency      	0.05f   // .05f
    #define MaxUrgency      	0.1f    // .1f

    //static int count = 0;

    float h = ps->settings.dt;

    printf("enter CPUEuler\n");
    //printf("enter CPUEuler: count= %d\n", count);
    //count++;

    float4 acc;
    float dist;

    float4 vel_sep, vel_aln, vel_coh;
    float4 acc_separation, acc_alignment, acc_cohesion;

    float w_sep = 0.0f;
    float w_aln = 0.0f;
    float w_coh = 0.0f;

    float4 bndMax = params.grid_max;// - params->boundary_distance;
    float4 bndMin = params.grid_min;// + params->boundary_distance;

    int nb_cells = (int)((bndMax.x-bndMin.x)/h) * (int)((bndMax.y-bndMin.y)/h) * (int)((bndMax.z-bndMin.z)/h);
    vector<int>* flockmates = new vector<int>[nb_cells];
//   int MaxFlockmates = num / 2;
//   int flockmates[MaxFlockmates];
//   int numFlockmates = 0;

printf("loop over all boids\n");
printf("nb_cells: %d\n", nb_cells);
printf("flockmates size: %d\n", (*flockmates).size()); 
//exit(0);

    // loop over all boids
    for(int i = 0; i < num; i++)
    {
	printf("boid %d\n", i);
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
        acc = float4(0.f, 0.f, 0.f, 0.f);
        //numFlockmates = 0;

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
             	if(j == i)
			continue;

              	float4 d = positions[i] - positions[j];
		dist = d.length();

                // is boid j a flockmate?
                if(dist <= searchradius){
                    	(*flockmates).push_back(j);
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
		for(int j=0; j < (*flockmates).size(); j++){
                	//pj = positions[flockmates[j]];
			//vj = velocities[flockmates[j]];

        		float4 separation =  positions[i] - positions[j];
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
        		}
		}

		if(count > 0){
			acc_separation = acc_separation / count;
			acc_separation = normalize3(acc_separation);
		}
		//acc_separation += separation;
		acc_separation.w = 1.f;

		// Alignment
		float4 avg_a = float4(0.f, 0.f, 0.f, 0.f);
		for(int j=0; j < (*flockmates).size(); j++){
                	avg_a = avg_a + positions[(*flockmates)[j]];
	    	}
		avg_a = (*flockmates).size() > 0 ? avg_a / (*flockmates).size() : avg_a;
		acc_alignment = avg_a - velocities[i];
		acc_alignment = normalize(acc_alignment); 
		//acc_alignment += v2;
	   	acc_alignment.w = 1.f;

		// Cohesion
                float4 avg_c = float4(0.f, 0.f, 0.f, 0.f);
                for(int j=0; j < (*flockmates).size(); j++){
                        avg_c = avg_c + positions[(*flockmates)[j]];
                }
                avg_c = (*flockmates).size() > 0 ? avg_c / (*flockmates).size() : avg_c;
                acc_cohesion = avg_c - positions[i];
                acc_cohesion = normalize(acc_cohesion);
		//acc_cohesion += p2;
		acc_cohesion.w = 1.f;
//p1.print("p1: ");
//p2.print("p2: ");
//exit(0);
	}   


	vel_sep = acc_separation * w_sep;
	vel_aln = acc_alignment  * w_aln;
	vel_coh = acc_cohesion   * w_coh;


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
acc.print("acceleration: ");


        // Step 4. Constrain acceleration
        float  acc_mag  = acc.length();
	float4 acc_norm = normalize3(acc);
 
        if(acc_mag > maxchange){
                // set magnitude to MaxChangeInAcc
                acc = acc_norm * maxchange;
        }


        // Step 5. Add acceleration to velocity
        velocities[i] = acc;

	// remove for debugging
        //v1.x += MinUrgency;

#if 0
        // Step 6. Constrain velocity
        float speed = magnitude3(v1);
        if(speed > maxspeed){
                // set magnitude to MaxSpeed
                v1 = v1 * (maxspeed/speed);
        }
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

#if 0
		if (p1.x != p1.x || p1.y != p1.y || p1.z != p1.z || p1.w != p1.w) {
			printf("h= %f\n", (float) h);
			v1.print("v1");
		}
#endif
    	// Step 8. Check boundary conditions
        if(positions[i].x >= bndMax.x){
                positions[i].x = bndMin.x;
        }
        else if(positions[i].x <= bndMin.x){
                positions[i].x = bndMax.x;
        }
        else if(positions[i].y >= bndMax.y){
                positions[i].y = bndMin.y;
        }
        else if(positions[i].y <= bndMin.y){
                positions[i].y = bndMax.y;
        }
        else if(positions[i].z >= bndMax.z){
                positions[i].z = bndMin.z;
        }
        else if(positions[i].z <= bndMin.z){
                positions[i].z = bndMax.z;
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
    }

    //printf("v.z %f p.z %f \n", velocities[0].z, positions[0].z);
}

}

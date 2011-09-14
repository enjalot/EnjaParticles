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


#include <FLOCK.h>
#include<math.h>

#include<time.h>

namespace rtps 
{
    //----------------------------------------------------------------------
    Rules::Rules(std::string wpath, CL* cli_, EB::Timer* timer_)
    {
        cli = cli_;
        timer = timer_;
        std::string path;

        printf("create rules kernel\n");

        // rules 
        try
        {
            path = wpath + "/rules.cl";
            k_rules= Kernel(cli, path, "rules");
        }
        catch (cl::Error er)
        {
            printf("ERROR(rules): %s(%s)\n", er.what(), oclErrorString(er.err()));
        }
    }

    //----------------------------------------------------------------------
    void Rules::execute(int num,
                    //input
                    float4 target,
                    Buffer<float4>& pos_s, 
                    Buffer<float4>& vel_s, 
                    Buffer<int4>& neigh_s, 
                    Buffer<float4>& sep_s, 
                    Buffer<float4>& align_s, 
                    Buffer<float4>& coh_s, 
                    Buffer<float4>& goal_s, 
                    Buffer<float4>& avoid_s, 
                    //output
                    Buffer<unsigned int>& ci_start,
                    Buffer<unsigned int>& ci_end,
                    //params
                    Buffer<GridParams>& gp,
                    Buffer<FLOCKParameters>& flockp,
                    //debug params
                    Buffer<float4>& clf_debug,
                    Buffer<int4>& cli_debug)
    { 
        int iarg = 0;
        k_rules.setArg(iarg++, target); 
        k_rules.setArg(iarg++, pos_s.getDevicePtr());
        k_rules.setArg(iarg++, vel_s.getDevicePtr());
        k_rules.setArg(iarg++, neigh_s.getDevicePtr());
        k_rules.setArg(iarg++, sep_s.getDevicePtr());
        k_rules.setArg(iarg++, align_s.getDevicePtr());
        k_rules.setArg(iarg++, coh_s.getDevicePtr());
        k_rules.setArg(iarg++, goal_s.getDevicePtr());
        k_rules.setArg(iarg++, avoid_s.getDevicePtr());
        k_rules.setArg(iarg++, ci_start.getDevicePtr());
        k_rules.setArg(iarg++, ci_end.getDevicePtr());
        k_rules.setArg(iarg++, gp.getDevicePtr());
        k_rules.setArg(iarg++, flockp.getDevicePtr());

        // ONLY IF DEBUGGING
        k_rules.setArg(iarg++, clf_debug.getDevicePtr());
        k_rules.setArg(iarg++, cli_debug.getDevicePtr());

        int local = 64;
        try
        {
            float gputime = k_rules.execute(num, local);
            if(gputime > 0)
                timer->set(gputime);

        }

        catch (cl::Error er)
        {
            printf("ERROR(rules): %s(%s)\n", er.what(), oclErrorString(er.err()));
        }
    }

    //----------------------------------------------------------------------
    void FLOCK::cpuRules()
    {
        
	    float spacing1 = spacing;

        float4 bndMax = grid_params.bnd_max;
        float4 bndMin = grid_params.bnd_min;
        
        int nb_cells = (int)((bndMax.x-bndMin.x)/spacing1) * (int)((bndMax.y-bndMin.y)/spacing1) * (int)((bndMax.z-bndMin.z)/spacing1);

        vector<int> flockmates;
        flockmates.resize(nb_cells);
        
        float4 pi, pj, pt;
        float4 vi, vj;
        
        int numFlockmates;

        //seed random
        srand ( time(NULL) );
        
        // Step 1. Loop over all boids
        for(int i = 0; i < num; i++)
        {
            pi = positions[i]; // * flock_params.simulation_scale;
            vi = velocities[i];

            numFlockmates = 0;
            (flockmates).clear();

            // Step 2. Search for neighbors
            for(int j=0; j < num; j++){
                
                pj = positions[j]; // * flock_params.simulation_scale;
            
                if(j == i){
			        continue;
                }
                float4 d = pi - pj; 
		        float dist = d.length();

                // is boid j a flockmate?
                if(dist <= flock_params.search_radius){
                    (flockmates).push_back(j);
                    numFlockmates++;
                }
            }   

            // Step 3. Compute the Rules    
            if(flock_params.w_sep > 0.f)
            {
		        int nearestFlockmates = 0;

                // 3.1 Separation
		        for(int j=0; j < numFlockmates; j++){
                    
                    pj = positions[(flockmates)[j]]; // * flock_params.simulation_scale;
                    
                    float4 s = pi - pj; 
                    float dist = s.length();
                
                    if(dist <= flock_params.min_dist){
                        //s = normalize3(s);
                        s /= dist;
				        separation[i] += s;
				        nearestFlockmates++;
        		    }   
		        }

		        if(nearestFlockmates > 0){
			        separation[i] /= nearestFlockmates;
			        //separation[i] = normalize3(separation[i]);
		        }
            }

            if(flock_params.w_align > 0.f)
            {
                // 3.2 Alignment
		        for(int j=0; j < numFlockmates; j++){
                    vj = velocities[(flockmates)[j]];
                    alignment[i] += vj; 
	    	    }   
		        
                if(numFlockmates > 0){
                    alignment[i] /= numFlockmates;
		            alignment[i] -= vi; 
		            //alignment[i] = normalize3(alignment[i]); 
                }
           
            }

            if(flock_params.w_coh > 0.f)
            {
                // 3.3 Cohesion
                for(int j=0; j < numFlockmates; j++){
                    pj = positions[(flockmates)[j]]; // * flock_params.simulation_scale;
                    
                    cohesion[i] += pj; 
                }

                if(numFlockmates > 0){ 
                    cohesion[i] /= numFlockmates;
                    cohesion[i] -= pi; 
                    //cohesion[i] = normalize3(cohesion[i]);
                }
	        }
            
            if(flock_params.w_goal > 0.f)
            {
                pt = ps->settings->target;//float4(3.f, 1.f, 4.f, 0.f);//flock_params.target;
                float4 dist = normalize3(pt - pi);
                float4 desiredVel = dist * flock_params.max_speed;
                goal[i] = desiredVel - vi;

            }
            
            if(flock_params.w_avoid > 0.f)
            {
                pt = ps->settings->target;//float4(3.f, 2.f, 2.f, 0.f);//flock_params.target;
                float4 dist = normalize3(pt - pi);
                float4 desiredVel = dist * flock_params.max_speed;
                avoid[i] = -desiredVel - vi;
            }

            if(flock_params.w_wander > 0.f){
                float r1 = rand();
                float r2 = rand();
                float r3 = rand();

                float4 rand;
                rand =  float4(r1, r2, r3, 0.f);
                float4 desiredVel = normalize3(rand);
                wander[i] = desiredVel * flock_params.max_speed;
            }

        }
    }
} 

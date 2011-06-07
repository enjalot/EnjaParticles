#include <FLOCK.h>
#include<math.h>

namespace rtps 
{
    //----------------------------------------------------------------------
    Rules::Rules(std::string wpath, CL* cli_, EB::Timer* timer_)
    {
        cli = cli_;
        timer = timer_;
        std::string path;

        printf("create rules kernel\n");

        // separation
        try
        {
            path = wpath + "/rules.cl";
            k_rules= Kernel(cli, path, "rules");
        }
        catch (cl::Error er)
        {
            printf("ERROR(rules): %s(%s)\n", er.what(), oclErrorString(er.err()));
        }
#if 0
        // flockmates 
        try
        {
            path = wpath + "/flockmates.cl";
            k_flockmates= Kernel(cli, path, "flockmates");
        }
        catch (cl::Error er)
        {
            printf("ERROR(flockmates): %s(%s)\n", er.what(), oclErrorString(er.err()));
        }

        // separation
        try
        {
            path = wpath + "/rule_separation.cl";
            k_rule_separation= Kernel(cli, path, "rule_separation");
        }
        catch (cl::Error er)
        {
            printf("ERROR(rule_separation): %s(%s)\n", er.what(), oclErrorString(er.err()));
        }

        // alignment
        try
        {
            path = wpath + "/rule_alignment.cl";
            k_rule_alignment= Kernel(cli, path, "rule_alignment");
        }
        catch (cl::Error er)
        {
            printf("ERROR(rule_alignment): %s(%s)\n", er.what(), oclErrorString(er.err()));
        }

        // cohesion
        try
        {
            path = wpath + "/rule_cohesion.cl";
            k_rule_cohesion= Kernel(cli, path, "rule_cohesion");
        }
        catch (cl::Error er)
        {
            printf("ERROR(rule_cohesion): %s(%s)\n", er.what(), oclErrorString(er.err()));
        }

        // leader following 
        try
        {
            path = wpath + "/rule_leaderfollowing.cl";
            k_rule_leaderfollowing= Kernel(cli, path, "rule_leaderfollowing");
        }
        catch (cl::Error er)
        {
            printf("ERROR(rule_leaderfollowing): %s(%s)\n", er.what(), oclErrorString(er.err()));
        }
#endif
    }

    //----------------------------------------------------------------------
    void Rules::execute(int num,
                    //input
                    Buffer<float4>& pos_s, 
                    Buffer<float4>& vel_s, 
                    Buffer<int4>& neigh_s, 
                    Buffer<float4>& sep_s, 
                    Buffer<float4>& align_s, 
                    Buffer<float4>& coh_s, 
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
        k_rules.setArg(iarg++, pos_s.getDevicePtr());
        k_rules.setArg(iarg++, vel_s.getDevicePtr());
        k_rules.setArg(iarg++, neigh_s.getDevicePtr());
        k_rules.setArg(iarg++, sep_s.getDevicePtr());
        k_rules.setArg(iarg++, align_s.getDevicePtr());
        k_rules.setArg(iarg++, coh_s.getDevicePtr());
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

#if 0
    //----------------------------------------------------------------------
    void Rules::executeSeparation(int num,
                    //input
                    Buffer<float4>& pos_s,
                    Buffer<float4>& sep_s,
                    Buffer<int4>& neigh_s, 
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
        k_rule_separation.setArg(iarg++, pos_s.getDevicePtr());
        k_rule_separation.setArg(iarg++, sep_s.getDevicePtr());
        k_rule_separation.setArg(iarg++, neigh_s.getDevicePtr());
        k_rule_separation.setArg(iarg++, ci_start.getDevicePtr());
        k_rule_separation.setArg(iarg++, ci_end.getDevicePtr());
        k_rule_separation.setArg(iarg++, gp.getDevicePtr());
        k_rule_separation.setArg(iarg++, flockp.getDevicePtr());

        // ONLY IF DEBUGGING
        k_rule_separation.setArg(iarg++, clf_debug.getDevicePtr());
        k_rule_separation.setArg(iarg++, cli_debug.getDevicePtr());

        int local = 64;
        try
        {
            float gputime = k_rule_separation.execute(num, local);
            if(gputime > 0)
                timer->set(gputime);

        }

        catch (cl::Error er)
        {
            printf("ERROR(rule_separation): %s(%s)\n", er.what(), oclErrorString(er.err()));
        }
    }

    //----------------------------------------------------------------------
    void Rules::executeAlignment(int num,
                    //input
                    Buffer<float4>& pos_s, 
                    Buffer<float4>& vel_s,
                    Buffer<float4>& align_s, 
                    Buffer<int4>& neigh_s, 
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
        k_rule_alignment.setArg(iarg++, pos_s.getDevicePtr());
        k_rule_alignment.setArg(iarg++, vel_s.getDevicePtr());
        k_rule_alignment.setArg(iarg++, align_s.getDevicePtr());
        k_rule_alignment.setArg(iarg++, neigh_s.getDevicePtr());
        k_rule_alignment.setArg(iarg++, ci_start.getDevicePtr());
        k_rule_alignment.setArg(iarg++, ci_end.getDevicePtr());
        k_rule_alignment.setArg(iarg++, gp.getDevicePtr());
        k_rule_alignment.setArg(iarg++, flockp.getDevicePtr());

        // ONLY IF DEBUGGING
        k_rule_alignment.setArg(iarg++, clf_debug.getDevicePtr());
        k_rule_alignment.setArg(iarg++, cli_debug.getDevicePtr());

        int local = 64;
        try
        {
            float gputime = k_rule_alignment.execute(num, local);
            if(gputime > 0)
                timer->set(gputime);

        }

        catch (cl::Error er)
        {
            printf("ERROR(rule_alignment): %s(%s)\n", er.what(), oclErrorString(er.err()));
        }
    }

    //----------------------------------------------------------------------
    void Rules::executeCohesion(int num,
                    //input
                    Buffer<float4>& pos_s,
                    Buffer<float4>& coh_s, 
                    Buffer<int4>& neigh_s, 
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
        k_rule_cohesion.setArg(iarg++, pos_s.getDevicePtr());
        k_rule_cohesion.setArg(iarg++, coh_s.getDevicePtr());
        k_rule_cohesion.setArg(iarg++, neigh_s.getDevicePtr());
        k_rule_cohesion.setArg(iarg++, ci_start.getDevicePtr());
        k_rule_cohesion.setArg(iarg++, ci_end.getDevicePtr());
        k_rule_cohesion.setArg(iarg++, gp.getDevicePtr());
        k_rule_cohesion.setArg(iarg++, flockp.getDevicePtr());

        // ONLY IF DEBUGGING
        k_rule_cohesion.setArg(iarg++, clf_debug.getDevicePtr());
        k_rule_cohesion.setArg(iarg++, cli_debug.getDevicePtr());

        int local = 64;
        try
        {
            float gputime = k_rule_cohesion.execute(num, local);
            if(gputime > 0)
                timer->set(gputime);

        }

        catch (cl::Error er)
        {
            printf("ERROR(rule_cohesion): %s(%s)\n", er.what(), oclErrorString(er.err()));
        }
    }

    //----------------------------------------------------------------------
    void Rules::executeLeaderFollowing(int num,
                    //input
                    Buffer<float4>& pos_s,
                    Buffer<float4>& vel_s,
                    Buffer<float4>& leadfoll_s,
                    Buffer<int4>& neigh_s, 
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
        k_rule_leaderfollowing.setArg(iarg++, pos_s.getDevicePtr());
        k_rule_leaderfollowing.setArg(iarg++, vel_s.getDevicePtr());
        k_rule_leaderfollowing.setArg(iarg++, leadfoll_s.getDevicePtr());
        k_rule_leaderfollowing.setArg(iarg++, neigh_s.getDevicePtr());
        k_rule_leaderfollowing.setArg(iarg++, ci_start.getDevicePtr());
        k_rule_leaderfollowing.setArg(iarg++, ci_end.getDevicePtr());
        k_rule_leaderfollowing.setArg(iarg++, gp.getDevicePtr());
        k_rule_leaderfollowing.setArg(iarg++, flockp.getDevicePtr());

        // ONLY IF DEBUGGING
        k_rule_leaderfollowing.setArg(iarg++, clf_debug.getDevicePtr());
        k_rule_leaderfollowing.setArg(iarg++, cli_debug.getDevicePtr());

        int local = 64;
        try
        {
            float gputime = k_rule_leaderfollowing.execute(num, local);
            if(gputime > 0)
                timer->set(gputime);

        }

        catch (cl::Error er)
        {
            printf("ERROR(rule_leaderfollowing): %s(%s)\n", er.what(), oclErrorString(er.err()));
        }
    }
#endif
    void FLOCK::cpuRules()
    {
        
	    float spacing1 = spacing;

        float4 bndMax = grid_params.bnd_max;
        float4 bndMin = grid_params.bnd_min;
        
        int nb_cells = (int)((bndMax.x-bndMin.x)/spacing1) * (int)((bndMax.y-bndMin.y)/spacing1) * (int)((bndMax.z-bndMin.z)/spacing1);
       
        //printf("*** nb_cells=%d***\n",nb_cells);

        vector<int> flockmates;
        flockmates.resize(nb_cells);
        
        // Step 1. Loop over all boids
        for(int i = 0; i < num; i++)
        {

            // Step 2. Search for neighbors
            for(int j=0; j < num; j++){
                if(j == i){
			        continue;
                }
                float4 d = positions[i] - positions[j];
		        float dist = d.length();

                // is boid j a flockmate?
                if(dist <= flock_params.search_radius){
                    (flockmates).push_back(j);
                }
            }   

            // Step 3. Compute the Rules    
            if(flock_params.w_sep > 0.f)
            {
		        int nearestFlockmates = 0;

                // 3.1 Separation
		        for(int j=0; j < (flockmates).size(); j++){
                    float4 s =  positions[i] - positions[(flockmates)[j]];
                    float dist = s.length();
        		
                    if(dist < flock_params.min_dist){
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
		        for(int j=0; j < (flockmates).size(); j++){
                    alignment[i] += velocities[(flockmates)[j]];
	    	    }   
		        
                if((flockmates).size() > 0)
                    alignment[i] /= (flockmates).size();

		        alignment[i] -= velocities[i];
		        //alignment[i] = normalize3(alignment[i]); 

            }

            if(flock_params.w_coh > 0.f)
            {
                // 3.3 Cohesion
                for(int j=0; j < (flockmates).size(); j++){
                    cohesion[i] += positions[(flockmates)[j]];
                }

                if((flockmates).size() > 0) 
                    cohesion[i] /= (flockmates).size();

                cohesion[i] -= positions[i];
                //cohesion[i] = normalize3(cohesion[i]);

                (flockmates).clear();
	        }
        }

    }
} 

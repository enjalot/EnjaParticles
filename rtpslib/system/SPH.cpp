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



#include <GL/glew.h>
#include <math.h>
#include <sstream>
#include <iomanip>
#include <string>

#include "System.h"
#include "SPH.h"
//#include "../domain/UniformGrid.h"
#include "Domain.h"
#include "IV.h"

#include "common/Hose.h"



//for random
#include<time.h>

namespace rtps
{
    using namespace sph;

	//----------------------------------------------------------------------
    SPH::SPH(RTPS *psfr, int n, int max_nb_in_cloud)
    {
        //store the particle system framework
        ps = psfr;
        settings = ps->settings;
        max_num = n;
		//printf("max_num= %d\n", max_num); exit(0);
        num = 0;
        nb_var = 10;

		// I should be able to not specify this, but GPU restrictions ...

        resource_path = settings->GetSettingAs<string>("rtps_path");
        printf("resource path: %s\n", resource_path.c_str());

        //seed random
        srand ( time(NULL) );

        grid = settings->grid;

        std::vector<SPHParams> vparams(0);
        vparams.push_back(sphp);
        cl_sphp = Buffer<SPHParams>(ps->cli, vparams);

        calculate();
        updateSPHP();

        //settings->printSettings();

        spacing = settings->GetSettingAs<float>("Spacing");

        //SPH settings depend on number of particles used
        //calculateSPHSettings();
        //set up the grid
        setupDomain();

        integrator = LEAPFROG;
        //integrator = EULER;


        //*** end Initialization

        setupTimers();

#ifdef CPU
        printf("RUNNING ON THE CPU\n");
#endif
#ifdef GPU
        printf("RUNNING ON THE GPU\n");

        
        //setup the sorted and unsorted arrays
        prepareSorted();
        setRenderer();

        ps->cli->addIncludeDir(sph_source_dir);
        ps->cli->addIncludeDir(common_source_dir);

#ifdef CLOUD_COLLISION
		// CLOUD INITIALIZATION
		cloudInitialize();
#endif



        //should be more cross platform
        sph_source_dir = resource_path + "/" + std::string(SPH_CL_SOURCE_DIR);
        common_source_dir = resource_path + "/" + std::string(COMMON_CL_SOURCE_DIR);

        hash = Hash(common_source_dir, ps->cli, timers["hash_gpu"]);
        bitonic = Bitonic<unsigned int>(common_source_dir, ps->cli );
        radix = Radix<unsigned int>(common_source_dir, ps->cli, max_num, 128);
        cellindices = CellIndices(common_source_dir, ps->cli, timers["ci_gpu"] );
        permute = Permute( common_source_dir, ps->cli, timers["perm_gpu"] );

        density = Density(sph_source_dir, ps->cli, timers["density_gpu"]);
        force = Force(sph_source_dir, ps->cli, timers["force_gpu"]);
        collision_wall = CollisionWall(sph_source_dir, ps->cli, timers["cw_gpu"]);
        collision_tri = CollisionTriangle(sph_source_dir, ps->cli, timers["ct_gpu"], 2048); //TODO expose max_triangles as a parameter
		

        //could generalize this to other integration methods later (leap frog, RK4)
        if (integrator == LEAPFROG)
        {
            //loadLeapFrog();
            leapfrog = LeapFrog(sph_source_dir, ps->cli, timers["leapfrog_gpu"]);
        }
        else if (integrator == EULER)
        {
            //loadEuler();
            euler = Euler(sph_source_dir, ps->cli, timers["euler_gpu"]);
        }

		// CLOUD Integrator
		// ADD Cloud timers later. 
        string lt_file = settings->GetSettingAs<string>("lt_cl");
        //lifetime = Lifetime(sph_source_dir, ps->cli, timers["lifetime_gpu"], lt_file);
#endif

    }

	//----------------------------------------------------------------------
    SPH::~SPH()
    {
        printf("SPH destructor\n");
        if (pos_vbo && managed)
        {
            glBindBuffer(1, pos_vbo);
            glDeleteBuffers(1, (GLuint*)&pos_vbo);
            pos_vbo = 0;
        }
        if (col_vbo && managed)
        {
            glBindBuffer(1, col_vbo);
            glDeleteBuffers(1, (GLuint*)&col_vbo);
            col_vbo = 0;
        }

        Hose* hose;
        int hs = hoses.size();  
        for(int i = 0; i < hs; i++)
        {
            hose = hoses[i];
            delete hose;

        }
    }

	//----------------------------------------------------------------------
    void SPH::update()
    {
		//printf("+++++++++++++ enter UPDATE()\n");
        //call kernels
        //TODO: add timings
#ifdef CPU
        updateCPU();
#endif
#ifdef GPU
        updateGPU();
#endif
    }

	//----------------------------------------------------------------------
    void SPH::updateCPU()
    {
        cpuDensity();
        cpuPressure();
        cpuViscosity();
        cpuXSPH();
        cpuCollision_wall();

        if (integrator == EULER)
        {
            cpuEuler();
        }
        else if (integrator == LEAPFROG)
        {
            cpuLeapFrog();
        }
        glBindBuffer(GL_ARRAY_BUFFER, pos_vbo);
        glBufferData(GL_ARRAY_BUFFER, num * sizeof(float4), &positions[0], GL_DYNAMIC_DRAW);
    }

	//----------------------------------------------------------------------
    void SPH::updateGPU()
    {
		//printf("**** enter updateGPU, num= %d\n", num);

        timers["update"]->start();
        glFinish();
        if (settings->has_changed()) updateSPHP();

        //settings->printSettings();

        //int sub_intervals = 3;  //should be a setting
        int sub_intervals =  settings->GetSettingAs<float>("sub_intervals");
        //this should go in the loop but avoiding acquiring and releasing each sub
        //interval for all the other calls.
        //this does end up acquire/release everytime sprayHoses calls pushparticles
        //should just do try/except?
        for (int i=0; i < sub_intervals; i++)
        {
            sprayHoses();
        }

        cl_position_u.acquire();
        cl_color_u.acquire();

        for (int i=0; i < sub_intervals; i++)
        {
            //if(num >0) printf("before hash and sort\n");
            hash_and_sort();

			//------------------
            //printf("before cellindices, num= %d\n", num);
            timers["cellindices"]->start();
            int nc = cellindices.execute(   num,
                cl_sort_hashes,
                cl_sort_indices,
                cl_cell_indices_start,
                cl_cell_indices_end,
                //cl_sphp,
                cl_GridParams,
                grid_params.nb_cells,
                clf_debug,
                cli_debug);
            timers["cellindices"]->stop();

			//if (num > 0) exit(1); //GE

       
			//-----------------
            //printf("*** enter fluid permute, num= %d\n", num);
            timers["permute"]->start();
            permute.execute(   num,
                cl_position_u,
                cl_position_s,
                cl_velocity_u,
                cl_velocity_s,
                cl_veleval_u,
                cl_veleval_s,
                cl_color_u,
                cl_color_s,
                cl_sort_indices,
                //cl_sphp,
                cl_GridParams,
                clf_debug,
                cli_debug);
            timers["permute"]->stop();
			//printf("exit after fluid permute\n");
			//if (num > 0) exit(0);
 
			// NUMBER OF CLOUD PARTICLES IS CONSTANT THROUGHOUT THE SIMULATION

			//---------------------
            if (nc <= num && nc >= 0)
            {
                //check if the number of particles has changed
                //(this happens when particles go out of bounds,
                //  either because of forces or by explicitly placing
                //  them in order to delete)
                //
                //if so we need to copy sorted into unsorted
                //and redo hash_and_sort
                printf("SOME PARTICLES WERE DELETED!\n");
                printf("nc: %d num: %d\n", nc, num);

                deleted_pos.resize(num-nc);
                deleted_vel.resize(num-nc);
                //The deleted particles should be the nc particles after num
                cl_position_s.copyToHost(deleted_pos, nc); //damn these will always be out of bounds here!
                cl_velocity_s.copyToHost(deleted_vel, nc);

 
                num = nc;
                settings->SetSetting("Number of Particles", num);
                //sphp.num = num;
                updateSPHP();
                renderer->setNum(sphp.num);

                //need to copy sorted arrays into unsorted arrays
//**** PREP(2)
                call_prep(2);
                //printf("HOW MANY NOW? %d\n", num);
                hash_and_sort();
                                //we've changed num and copied sorted to unsorted. skip this iteration and do next one
                //this doesn't work because sorted force etc. are having an effect?
                //continue; 
            }


			//-------------------------------------
            //if(num >0) printf("density\n");
            timers["density"]->start();
            density.execute(   num,
                //cl_vars_sorted,
                cl_position_s,
                cl_density_s,
                cl_cell_indices_start,
                cl_cell_indices_end,
                cl_sphp,
                cl_GridParamsScaled, // GE: Might have to fix this. Do not know. 
                //cl_GridParams,
                clf_debug,
                cli_debug);
            timers["density"]->stop();
            
			//-------------------------------------
            //if(num >0) printf("force\n");
            timers["force"]->start();
            force.execute(   num,
                //cl_vars_sorted,
                cl_position_s,
                cl_density_s,
                cl_veleval_s,
                cl_force_s,
                cl_xsph_s,
                cl_cell_indices_start,
                cl_cell_indices_end,
                cl_sphp,
                //cl_GridParams,
                cl_GridParamsScaled,
                clf_debug,
                cli_debug);

            timers["force"]->stop();

            collision();

#ifdef CLOUD_COLLISION
			cloudUpdate();
#endif
            integrate(); // includes boundary force
        }

#ifdef CLOUD_COLLISION
		cloudCleanup();
#endif

        cl_position_u.release();
        cl_color_u.release();

        timers["update"]->stop();
    }

	//----------------------------------------------------------------------
    void SPH::hash_and_sort()
    {
        //printf("hash\n");
        timers["hash"]->start();
        hash.execute(   num,
                //cl_vars_unsorted,
                cl_position_u,
                cl_sort_hashes,
                cl_sort_indices,
                //cl_sphp,
                cl_GridParams,
                clf_debug,
                cli_debug);
        timers["hash"]->stop();

        //printf("bitonic_sort\n");
        //defined in Sort.cpp
        timers["bitonic"]->start();
        bitonic_sort();
        timers["bitonic"]->stop();
        //timers["radix"]->start();
        //timers["bitonic"]->start();
        //radix_sort();
        //timers["bitonic"]->stop();
        //timers["radix"]->stop();
    }

	//----------------------------------------------------------------------
    void SPH::collision()
    {
        //when implemented other collision routines can be chosen here
        timers["collision_wall"]->start();
        //collide_wall();
        collision_wall.execute(num,
                //cl_vars_sorted, 
                cl_position_s,
                cl_velocity_s,
                cl_force_s,
                cl_sphp,
                //cl_GridParams,
                cl_GridParamsScaled,
                //debug
                clf_debug,
                cli_debug);

        //k_collision_wall.execute(num, local_size);
        timers["collision_wall"]->stop();

        timers["collision_tri"]->start();
        //collide_triangles();
        collision_tri.execute(num,
                settings->dt,
                //cl_vars_sorted, 
                cl_position_s,
                cl_velocity_s,
                cl_force_s,
                cl_sphp,
                //debug
                clf_debug,
                cli_debug);
        timers["collision_tri"]->stop();
    }
	//----------------------------------------------------------------------

    void SPH::integrate()
    {
        timers["integrate"]->start();

        if (integrator == EULER)
        {
            //euler();
            euler.execute(num,
                settings->dt,
                cl_position_u,
                cl_position_s,
                cl_velocity_u,
                cl_velocity_s,
                cl_force_s,
                //cl_vars_unsorted, 
                //cl_vars_sorted, 
                cl_sort_indices,
                cl_sphp,
                //debug
                clf_debug,
                cli_debug);


        }
        else if (integrator == LEAPFROG)
        {
            //leapfrog();
             leapfrog.execute(num,
                settings->dt,
                cl_position_u,
                cl_position_s,
                cl_velocity_u,
                cl_velocity_s,
                cl_veleval_u,
                cl_force_s,
                cl_xsph_s,
                //cl_vars_unsorted, 
                //cl_vars_sorted, 
                cl_sort_indices,
                cl_sphp,
                //debug
                clf_debug,
                cli_debug);
        }

		// Perhaps I am messed up by Courant condition if cloud point 
		// velocities are too large? 

		static int count=0;

    	timers["integrate"]->stop();
    }

	// GE: WHY IS THIS NEEDED?
	//----------------------------------------------------------------------
    void SPH::call_prep(int stage)
    {
		// copy from sorted to unsorted arrays at the beginning of each 
		// iteration
		// copy from cl_position_s to cl_position_u
		// Only called if number of fluid particles changes from one iteration
		// to the other

            cl_position_u.copyFromBuffer(cl_position_s, 0, 0, num);
            cl_velocity_u.copyFromBuffer(cl_velocity_s, 0, 0, num);
            cl_veleval_u.copyFromBuffer(cl_veleval_s, 0, 0, num);
            cl_color_u.copyFromBuffer(cl_color_s, 0, 0, num);
    }

	//----------------------------------------------------------------------
    int SPH::setupTimers()
    {
        //int print_freq = 20000;
        int print_freq = 1000; //one second
        int time_offset = 5;
        timers["update"] = new EB::Timer("Update loop", time_offset);
        timers["hash"] = new EB::Timer("Hash function", time_offset);
        timers["hash_gpu"] = new EB::Timer("Hash GPU kernel execution", time_offset);
        timers["cellindices"] = new EB::Timer("CellIndices function", time_offset);
        timers["ci_gpu"] = new EB::Timer("CellIndices GPU kernel execution", time_offset);
        timers["permute"] = new EB::Timer("Permute function", time_offset);
        timers["cloud_permute"] = new EB::Timer("CloudPermute function", time_offset);
        timers["perm_gpu"] = new EB::Timer("Permute GPU kernel execution", time_offset);
        timers["ds_gpu"] = new EB::Timer("DataStructures GPU kernel execution", time_offset);
        timers["bitonic"] = new EB::Timer("Bitonic Sort function", time_offset);
        timers["density"] = new EB::Timer("Density function", time_offset);
        timers["density_gpu"] = new EB::Timer("Density GPU kernel execution", time_offset);
        timers["force"] = new EB::Timer("Force function", time_offset);
        timers["force_gpu"] = new EB::Timer("Force GPU kernel execution", time_offset);
        timers["collision_wall"] = new EB::Timer("Collision wall function", time_offset);
        timers["cw_gpu"] = new EB::Timer("Collision Wall GPU kernel execution", time_offset);
        timers["collision_tri"] = new EB::Timer("Collision triangles function", time_offset);
        timers["ct_gpu"] = new EB::Timer("Collision Triangle GPU kernel execution", time_offset);
        timers["collision_cloud"] = new EB::Timer("Collision cloud function", time_offset);
        timers["integrate"] = new EB::Timer("Integration function", time_offset);
        timers["leapfrog_gpu"] = new EB::Timer("LeapFrog Integration GPU kernel execution", time_offset);
        timers["euler_gpu"] = new EB::Timer("Euler Integration GPU kernel execution", time_offset);
        //timers["lifetime_gpu"] = new EB::Timer("Lifetime GPU kernel execution", time_offset);
        //timers["prep_gpu"] = new EB::Timer("Prep GPU kernel execution", time_offset);
		return 0;
    }

	//----------------------------------------------------------------------
    void SPH::printTimers()
    {
        printf("Number of Particles: %d\n", num);
        timers.printAll();
        std::ostringstream oss; 
        oss << "sph_timer_log_" << std::setw( 7 ) << std::setfill( '0' ) <<  num; 
        //printf("oss: %s\n", (oss.str()).c_str());

        timers.writeToFile(oss.str()); 
    }

	//----------------------------------------------------------------------
    void SPH::prepareSorted()
    {
//#include "sph/cl_src/cl_macros.h"

        positions.resize(max_num);
        colors.resize(max_num);
        forces.resize(max_num);
        velocities.resize(max_num);
        veleval.resize(max_num);
        densities.resize(max_num);
        xsphs.resize(max_num);

        //for reading back different values from the kernel
        std::vector<float4> error_check(max_num);
        
        float4 pmax = grid_params.grid_max + grid_params.grid_size;
        //std::fill(positions.begin(), positions.end(), pmax);

        //float4 color = float4(0.0, 1.0, 0.0, 1.0f);
        //std::fill(colors.begin(), colors.end(),color);
        std::fill(forces.begin(), forces.end(), float4(0.0f, 0.0f, 1.0f, 0.0f));
        std::fill(velocities.begin(), velocities.end(), float4(0.0f, 0.0f, 0.0f, 0.0f));
        std::fill(veleval.begin(), veleval.end(), float4(0.0f, 0.0f, 0.0f, 0.0f));

        std::fill(densities.begin(), densities.end(), 0.0f);
        std::fill(xsphs.begin(), xsphs.end(), float4(0.0f, 0.0f, 0.0f, 0.0f));
        std::fill(error_check.begin(), error_check.end(), float4(0.0f, 0.0f, 0.0f, 0.0f));

        // VBO creation, TODO: should be abstracted to another class
        managed = true;
        printf("positions: %zd, %zd, %zd\n", positions.size(), sizeof(float4), positions.size()*sizeof(float4));
        pos_vbo = createVBO(&positions[0], positions.size()*sizeof(float4), GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW);
        printf("pos vbo: %d\n", pos_vbo);
        col_vbo = createVBO(&colors[0], colors.size()*sizeof(float4), GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW);
        printf("col vbo: %d\n", col_vbo);
        // end VBO creation

        //vbo buffers
        cl_position_u = Buffer<float4>(ps->cli, pos_vbo);
        cl_position_s = Buffer<float4>(ps->cli, positions);
        cl_color_u = Buffer<float4>(ps->cli, col_vbo);
        cl_color_s = Buffer<float4>(ps->cli, colors);

        //pure opencl buffers: these are deprecated
        cl_velocity_u = Buffer<float4>(ps->cli, velocities);
        cl_velocity_s = Buffer<float4>(ps->cli, velocities);
        cl_veleval_u = Buffer<float4>(ps->cli, veleval);
        cl_veleval_s = Buffer<float4>(ps->cli, veleval);
        cl_density_s = Buffer<float>(ps->cli, densities);
        cl_force_s = Buffer<float4>(ps->cli, forces);
        cl_xsph_s = Buffer<float4>(ps->cli, xsphs);

        //cl_error_check= Buffer<float4>(ps->cli, error_check);

        
        //TODO make a helper constructor for buffer to make a cl_mem from a struct
        //Setup Grid Parameter structs
        std::vector<GridParams> gparams(0);
        gparams.push_back(grid_params);
        cl_GridParams = Buffer<GridParams>(ps->cli, gparams);

        //scaled Grid Parameters
        std::vector<GridParams> sgparams(0);
        sgparams.push_back(grid_params_scaled);
        cl_GridParamsScaled = Buffer<GridParams>(ps->cli, sgparams);


        //setup debug arrays
        std::vector<float4> clfv(max_num);
        std::fill(clfv.begin(), clfv.end(),float4(0.0f, 0.0f, 0.0f, 0.0f));
        std::vector<int4> cliv(max_num);
        std::fill(cliv.begin(), cliv.end(),int4(0.0f, 0.0f, 0.0f, 0.0f));
        clf_debug = Buffer<float4>(ps->cli, clfv);
        cli_debug = Buffer<int4>(ps->cli, cliv);


        std::vector<unsigned int> keys(max_num);
        //to get around limits of bitonic sort only handling powers of 2
#include "limits.h"
        std::fill(keys.begin(), keys.end(), INT_MAX);
        cl_sort_indices  = Buffer<unsigned int>(ps->cli, keys);
        cl_sort_hashes   = Buffer<unsigned int>(ps->cli, keys);

        // for debugging. Store neighbors of indices
        // change nb of neighbors in cl_macro.h as well
        //cl_index_neigh = Buffer<int>(ps->cli, max_num*50);

        // Size is the grid size + 1, the last index is used to signify how many particles are within bounds
        // That is a problem since the number of
        // occupied cells could be much less than the number of grid elements.
        printf("%d\n", grid_params.nb_cells);
        std::vector<unsigned int> gcells(grid_params.nb_cells+1);
        int minus = 0xffffffff;
        std::fill(gcells.begin(), gcells.end(), 666);

        cl_cell_indices_start = Buffer<unsigned int>(ps->cli, gcells);
        cl_cell_indices_end   = Buffer<unsigned int>(ps->cli, gcells);
        //printf("gp.nb_points= %d\n", gp.nb_points); exit(0);

        // For bitonic sort. Remove when bitonic sort no longer used
        // Currently, there is an error in the Radix Sort (just run both
        // sorts and compare outputs visually
        cl_sort_output_hashes = Buffer<unsigned int>(ps->cli, keys);
        cl_sort_output_indices = Buffer<unsigned int>(ps->cli, keys);

		// Eventually, if I must sort every iteration, I can reuse these arrays. 
		// Due to potentially, large grid, this is very expensive, and one could run 
		// out of memory on CPU and GPU. 

		printf("keys.size= %d\n", keys.size()); // 
		printf("gcells.size= %d\n", gcells.size()); // 1729
		//exit(1);
     }

	//----------------------------------------------------------------------
    void SPH::setupDomain()
    {
        grid->calculateCells(sphp.smoothing_distance / sphp.simulation_scale);

        grid_params.grid_min = grid->getMin();
        grid_params.grid_max = grid->getMax();
        grid_params.bnd_min  = grid->getBndMin();
        grid_params.bnd_max  = grid->getBndMax();

        //grid_params.bnd_min = float4(1, 1, 1,0);
        //grid_params.bnd_max =  float4(4, 4, 4, 0);

        grid_params.grid_res = grid->getRes();
        grid_params.grid_size = grid->getSize();
        grid_params.grid_delta = grid->getDelta();
        grid_params.nb_cells = (int) (grid_params.grid_res.x*grid_params.grid_res.y*grid_params.grid_res.z);

        //printf("gp nb_cells: %d\n", grid_params.nb_cells);


        /*
        grid_params.grid_inv_delta.x = 1. / grid_params.grid_delta.x;
        grid_params.grid_inv_delta.y = 1. / grid_params.grid_delta.y;
        grid_params.grid_inv_delta.z = 1. / grid_params.grid_delta.z;
        grid_params.grid_inv_delta.w = 1.;
        */

        float ss = sphp.simulation_scale;

        grid_params_scaled.grid_min = grid_params.grid_min * ss;
        grid_params_scaled.grid_max = grid_params.grid_max * ss;
        grid_params_scaled.bnd_min  = grid_params.bnd_min * ss;
        grid_params_scaled.bnd_max  = grid_params.bnd_max * ss;
        grid_params_scaled.grid_res = grid_params.grid_res;
        grid_params_scaled.grid_size = grid_params.grid_size * ss;
        grid_params_scaled.grid_delta = grid_params.grid_delta / ss;
        //grid_params_scaled.nb_cells = (int) (grid_params_scaled.grid_res.x*grid_params_scaled.grid_res.y*grid_params_scaled.grid_res.z);
        grid_params_scaled.nb_cells = grid_params.nb_cells;
        //grid_params_scaled.grid_inv_delta = grid_params.grid_inv_delta / ss;
        //grid_params_scaled.grid_inv_delta.w = 1.0f;

        grid_params.print();
        grid_params_scaled.print();
    }

	//----------------------------------------------------------------------
    int SPH::addBox(int nn, float4 min, float4 max, bool scaled, float4 color)
    {
        float scale = 1.0f;
		#if 0
        if (scaled)
        {
            scale = sphp.simulation_scale;
        }
		#endif
		//printf("GEE inside addBox, before addRect, scale= %f\n", scale);
		//printf("GEE inside addBox, sphp.simulation_scale= %f\n", sphp.simulation_scale);
		//printf("GEE addBox spacing = %f\n", spacing);
        vector<float4> rect = addRect(nn, min, max, spacing, scale);
        float4 velo(0, 0, 0, 0);
        pushParticles(rect, velo, color);
        return rect.size();
    }

    void SPH::addBall(int nn, float4 center, float radius, bool scaled)
    {
        float scale = 1.0f;
        if (scaled)
        {
            scale = sphp.simulation_scale;
        }
        vector<float4> sphere = addSphere(nn, center, radius, spacing, scale);
        float4 velo(0, 0, 0, 0);
        pushParticles(sphere,velo);
    }

	//----------------------------------------------------------------------
    int SPH::addHose(int total_n, float4 center, float4 velocity, float radius, float4 color)
    {
        //in sph we just use sph spacing
        radius *= spacing;
        Hose *hose = new Hose(ps, total_n, center, velocity, radius, spacing, color);
        hoses.push_back(hose);
        //return the index
        return hoses.size()-1;
        //printf("size of hoses: %d\n", hoses.size());
    }
    void SPH::updateHose(int index, float4 center, float4 velocity, float radius, float4 color)
    {
        //we need to expose the vector of hoses somehow
        //doesn't seem right to make user manage an index
        //in sph we just use sph spacing
        radius *= spacing;
        hoses[index]->update(center, velocity, radius, spacing, color);
        //printf("size of hoses: %d\n", hoses.size());
    }
    void SPH::refillHose(int index, int refill)
    {
        hoses[index]->refill(refill);
    }



    void SPH::sprayHoses()
    {

        std::vector<float4> parts;
        for (int i = 0; i < hoses.size(); i++)
        {
            parts = hoses[i]->spray();
            if (parts.size() > 0)
                pushParticles(parts, hoses[i]->getVelocity(), hoses[i]->getColor());
        }
    }

    void SPH::testDelete()
    {

        //cut = 1;
        std::vector<float4> poss(40);
        float4 posx(100.,100.,100.,1.);
        std::fill(poss.begin(), poss.end(),posx);
        //cl_vars_unsorted.copyToDevice(poss, max_num + 2);
        cl_position_u.acquire();
        cl_position_u.copyToDevice(poss);
        cl_position_u.release();
        ps->cli->queue.finish();
    }
	//----------------------------------------------------------------------
    void SPH::pushParticles(vector<float4> pos, float4 velo, float4 color)
    {
        int nn = pos.size();
        std::vector<float4> vels(nn);
        std::fill(vels.begin(), vels.end(), velo);
        pushParticles(pos, vels, color);

    }
	//----------------------------------------------------------------------
    void SPH::pushParticles(vector<float4> pos, vector<float4> vels, float4 color)
    {
        //cut = 1;

        int nn = pos.size();
        if (num + nn > max_num)
        {
			printf("pushParticles: exceeded max nb(%d) of particles allowed\n", max_num);
            return;
        }
        //float rr = (rand() % 255)/255.0f;
        //float4 color(rr, 0.0f, 1.0f - rr, 1.0f);
        //printf("random: %f\n", rr);
        //float4 color(1.0f,1.0f,1.0f,1.0f);

        std::vector<float4> cols(nn);
        //printf("color: %f %f %f %f\n", color.x, color.y, color.z, color.w);

        std::fill(cols.begin(), cols.end(),color);
        //float v = .5f;
        //float v = 0.0f;
        //float4 iv = float4(v, v, -v, 0.0f);
        //float4 iv = float4(0, v, -.1, 0.0f);
        //std::fill(vels.begin(), vels.end(),iv);


#ifdef GPU
        glFinish();
        cl_position_u.acquire();
        cl_color_u.acquire();

        //printf("about to prep 0\n");
        //call_prep(0);
        //printf("done with prep 0\n");

		// Allocate max_num particles on the GPU. That wastes memory, but is useful. 
		// There should be a way to update this during the simulation. 
        cl_position_u.copyToDevice(pos, num);
        cl_color_u.copyToDevice(cols, num);
        cl_velocity_u.copyToDevice(vels, num);

        //sphp.num = num+nn;
        settings->SetSetting("Number of Particles", num+nn);
        updateSPHP();

        //cl_position.acquire();
        //cl_color_u.acquire();
        //reprep the unsorted (packed) array to account for new particles
        //might need to do it conditionally if particles are added or subtracted
        // -- no longer needed: april, enjalot
        //printf("about to prep\n");
        //call_prep(1);
        //printf("done with prep\n");
        cl_position_u.release();
        cl_color_u.release();
#endif
        num += nn;  //keep track of number of particles we use
        renderer->setNum(num);
    }
	//----------------------------------------------------------------------
    void SPH::render()
    {
        renderer->render_box(grid->getBndMin(), grid->getBndMax());
        //renderer->render_table(grid->getBndMin(), grid->getBndMax());
        System::render();
    }
	//----------------------------------------------------------------------
    void SPH::setRenderer()
    {
        switch(ps->settings->getRenderType())
        {
            case RTPSettings::SPRITE_RENDER:
                renderer = new SpriteRender(pos_vbo,col_vbo,num,ps->cli, ps->settings);
                //printf("spacing for radius %f\n", spacing);
                break;
            case RTPSettings::SCREEN_SPACE_RENDER:
                //renderer = new ScreenSpaceRender();
                renderer = new SSFRender(pos_vbo,col_vbo,num,ps->cli, ps->settings);
                break;
            case RTPSettings::RENDER:
                renderer = new Render(pos_vbo,col_vbo,num,ps->cli, ps->settings);
                break;
            default:
                //should be an error
                renderer = new Render(pos_vbo,col_vbo,num,ps->cli, ps->settings);
                break;
        }
        //renderer->setParticleRadius(spacing*0.5);
        renderer->setParticleRadius(spacing);
		//renderer->setRTPS(
    }
	//----------------------------------------------------------------------
    void SPH::radix_sort()
    {
    try 
        {   
            int snum = nlpo2(num);
            if(snum < 1024)
            {   
                snum = 1024;
            }   
            //printf("sorting snum: %d", snum); 
            radix.sort(snum, &cl_sort_hashes, &cl_sort_indices);
        }   
        catch (cl::Error er) 
        {   
            printf("ERROR(radix sort): %s(%s)\n", er.what(), oclErrorString(er.err()));
        }   

    }
	//----------------------------------------------------------------------
    void SPH::bitonic_sort()
    {
        try
        {
            int dir = 1;        // dir: direction
            //int batch = num;

            int arrayLength = nlpo2(num);
            //printf("num: %d\n", num);
            //printf("nlpo2(num): %d\n", arrayLength);
            //int arrayLength = max_num;
            //int batch = max_num / arrayLength;
            int batch = 1;

            //printf("about to try sorting\n");
            bitonic.Sort(batch, 
                        arrayLength, 
                        dir,
                        &cl_sort_output_hashes,
                        &cl_sort_output_indices,
                        &cl_sort_hashes,
                        &cl_sort_indices );

        }
        catch (cl::Error er)
        {
            printf("ERROR(bitonic sort): %s(%s)\n", er.what(), oclErrorString(er.err()));
            exit(0);
        }

        ps->cli->queue.finish();

        /*
        int nbc = 10;
        std::vector<int> sh = cl_sort_hashes.copyToHost(nbc);
        std::vector<int> eci = cl_cell_indices_end.copyToHost(nbc);
    
        for(int i = 0; i < nbc; i++)
        {
            printf("before[%d] %d eci: %d\n; ", i, sh[i], eci[i]);
        }
        printf("\n");
        */


        cl_sort_hashes.copyFromBuffer(cl_sort_output_hashes, 0, 0, num);
        cl_sort_indices.copyFromBuffer(cl_sort_output_indices, 0, 0, num);

        /*
        scopy(num, cl_sort_output_hashes.getDevicePtr(), 
              cl_sort_hashes.getDevicePtr());
        scopy(num, cl_sort_output_indices.getDevicePtr(), 
              cl_sort_indices.getDevicePtr());
        */

        ps->cli->queue.finish();
#if 0
    
        printf("********* Bitonic Sort Diagnostics **************\n");
        int nbc = 20;
        //sh = cl_sort_hashes.copyToHost(nbc);
        //eci = cl_cell_indices_end.copyToHost(nbc);
        std::vector<unsigned int> sh = cl_sort_hashes.copyToHost(nbc);
        std::vector<unsigned int> si = cl_sort_indices.copyToHost(nbc);
        //std::vector<int> eci = cl_cell_indices_end.copyToHost(nbc);

    
        for(int i = 0; i < nbc; i++)
        {
            //printf("after[%d] %d eci: %d\n; ", i, sh[i], eci[i]);
            printf("sh[%d] %d si: %d\n ", i, sh[i], si[i]);
        }

#endif
    }
    
#ifdef CLOUD_COLLISION
	//----------------------------------------------------------------------
	void SPH::cloudInitialize()
	{
		int max_nb_in_cloud = 8192;  // 1 << 13
		//printf("sphp scale= %f\n", sphp.simulation_scale); exit(1);
		cloud = new CLOUD(ps, sphp, &cl_GridParams, &cl_GridParamsScaled, 
		   &grid_params, &grid_params_scaled, max_nb_in_cloud);

		// I can define cl_sphp later since I am using pointers
		cloud->setSPHP(&cl_sphp);
		//printf("grid_params nb_cells= %d\n", grid_params.nb_cells); exit(1);
		//cloud->setGridParams(&cl_GridParams, &grid_params);
		cloud->setRenderer(renderer); // cloud arrays in cloud/ must be created
	}
	//----------------------------------------------------------------------
	void SPH::cloudUpdate()
	{
            cloud->cloud_hash_and_sort();
            cloud->cellindicesExecute();
            cloud->permuteExecute();
			if (num > 0) {
				cloud->collision(cl_position_s, cl_velocity_s, cl_force_s, cl_sphp, num);
;
			}
    		cloud->cloudVelocityExecute(); // before collision?
			cloud->integrate();
	}
	//----------------------------------------------------------------------
	void SPH::cloudCleanup()
	{
		// Cleanup afer iteration
		;
	}
	//----------------------------------------------------------------------
#endif

}; //end namespace

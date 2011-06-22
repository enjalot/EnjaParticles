#define CLOUD_COLLISION 1

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


    SPH::SPH(RTPS *psfr, int n, int max_nb_in_cloud)
    {
        //store the particle system framework
        ps = psfr;
        settings = ps->settings;
        max_num = n;
		//printf("max_num= %d\n", max_num); exit(0);
        num = 0;
        nb_var = 10;

		max_cloud_num = max_nb_in_cloud; // remove max_outer_num
		cloud_num = 0;
		// max_outer_particles defined in RTPSettings (used?)

		// I should be able to not specify this, but GPU restrictions ...

        resource_path = settings->GetSettingAs<string>("rtps_path");
        printf("resource path: %s\n", resource_path.c_str());

        //seed random
        srand ( time(NULL) );

        grid = settings->grid;

        //sphsettings = new SPHSettings(grid, max_num);
        //sphsettings->printSettings();
        //sphsettings->updateSPHP(sphp);

        std::vector<SPHParams> vparams(0);
        vparams.push_back(sphp);
        cl_sphp = Buffer<SPHParams>(ps->cli, vparams);

		std::vector<CLOUDParams> vcparams(0);
		vcparams.push_back(cloudp);
		cl_cloudp = Buffer<CLOUDParams>(ps->cli, vcparams);
        
        calculate();
        updateSPHP();
        updateCLOUDP();

        //settings->printSettings();

        spacing = settings->GetSettingAs<float>("Spacing");

        //SPH settings depend on number of particles used
        //calculateSPHSettings();
        //set up the grid
        setupDomain();

        integrator = LEAPFROG;
        //integrator = EULER;

		// inside main
    	//rtps::Domain* grid = new Domain(float4(0,0,0,0), float4(5, 5, 5, 0));
		// Create cloud object for testing
		//min = float4(1.2, 1.2, 3.2, 1.0f);
		//max = float4(2., 2., 4., 1.0f);
		// with a large radius, I am simulating a plane that particles 
		// should bounce off of
		float radius_in = 2.5;
		float radius_out = radius_in + .5;
		float4 center(1.6,1.6,2.7-radius_out, 0.0);
		//vector<float4> cloud_normals;
		bool scaled = true;

        //*** end Initialization

        setupTimers();

#ifdef CPU
        printf("RUNNING ON THE CPU\n");
#endif
#ifdef GPU
        printf("RUNNING ON THE GPU\n");

        
        //setup the sorted and unsorted arrays
        prepareSorted();

        //should be more cross platform
        sph_source_dir = resource_path + "/" + std::string(SPH_CL_SOURCE_DIR);
        common_source_dir = resource_path + "/" + std::string(COMMON_CL_SOURCE_DIR);

        ps->cli->addIncludeDir(sph_source_dir);
        ps->cli->addIncludeDir(common_source_dir);

        hash = Hash(common_source_dir, ps->cli, timers["hash_gpu"]);
        bitonic = Bitonic<unsigned int>(common_source_dir, ps->cli );
        cellindices = CellIndices(common_source_dir, ps->cli, timers["ci_gpu"] );
        permute = Permute( common_source_dir, ps->cli, timers["perm_gpu"] );
		printf("before cloud_permute\n");
        cloud_permute = CloudPermute( common_source_dir, ps->cli, timers["perm_gpu"] );

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

        string lt_file = settings->GetSettingAs<string>("lt_cl");
        //lifetime = Lifetime(sph_source_dir, ps->cli, timers["lifetime_gpu"], lt_file);


#endif

        // settings defaults to 0
        //renderer = new Render(pos_vbo,col_vbo,num,ps->cli, &ps->settings);
        setRenderer();

        //printf("MAIN settings: \n");
        //settings->printSettings();
        //printf("=================================================\n");

		// must be called after prepareSorted
		center = float4(2.5, 2.5, 0., 0.0);
		//center = float4(5.0, 2.5, 2.5, 0.0);
		//addHollowBall(2000, center, radius_in, radius_out, scaled, cloud_normals);
		int nn = 4000;
    	//addNewxyPlane(nn, scaled, cloud_normals); 
		readPointCloud(cloud_positions, cloud_normals, cloud_faces, cloud_faces_normals);

		//  ADD A SWITCH TO HANDLE CLOUD IF PRESENT
		// Must be called after a point cloud has been created. 
		if (cloud_num > 0) {
			collision_cloud = CollisionCloud(sph_source_dir, ps->cli, timers["ct_pgu"], max_cloud_num); // Last argument is? ??
		}

	//printf("max_cloud_num=%d\n", max_cloud_num);
		printf("cloud_positions capacity: %d\n", cloud_positions.capacity());
		printf("cloud_normals capacity: %d\n", cloud_normals.capacity());

		cloud_positions.resize(cloud_positions.capacity());
		cloud_normals.resize(cloud_normals.capacity());
        //exit(0);
		// only needs to be done once if cloud not moving
		// ideally, cloud should be stored in vbos. 
        cl_cloud_position_u.copyToHost(cloud_positions);
		renderer->setCloudData(cloud_positions, cloud_normals, cloud_faces, cloud_faces_normals, cloud_num);
    }

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
        //call kernels
        //TODO: add timings
#ifdef CPU
        updateCPU();
#endif
#ifdef GPU
        updateGPU();
#endif
    }

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
#if 0
        //printf("positions[0].z %f\n", positions[0].z);
        for (int i = 0; i < 100; i++)
        {
            //if(xsphs[i].z != 0.0)
            //printf("force: %f %f %f  \n", veleval[i].x, veleval[i].y, veleval[i].z);
            printf("force: %f %f %f  \n", xsphs[i].x, xsphs[i].y, xsphs[i].z);
            //printf("force: %f %f %f  \n", velocities[i].x, velocities[i].y, velocities[i].z);
        }
        //printf("cpu execute!\n");
#endif
        glBindBuffer(GL_ARRAY_BUFFER, pos_vbo);
        glBufferData(GL_ARRAY_BUFFER, num * sizeof(float4), &positions[0], GL_DYNAMIC_DRAW);
    }

	//----------------------------------------------------------------------
    void SPH::updateGPU()
    {
		printf("enter updateGPU, num= %d\n", num);

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
        //sub-intervals
        for (int i=0; i < sub_intervals; i++)
        {

            //if(num >0) printf("before hash and sort\n");
            hash_and_sort();

			// only for clouds (if cloud_num > 0)
#if CLOUD_COLLISION
			if (cloud_num > 0) {
            	cloud_hash_and_sort();
			}
#endif
            //if(num >0) printf("after hash and sort\n");

            //printf("data structures\n");
			// WHY ISN't THIS USED? GE
            /*
            timers["datastructures"]->start();
            int nc = datastructures.execute(   num,
                cl_position_u,
                cl_position_s,
                cl_velocity_u,
                cl_velocity_s,
                cl_veleval_u,
                cl_veleval_s,
                //cl_vars_unsorted,
                cl_color_u,
                cl_sort_hashes,
                cl_sort_indices,
                cl_cell_indices_start,
                cl_cell_indices_end,
                cl_sphp,
                cl_GridParams,
                grid_params.nb_cells,
                clf_debug,
                cli_debug);
            timers["datastructures"]->stop();
            */

            printf("before cellindices, num= %d\n", num);
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

			printf("num= %d\n", num);
			//if (num > 0) exit(1); //GE

			// I should be able to overlap with fluid sorting or fluid calculation
		#if CLOUD_COLLISION
			if (num > 0) { // SHOULD NOT BE NEEDED
			// SORT CLOUD
            printf("before cloud cellindices, num= %d, cloud_num= %d\n", num, cloud_num);
            timers["cellindices"]->start();

            int cloud_nc = cellindices.execute(cloud_num,
                cl_cloud_sort_hashes,
                cl_cloud_sort_indices,
                cl_cloud_cell_indices_start,
                cl_cloud_cell_indices_end,
                //cl_sphp,
                cl_GridParams,
                grid_params.nb_cells,
                clf_debug,
                cli_debug);
            timers["cellindices"]->stop();
			//printf("(deleted cloud particles?) cloud_nc= %d\n", cloud_nc);
			//exit(1);
			}
		#endif
       
            printf("*** enter fluid permute, num= %d\n", num);
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
			printf("exit after fluid permute\n");
			//if (num > 0) exit(0);
 
			// NUMBER OF CLOUD PARTICLES IS CONSTANT THROUGHOUT THE SIMULATION
 

            //printf("num %d, nc %d\n", num, nc);
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
                updateCLOUDP();
                renderer->setNum(sphp.num);
                //need to copy sorted arrays into unsorted arrays
                call_prep(2);
                //printf("HOW MANY NOW? %d\n", num);
                hash_and_sort();
                                //we've changed num and copied sorted to unsorted. skip this iteration and do next one
                //this doesn't work because sorted force etc. are having an effect?
                //continue; 
            }

		#if CLOUD_COLLISION
			if (num > 0) {
            //printf("permute\n");
            timers["cloud_permute"]->start();

			#if 1
            cloud_permute.execute(
			    cloud_num,
                cl_cloud_position_u,
                cl_cloud_position_s,
                cl_cloud_normal_u, 
                cl_cloud_normal_s,
                cl_cloud_sort_indices,
                cl_GridParams,
                clf_debug,
                cli_debug);
			#endif

            timers["cloud_permute"]->stop();
			//printf("exit cloud_permite\n"); exit(1);
		    }
		#endif


            //if(num >0) printf("density\n");
            timers["density"]->start();
            density.execute(   num,
                //cl_vars_sorted,
                cl_position_s,
                cl_density_s,
                cl_cell_indices_start,
                cl_cell_indices_end,
                cl_sphp,
                cl_GridParamsScaled,
                clf_debug,
                cli_debug);
            timers["density"]->stop();
            
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
                cl_GridParamsScaled,
                clf_debug,
                cli_debug);

            timers["force"]->stop();

			printf("before collision\n");
            collision();
			printf("after collision\n");
            timers["integrate"]->start();
            integrate();
            timers["integrate"]->stop();

            /*
            lifetime.execute( num,
                              settings->GetSettingAs<float>("lt_increment"),
                              cl_position_u,
                              cl_color_u,
                              cl_color_s,
                              cl_sort_indices,
                              clf_debug,
                              cli_debug
                              );

            */

            //
            //Andrew's rendering emporium
            //neighborSearch(4);
        }

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

    }

    void SPH::cloud_hash_and_sort()
    {
        //printf("cloud hash and sort\n"); exit(0);
        timers["hash"]->start();
        hash.execute(   cloud_num,
                //cl_vars_unsorted,
                cl_cloud_position_u,
                cl_cloud_sort_hashes,
                cl_cloud_sort_indices,
                //cl_sphp,
                cl_GridParams,
                clf_debug,
                cli_debug);
        timers["hash"]->stop();

        //printf("bitonic_sort\n");
        //defined in Sort.cpp
        timers["bitonic"]->start();
        cloud_bitonic_sort(); // DEFINED WHERE?
        timers["bitonic"]->stop();
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

		// NEED TIMER FOR POINT CLOUD COLLISIONS (GE)
		// Messed collisions up
		#if CLOUD_COLLISION
		if (num > 0) {
			collision_cloud.execute(num, cloud_num, 
				cl_position_s, 
				cl_velocity_s, 
				cl_cloud_position_s, 
				cl_cloud_normal_s,
				cl_force_s, // output

            	cl_cloud_cell_indices_start,
            	cl_cloud_cell_indices_end,

				cl_sphp,    // IS THIS CORRECT?
				cl_GridParamsScaled,
				// debug
				clf_debug,
				cli_debug);
		}
		#endif

    }
	//----------------------------------------------------------------------

    void SPH::integrate()
    {
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

#if 0
        if (num > 0)
        {
            std::vector<float4> pos = cl_position.copyToHost(num);
            for (int i = 0; i < num; i++)
            {
                printf("pos[%d] = %f %f %f\n", i, pos[i].x, pos[i].y, pos[i].z);
            }
        }
#endif


    }

	// GE: WHY IS THIS NEEDED?
    void SPH::call_prep(int stage)
    {
            cl_position_u.copyFromBuffer(cl_position_s, 0, 0, num);
            cl_velocity_u.copyFromBuffer(cl_velocity_s, 0, 0, num);
            cl_veleval_u.copyFromBuffer(cl_veleval_s, 0, 0, num);
            cl_color_u.copyFromBuffer(cl_color_s, 0, 0, num);
    }

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

		// BEGIN CLOUD
		cloud_positions.resize(max_cloud_num); // replace by max_cloud_num
		cloud_normals.resize(max_cloud_num);
		// END CLOUD

        //for reading back different values from the kernel
        std::vector<float4> error_check(max_num);
        
        float4 pmax = grid_params.grid_max + grid_params.grid_size;
        //std::fill(positions.begin(), positions.end(), pmax);

        //float4 color = float4(0.0, 1.0, 0.0, 1.0f);
        //std::fill(colors.begin(), colors.end(),color);
        std::fill(forces.begin(), forces.end(),float4(0.0f, 0.0f, 1.0f, 0.0f));
        std::fill(velocities.begin(), velocities.end(),float4(0.0f, 0.0f, 0.0f, 0.0f));
        std::fill(veleval.begin(), veleval.end(),float4(0.0f, 0.0f, 0.0f, 0.0f));

        std::fill(densities.begin(), densities.end(), 0.0f);
        std::fill(xsphs.begin(), xsphs.end(),float4(0.0f, 0.0f, 0.0f, 0.0f));
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

		//CLOUD BUFFERS
		if (max_cloud_num > 0) {
			printf("max_cloud_num= %d\n", max_cloud_num);
printf("cloud_positions size: %d\n", cloud_positions.size());
printf("cloud_positions size: %d\n", cloud_normals.size());
//exit(0);
        	cl_cloud_position_u = Buffer<float4>(ps->cli, cloud_positions);
        	cl_cloud_position_s = Buffer<float4>(ps->cli, cloud_positions);
        	cl_cloud_normal_u = Buffer<float4>(ps->cli, cloud_normals);
        	cl_cloud_normal_s = Buffer<float4>(ps->cli, cloud_normals);
		}

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


        /*
        //sorted and unsorted arrays
        std::vector<float4> unsorted(max_num*nb_var);
        std::vector<float4> sorted(max_num*nb_var);

        std::fill(unsorted.begin(), unsorted.end(),float4(0.0f, 0.0f, 0.0f, 1.0f));
        std::fill(sorted.begin(), sorted.end(),float4(0.0f, 0.0f, 0.0f, 1.0f));
        //std::fill(unsorted.begin(), unsorted.end(), pmax);
        //std::fill(sorted.begin(), sorted.end(), pmax);
        cl_vars_unsorted = Buffer<float4>(ps->cli, unsorted);
        cl_vars_sorted = Buffer<float4>(ps->cli, sorted);
        */

        std::vector<unsigned int> keys(max_num);
        std::vector<unsigned int> cloud_keys(max_cloud_num);
        //to get around limits of bitonic sort only handling powers of 2
#include "limits.h"
        std::fill(keys.begin(), keys.end(), INT_MAX);
        std::fill(cloud_keys.begin(), cloud_keys.end(), INT_MAX);
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

		if (max_cloud_num > 0) {
			//keys.resize(max_cloud_num);
        	cl_cloud_cell_indices_start = Buffer<unsigned int>(ps->cli, gcells);
        	cl_cloud_cell_indices_end   = Buffer<unsigned int>(ps->cli, gcells);
        	cl_cloud_sort_indices       = Buffer<unsigned int>(ps->cli, cloud_keys);
        	cl_cloud_sort_hashes        = Buffer<unsigned int>(ps->cli, cloud_keys);
        	cl_cloud_sort_output_hashes  = Buffer<unsigned int>(ps->cli, cloud_keys);
        	cl_cloud_sort_output_indices = Buffer<unsigned int>(ps->cli, cloud_keys);
		}

		printf("keys.size= %d\n", keys.size()); // 
		printf("cloud_keys.size= %d\n", cloud_keys.size()); // 4k
		printf("gcells.size= %d\n", gcells.size()); // 1729
		//exit(1);
     }

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
	void SPH::pushCloudParticles(vector<float4>& pos, vector<float4>& normals)
	{
		if ((cloud_num+pos.size()) > max_cloud_num) {
			printf("exceeded max number of cloud particles\n");
			exit(2); //GE
			return;
		}

		printf("pos.size= %d\n", pos.size());
		printf("normals.size= %d\n", normals.size());
		//exit(0);

		printf("cloud_num on entry: %d\n", cloud_num);
		//cloud_positions.resize(max_cloud_num); // replace by max_cloud_num
		//cloud_normals.resize(max_cloud_num);
		printf("pos.size= %d\n", pos.size());
        cl_cloud_position_u.copyToDevice(pos, cloud_num);

		//printf("cloud_num= %d\n", cloud_num);
		for (int i=0; i < normals.size(); i++) {
			//printf("%d\n", i);
			//normals[i].print("normals");
		}
        cl_cloud_normal_u.copyToDevice(normals, cloud_num);

		// Should be sorted, so this is temporary to check collision code
        //cl_cloud_position_s.copyToDevice(pos, cloud_num);
        //cl_cloud_normal_s.copyToDevice(normals, cloud_num);

		cloud_num += pos.size();
		printf("cloud_num= %d\n", cloud_num);

		#if 0
		for (int i=0; i < pos.size(); i++) {
			printf("i= %d, ", i);
			pos[i].print("pos");
		}
		for (int i=0; i < normals.size(); i++) {
			printf("i= %d, ", i);
			normals[i].print("norm");
		}
		#endif
		//printf("*******************\n"); exit(1);
		return;
	}
	//----------------------------------------------------------------------
	void SPH::readPointCloud(std::vector<float4>& cloud_positions, 
							 std::vector<float4>& cloud_normals,
							 std::vector<int4>& cloud_faces,
							 std::vector<int4>& cloud_faces_normals)
	{
    	//std::string file_vertex = "/Users/erlebach/arm1_vertex.txt";
    	//std::string file_normal = "/Users/erlebach/arm1_normal.txt";
    	//std::string file_face = "/Users/erlebach/arm1_faces.txt";

		std::string base = "/Users/erlebach/Documents/src/blender-particles/EnjaParticles/data/";

    	std::string file_vertex = base + "arm1_vertex.txt";
    	std::string file_normal = base + "arm1_normal.txt";
    	std::string file_face = base + "arm1_faces.txt";

		// I should really do: resize(cloud_num) which is initially zero
		cloud_positions.resize(0);
		cloud_normals.resize(0);
		cloud_faces.resize(0);

    	FILE* fd = fopen((const char*) file_vertex.c_str(), "r");
		if (fd == 0) {
			printf("cannot open: %s\n", file_vertex.c_str());
			exit(1);
		}
    	int nb = 5737;
    	float x, y, z;
    	for (int i=0; i < nb; i++) {
        	fscanf(fd, "%f %f %f\n", &x, &y, &z);
        	//printf("x,y,z= %f, %f, %f\n", x, y, z);
			cloud_positions.push_back(float4(x,y,z,1.));
			//cloud_positions[i] = float4(x,y,z,1.);
    	}

    	fclose(fd);

		printf("before normal read: normal size: %d\n", cloud_normals.size());

		printf("file_normal: %s\n", file_normal.c_str());
    	fd = fopen((const char*) file_normal.c_str(), "r");
    	for (int i=0; i < nb; i++) {
        	fscanf(fd, "%f %f %f\n", &x, &y, &z);
        	//printf("x,y,z= %f, %f, %f\n", x, y, z);
			cloud_normals.push_back(float4(x,y,z,0.));
			//cloud_normals[i] = float4(x,y,z,0.);
    	}


		// rescale point clouds
		// domain center is x=y=z=2.5
		float4 center(2.5, 2.5, 1.50, 1.); // center of domain
		// compute bounding box
		float xmin = 1.e10, ymin= 1.e10, zmin=1.e10;
		float xmax = -1.e10, ymax= -1.e10, zmax= -1.e10;
		for (int i=0; i < nb; i++) {
			float4& f = cloud_positions[i];
			xmin = (f.x < xmin) ? f.x : xmin;
			ymin = (f.y < ymin) ? f.y : ymin;
			zmin = (f.z < zmin) ? f.z : zmin;
			xmax = (f.x > xmax) ? f.x : xmax;
			ymax = (f.y > ymax) ? f.y : ymax;
			zmax = (f.z > zmax) ? f.z : zmax;
		}

		// center of hand
		float4 rcenter(0.5*(xmin+xmax), 0.5*(ymin+ymax), 0.5*(zmin+zmax), 1.);

		float xr = (xmax-xmin);
		float yr = (ymax-ymin);
		float zr = (zmax-zmin);

		float maxr = xr;
		maxr = yr > maxr ? yr : maxr;
		maxr = zr > maxr ? zr : maxr; // max size of box

		float scale = 2.;  // set back to 2 or 3
		float4 trans = center - rcenter;
		rcenter.print("rcenter");
		center.print("center");
		trans.print("trans");
		float s;

		// arms is now in [0,1]^3 with center (0.5)^3
		for (int i=0; i < nb; i++) {
			float4& f = cloud_positions[i];
			f.x = (f.x + trans.x - center.x)*scale + center.x; 
			f.y = (f.y + trans.y - center.y)*scale + center.y; 
			f.z = (f.z + trans.z - center.z)*scale + center.z; 
			//f.y = f.y + trans.y; 
			//f.z = f.z + trans.z; 
			f.x = center.x - (f.x-center.x);
		}

		// scale hand and move to center
		#if 0
		for (int i=0; i < nb; i++) {
			float4& f = cloud_positions[i];
			float s = f.y;
		}
		#endif

		nb = 5713;

    	int v1, v2, v3, v4;
    	int n1, n2, n3, n4;
    	fd = fopen((const char*) file_face.c_str(), "r");
    	for (int i=0; i < nb; i++) {
        	fscanf(fd, "%d//%d %d//%d %d//%d %d//%d\n", &v1, &n1, &v2, &n2, &v3, &n3, &v4, &n4);
        	//printf("x,y,z= %d, %d, %d\n", v1, v2, v3);
        	//printf("x,y,z= %d, %d, %d\n", n2, n2, n3);
			//exit(1);
        	//printf("--------------------\n");
			cloud_faces.push_back(int4(v1,v2,v3,v4)); // forms a face
			cloud_faces_normals.push_back(int4(n1,n2,n3,n4)); // forms a face
    	}


		// push onto GPU
		pushCloudParticles(cloud_positions, cloud_normals);
	}

	//----------------------------------------------------------------------
    void SPH::addNewxyPlane(int np, bool scaled, vector<float4>& normals)
	{
		float scale = 1.0f;
		float4 mmin = float4(-.5,-.5,2.5,1.);
		float4 mmax = float4(5.5,5.5,2.5,1.);
		float zlevel = 2.;
		float sp = spacing / 4. ;
		//printf("spacing= %f\n", spacing); exit(0);
		vector<float4> plane = addxyPlane(6000, mmin, mmax, sp, scale, zlevel, normals);
		//printf("plane size: %d\n", plane.size()); exit(0);
        printf("plane.size(): %zd\n", plane.size());
        printf("normals.size(): %zd\n", plane.size());
        pushCloudParticles(plane,normals);

    	for (int i=0; i < plane.size(); i++) {
			float4& n = normals[i];
			float4& v = plane[i];
			//n.print("normal");
			//v.print("vertex");
    	}
		//exit(1);
	}
	//----------------------------------------------------------------------
    void SPH::addHollowBall(int nn, float4 center, float radius_in, float radius_out, bool scaled, vector<float4>& normals)
    {
        float scale = 1.0f;
		#if 0
        if (scaled)
        {
            scale = sphp.simulation_scale;
        }
		#endif
		//printf("GEE inside addHollowBall, before addHollowSphere, scale= %f\n", scale);
		//printf("GEE inside addHollowBall, sphp.simulation_scale= %f\n", sphp.simulation_scale);
		//printf("spacing= %f\n", spacing);
		//printf("GEE addHollowSphere spacing = %f\n", spacing);

printf("before HollowSphere\n");
printf("cloud_positions size: %d\n", cloud_positions.size());
printf("cloud_normals size: %d\n", cloud_normals.size());
        vector<float4> sphere = addHollowSphere(nn, center, radius_in, radius_out, spacing/2., scale, normals);
printf("after HollowSphere\n");
printf("cloud_positions size: %d\n", cloud_positions.size());
printf("cloud_normals size: %d\n", cloud_normals.size());
printf("normals size: %d\n", normals.size());
printf("sphere size: %d\n", sphere.size());
        float4 velo(0, 0, 0, 0);
		printf("** addHollowBall: nb particles: %d\n", sphere.size());
		for (int i=0; i < sphere.size(); i++) {
			printf("%d, %f, %f, %f\n", i, sphere[i].x, sphere[i].y, sphere[i].z);
		}
		// pos of cloud in world coordinates
		// simulation coord = (world coord) * simulation_scale
        pushCloudParticles(sphere,normals);
printf("cloud_positions size: %d\n", cloud_positions.size());
printf("cloud_normals size: %d\n", cloud_normals.size());
//exit(0);
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
        updateCLOUDP();

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

}; //end namespace

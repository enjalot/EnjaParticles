#define CLOUD_COLLISION 0

#include <GL/glew.h>
#include <math.h>
#include <sstream>
#include <iomanip>
#include <string>

#include "System.h"
#include "Cloud.h"
//#include "../domain/UniformGrid.h"
#include "Domain.h"
#include "IV.h"

#include "common/Hose.h"


//for random
#include<time.h>

namespace rtps
{
    //using namespace sph;

	void CLOUD::printDevArray(Buffer<float4>& cl_array, char* msg, int nb_el, int nb_print)
	{
		std::vector<float4> pos(nb_el);
		cl_array.copyToHost(pos);
		printf("*** %s ***\n", msg);
		for (int i=0; i < nb_print; i++) {
			printf("i= %d: ", i);
			pos[i].print(msg);
		}
	}
	//----------------------------------------------------------------------
    CLOUD::CLOUD(RTPS *psfr, SPHParams& sphp, Buffer<GridParams>* cl_GridParams, GridParams* grid_params, int max_nb_in_cloud) 
    {
		//this->sphp = &sphp; // ADD LATER?

		this->cl_GridParams = cl_GridParams;
		this->grid_params   = grid_params;

        //store the particle system framework
        ps = psfr;
        settings = ps->settings;
        //max_num = n;

		cloud_max_num = max_nb_in_cloud; // remove max_outer_num
		cloud_num = 0;
		// max_outer_particles defined in RTPSettings (used?)

		// I should be able to not specify this, but GPU restrictions ...
        resource_path = settings->GetSettingAs<string>("rtps_path");
        printf("resource path: %s\n", resource_path.c_str());

        //seed random
        srand ( time(NULL) );

        //grid = settings->grid;

		std::vector<CLOUDParams> vcparams(0);
		vcparams.push_back(cloudp);
		cl_cloudp = Buffer<CLOUDParams>(ps->cli, vcparams);
        
        //calculate();
        //updateCLOUDP();

        //settings->printSettings();

        spacing = settings->GetSettingAs<float>("Spacing");

		// should bounce off 
		float radius_in = 2.5;
		float radius_out = radius_in + .5;
		float4 center(1.6,1.6,2.7-radius_out, 0.0);
		//vector<float4> cloud_normals;
		bool scaled = true;

		// Ideally, change every time step, and update all points in 
		// the cloud on the GPU (in which routine?)
		avg_cloud_velocity = float4(3., 0., 0., 1.);

        setupTimers();

#ifdef CPU
        printf("RUNNING ON THE CPU\n");
		printf("No CPU implementation for point clouds\n");
		exit(1);
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
        bitonic = CloudBitonic<unsigned int>(common_source_dir, ps->cli );
        cellindices = CellIndices(common_source_dir, ps->cli, timers["ci_gpu"] );
        permute = Permute( common_source_dir, ps->cli, timers["perm_gpu"] );

        cloud_permute = CloudPermute( common_source_dir, ps->cli, timers["perm_gpu"] );

		// CLOUD Integrator
		// ADD Cloud timers later. 
		cloud_euler = CloudEuler(sph_source_dir, ps->cli, timers["euler_gpu"]);

        string lt_file = settings->GetSettingAs<string>("lt_cl");
#endif

		// must be called after prepareSorted
		center = float4(2.5, 2.5, 0., 0.0);
		//center = float4(5.0, 2.5, 2.5, 0.0);
		//addHollowBall(2000, center, radius_in, radius_out, scaled, cloud_normals);
		//int nn = 4000;
    	//addNewxyPlane(nn, scaled, cloud_normals); 
		readPointCloud(cloud_positions, cloud_normals, cloud_faces, cloud_faces_normals);

		//printf("cloud_num = %d\n", cloud_num); exit(4);

		//  ADD A SWITCH TO HANDLE CLOUD IF PRESENT
		// Must be called after a point cloud has been created. 
		if (cloud_num > 0) {
			collision_cloud = CollisionCloud(sph_source_dir, ps->cli, timers["ct_pgu"], cloud_max_num); // Last argument is? ??
		}

		printf("cloud_positions capacity: %d\n", cloud_positions.capacity());
		printf("cloud_normals capacity: %d\n", cloud_normals.capacity());

		cloud_positions.resize(cloud_positions.capacity());
		cloud_normals.resize(cloud_normals.capacity());
		// only needs to be done once if cloud not moving
		// ideally, cloud should be stored in vbos. 
        cl_cloud_position_u.copyToHost(cloud_positions);

		//printf("*** end of CLOUD constructor ***\n");
		//printf("*** Unsorted cloud particles are place on the GPU ***\n");
		#if 1
		//cl_cloud_position_u.copyToHost(cloud_positions);
		for (int i=0; i < 3; i++) {
			printf("i= %d, ", i);
			cloud_positions[i].print("pos_u");
		}
		#endif
		//printf("cloud_num = %d\n", cloud_num); exit(0);
        settings->SetSetting("Maximum Number of Cloud Particles", cloud_max_num);
    }

	//----------------------------------------------------------------------
    CLOUD::~CLOUD()
    {
    }

	//----------------------------------------------------------------------
    void CLOUD::update()
    {
		//printf("+++++++++++++ enter UPDATE()\n");
        //call kernels
        //TODO: add timings
#ifdef CPU
        //updateCPU();
#endif
#ifdef GPU
        updateGPU();
#endif
    }

	//----------------------------------------------------------------------
    void CLOUD::updateGPU()
    {

        timers["update"]->start();
        //glFinish();
        //if (settings->has_changed()) updateSPHP();

        //settings->printSettings();

        //int sub_intervals = 3;  //should be a setting
        int sub_intervals =  settings->GetSettingAs<float>("sub_intervals");
        //this should go in the loop but avoiding acquiring and releasing each sub
        //interval for all the other calls.
        //this does end up acquire/release everytime sprayHoses calls pushparticles
        //should just do try/except?

        for (int i=0; i < sub_intervals; i++)
        {

			// only for clouds (if cloud_num > 0)
#if CLOUD_COLLISION
			if (cloud_num > 0) {
				//printf("BEFORE CLOUD_HASH_AND_SORT\n");
				//printDevArray(cl_cloud_position_u, "pos_u", cloud_num, 3); // OK
				//printDevArray(cl_cloud_position_s, "pos_s", cloud_num, 3); // WRONG
            	cloud_hash_and_sort();
				//printf("AFTER CLOUD_HASH_AND_SORT\n");
				//printDevArray(cl_cloud_position_u, "pos_u", cloud_num, 3); // OK
				//printDevArray(cl_cloud_position_s, "pos_s", cloud_num, 3); // WRONG
			}
#endif

		#if CLOUD_COLLISION
		//	if (num > 0) { // SHOULD NOT BE NEEDED
			// SORT CLOUD
            //printf("before cloud cellindices, num= %d, cloud_num= %d\n", num, cloud_num);
            timers["cellindices"]->start();

            cellindices.execute(cloud_num,
                cl_cloud_sort_hashes,
                cl_cloud_sort_indices,
                cl_cloud_cell_indices_start,
                cl_cloud_cell_indices_end,
                //cl_sphp,
                &cl_GridParams,
                grid_params->nb_cells,
                clf_debug,
                cli_debug);
            timers["cellindices"]->stop();
			//}
		#endif
       
		#if CLOUD_COLLISION
			if (cloud_num > 0) {
            //printf("permute\n");
            timers["cloud_permute"]->start();

			//printf("BEFORE CLOUD_PERMUTE\n");
			//printDevArray(cl_cloud_position_u, "pos_u", cloud_num, 3); // OK
			//printDevArray(cl_cloud_position_s, "pos_s", cloud_num, 3); // WRONG

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

			//printf("AFTER CLOUD_PERMUTE\n");
			//printDevArray(cl_cloud_position_u, "pos_u", cloud_num, 3); // OK
			//printDevArray(cl_cloud_position_s, "pos_s", cloud_num, 3); // WRONG
			cl_cloud_position_s.copyToHost(cloud_positions);

            timers["cloud_permute"]->stop();
			//printf("exit cloud_permute\n"); exit(1);
		    }
		#endif

            //collision();
			// CALL THIS FROM SPH

            integrate(); // includes boundary force
        }

		cl_cloud_position_s.copyToHost(cloud_positions);

        timers["update"]->stop();
    }

	//----------------------------------------------------------------------
	void CLOUD::permuteExecute()
	{
            cloud_permute.execute(
			    cloud_num,
                cl_cloud_position_u,
                cl_cloud_position_s,
                cl_cloud_normal_u, 
                cl_cloud_normal_s,
                cl_cloud_sort_indices,
                *cl_GridParams,
                clf_debug,
                cli_debug);
	}
	//----------------------------------------------------------------------
	void CLOUD::cellindicesExecute()
	{
		cellindices.execute(cloud_num,
                cl_cloud_sort_hashes,
                cl_cloud_sort_indices,
                cl_cloud_cell_indices_start,
                cl_cloud_cell_indices_end,
                //cl_sphp,
                *cl_GridParams,
                grid_params->nb_cells,
                clf_debug,
                cli_debug);
	}
	//----------------------------------------------------------------------
    void CLOUD::cloud_hash_and_sort()
    {
        timers["hash"]->start();
        hash.execute( cloud_num,
                cl_cloud_position_u,
                cl_cloud_sort_hashes,
                cl_cloud_sort_indices,
                *cl_GridParams,
                clf_debug,
                cli_debug);
        timers["hash"]->stop();

        cloud_bitonic_sort(); 
    }

	//----------------------------------------------------------------------
    void CLOUD::collision(Buffer<float4>& cl_pos_s, Buffer<float4>& cl_vel_s, 
	          Buffer<float4>& cl_force_s, Buffer<SPHParams>& cl_sphp, int num_sph)
    {
		;
		// NEED TIMER FOR POINT CLOUD COLLISIONS (GE)
		// Messed collisions up
		#if CLOUD_COLLISION
		// ****** Call from SPH? *****
		#if 1
		if (num_sph > 0) {
			collision_cloud.execute(num_sph, cloud_num, 
				cl_pos_s, 
				cl_vel_s,  
				cl_cloud_position_s, 
				cl_cloud_normal_s,
				cl_cloud_velocity_s,
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
		#endif
    }
	//----------------------------------------------------------------------

    void CLOUD::integrate()
    {

        timers["integrate"]->start();
		// Perhaps I am messed up by Courant condition if cloud point 
		// velocities are too large? 

		static int count=0;

		printf("settings->dt= %f\n", settings->dt);

		// CLOUD INTEGRATION
		#if 1
		//printf("*** BEFORE CLOUD_EULER ***\n");
		//printDevArray(cl_cloud_position_s, "pos_s", cloud_num, 3); // WRONG
		//printDevArray(cl_cloud_position_u, "pos_u", cloud_num, 3); // WRONG

		// start the arm moving after x iterations
		if (count > 10) {

			// How to prevent the cloud from advecting INTO THE FLUID? 
			//printf("cloud euler, cloud_num= %d\n", cloud_num);
			// returns unsorted positions
            cloud_euler.execute(cloud_num,
                settings->dt,
                cl_cloud_position_u,
                cl_cloud_position_s,
                cl_cloud_normal_u,
                cl_cloud_normal_s,
                cl_cloud_velocity_u,
                cl_cloud_velocity_s,
                cl_cloud_sort_indices,
                *cl_sphp,
                //debug
                clf_debug,
                cli_debug);

		}
		count++;

			//printf("*** AFTER CLOUD_EULER ***\n");
			//printDevArray(cl_cloud_position_s, "pos_s", cloud_num, 3); // WRONG
			//printDevArray(cl_cloud_position_u, "pos_u", cloud_num, 3); // WRONG
		#endif

        timers["integrate"]->stop();
    }

	//----------------------------------------------------------------------
	//----------------------------------------------------------------------
    int CLOUD::setupTimers()
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
    void CLOUD::printTimers()
    {
		#if 0
        timers.printAll();
        std::ostringstream oss; 
        oss << "sph_timer_log_" << std::setw( 7 ) << std::setfill( '0' ) <<  num; 
        //printf("oss: %s\n", (oss.str()).c_str());

        timers.writeToFile(oss.str()); 
		#endif
    }

	//----------------------------------------------------------------------
    void CLOUD::prepareSorted()
    {
		// BEGIN CLOUD
		cloud_positions.resize(cloud_max_num); // replace by cloud_max_num
		cloud_normals.resize(cloud_max_num);
		cloud_velocity.resize(cloud_max_num);
		// Should really be done every iteration unless constant
		std::fill(cloud_velocity.begin(), cloud_velocity.end(), avg_cloud_velocity);
		// END CLOUD

        //for reading back different values from the kernel
        //std::vector<float4> error_check(max_num);
        
		//grid_params->grid_max.print("xxx"); exit(0);
        float4 pmax = grid_params->grid_max + grid_params->grid_size;

        //std::fill(error_check.begin(), error_check.end(), float4(0.0f, 0.0f, 0.0f, 0.0f));

		printf("cloud prepare\n");

		//CLOUD BUFFERS
		if (cloud_max_num > 0) {
			printf("cloud_max_num= %d\n", cloud_max_num);
			printf("cloud_positions size: %d\n", cloud_positions.size());
        	cl_cloud_position_u = Buffer<float4>(ps->cli, cloud_positions);
        	cl_cloud_position_s = Buffer<float4>(ps->cli, cloud_positions);
        	cl_cloud_normal_u   = Buffer<float4>(ps->cli, cloud_normals);
        	cl_cloud_normal_s   = Buffer<float4>(ps->cli, cloud_normals);
        	cl_cloud_velocity_u = Buffer<float4>(ps->cli, cloud_velocity);
        	cl_cloud_velocity_s = Buffer<float4>(ps->cli, cloud_velocity);
		}

		//cl_cloud_position_u.copyToHost(cloud_positions); printf("xx\n");exit(1);

        //TODO make a helper constructor for buffer to make a cl_mem from a struct
        //Setup Grid Parameter structs
        //std::vector<GridParams> gparams(0);
        //gparams.push_back(grid_params);
        //cl_GridParams = Buffer<GridParams>(ps->cli, gparams);
        //scaled Grid Parameters
        //std::vector<GridParams> sgparams(0);
        //sgparams.push_back(grid_params_scaled);
        //cl_GridParamsScaled = Buffer<GridParams>(ps->cli, sgparams);

        //setup debug arrays
        std::vector<float4> clfv(cloud_max_num);
        std::fill(clfv.begin(), clfv.end(),float4(0.0f, 0.0f, 0.0f, 0.0f));
        std::vector<int4> cliv(cloud_max_num);
        std::fill(cliv.begin(), cliv.end(),int4(0.0f, 0.0f, 0.0f, 0.0f));
        clf_debug = Buffer<float4>(ps->cli, clfv);
        cli_debug = Buffer<int4>(ps->cli, cliv);


        std::vector<unsigned int> cloud_keys(cloud_max_num);
        //to get around limits of bitonic sort only handling powers of 2
#include "limits.h"
        //std::fill(keys.begin(), keys.end(), INT_MAX);
        std::fill(cloud_keys.begin(), cloud_keys.end(), INT_MAX);

        // Size is the grid size + 1, the last index is used to signify how many particles are within bounds
        // That is a problem since the number of
        // occupied cells could be much less than the number of grid elements.
        printf("grid_params->nb_cells: %d\n", grid_params->nb_cells);
        std::vector<unsigned int> gcells(grid_params->nb_cells+1);
        int minus = 0xffffffff;
        std::fill(gcells.begin(), gcells.end(), 666);

		//printf("gcells size: %d\n", gcells.size()); exit(0);

		if (cloud_max_num > 0) {
			//keys.resize(cloud_max_num);
        	cl_cloud_cell_indices_start  = Buffer<unsigned int>(ps->cli, gcells);
        	cl_cloud_cell_indices_end    = Buffer<unsigned int>(ps->cli, gcells);
        	cl_cloud_sort_indices        = Buffer<unsigned int>(ps->cli, cloud_keys);
        	cl_cloud_sort_hashes         = Buffer<unsigned int>(ps->cli, cloud_keys);
        	cl_cloud_sort_output_hashes  = Buffer<unsigned int>(ps->cli, cloud_keys);
        	cl_cloud_sort_output_indices = Buffer<unsigned int>(ps->cli, cloud_keys);
		}

		//printf("keys.size= %d\n", keys.size()); // 
		printf("cloud_keys.size= %d\n", cloud_keys.size()); // 8192
		printf("gcells.size= %d\n", gcells.size()); // 1729
		//exit(0);
     }
	 //----------------------------------------------------------------------

    int CLOUD::addBox(int nn, float4 min, float4 max, bool scaled, float4 color)
    {
	#if 0
        float scale = 1.0f;
		#if 0
        if (scaled)
        {
            scale = sphp->simulation_scale;
        }
		#endif
		//printf("GEE inside addBox, before addRect, scale= %f\n", scale);
		//printf("GEE inside addBox, sphp->simulation_scale= %f\n", sphp->simulation_scale);
		//printf("GEE addBox spacing = %f\n", spacing);
        vector<float4> rect = addRect(nn, min, max, spacing, scale);
        float4 velo(0, 0, 0, 0);
        pushParticles(rect, velo, color);
        return rect.size();
	#endif
    }

    void CLOUD::addBall(int nn, float4 center, float radius, bool scaled)
    {
	#if 0
        float scale = 1.0f;
        if (scaled)
        {
            scale = sphp->simulation_scale;
        }
        vector<float4> sphere = addSphere(nn, center, radius, spacing, scale);
        float4 velo(0, 0, 0, 0);
        pushParticles(sphere,velo);
	#endif
    }

	//----------------------------------------------------------------------
	void CLOUD::pushCloudParticles(vector<float4>& pos, vector<float4>& normals)
	{
		if ((cloud_num+pos.size()) > cloud_max_num) {
			printf("exceeded max number of cloud particles\n");
			exit(2); //GE
			return;
		}

		printf("pos.size= %d\n", pos.size());
		printf("normals.size= %d\n", normals.size());
		//exit(0);

		printf("cloud_num on entry: %d\n", cloud_num);
		printf("pos.size= %d\n", pos.size());
        cl_cloud_position_u.copyToDevice(pos, cloud_num);

		//printf("cloud_num= %d\n", cloud_num);
		printf("INSIDE pushCloudeParticles\n");
		for (int i=0; i < 3; i++) {
			printf("%d\n", i);
			pos[i].print("cloud_pos");
		}
        cl_cloud_normal_u.copyToDevice(normals, cloud_num);

		cloud_num += pos.size();
		printf("cloud_num= %d\n", cloud_num);

		return;
	}
	//----------------------------------------------------------------------
	void CLOUD::readPointCloud(std::vector<float4>& cloud_positions, 
							   std::vector<float4>& cloud_normals,
							   std::vector<int4>&   cloud_faces,
							   std::vector<int4>&   cloud_faces_normals)
	{
		// mac
		std::string base = "/Users/erlebach/Documents/src/blender-particles/EnjaParticles/data/";

		// Linux/Vislab
		//std::string base = "/panfs/panasas1/users/gerlebacher/vislab/src/blender-particles/EnjaParticles_nogit/EnjaParticles/data/";

    	std::string file_vertex = base + "arm1_vertex.txt";
    	std::string file_normal = base + "arm1_normal.txt";
    	std::string file_face = base + "arm1_faces.txt";

		// I should really do: resize(cloud_num) which is initially zero
		cloud_positions.resize(0);
		cloud_normals.resize(0);
		cloud_faces.resize(0);
		cloud_faces_normals.resize(0);

    	FILE* fd = fopen((const char*) file_vertex.c_str(), "r");
		if (fd == 0) {
			printf("cannot open: %s\n", file_vertex.c_str());
			exit(1);
		}
    	int nb = 5737;
    	float x, y, z;
    	for (int i=0; i < nb; i++) {
        	fscanf(fd, "%f %f %f\n", &x, &y, &z);
        	//printf("vertex %d:, x,y,z= %f, %f, %f\n", i, x, y, z);
			cloud_positions.push_back(float4(x,y,z,1.));
    	}

    	fclose(fd);

		//printf("before normal read: normal size: %d\n", cloud_normals.size());

		//printf("file_normal: %s\n", file_normal.c_str());
    	fd = fopen((const char*) file_normal.c_str(), "r");
    	for (int i=0; i < nb; i++) {
        	fscanf(fd, "%f %f %f\n", &x, &y, &z);
        	//printf("normal %d: x,y,z= %f, %f, %f\n", i, x, y, z);
			cloud_normals.push_back(float4(x,y,z,0.));
    	}


		// rescale point clouds
		// domain center is x=y=z=2.5
		float4 center(2.5, 2.5, 1.50, 1.); // center of domain (shifted in z)
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
		float s;

		// arms is now in [0,1]^3 with center (0.5)^3
		for (int i=0; i < nb; i++) {
			float4& f = cloud_positions[i];
			f.x = (f.x + trans.x - center.x)*scale + center.x; 
			f.y = (f.y + trans.y - center.y)*scale + center.y; 
			f.z = (f.z + trans.z - center.z)*scale + center.z; 
			f.x = center.x - (f.x-center.x);
		}

		// scale hand and move to center

		nb = 5713;

    	int v1, v2, v3, v4;
    	int n1, n2, n3, n4;
    	fd = fopen((const char*) file_face.c_str(), "r");
    	for (int i=0; i < nb; i++) {
        	fscanf(fd, "%d//%d %d//%d %d//%d %d//%d\n", &v1, &n1, &v2, &n2, &v3, &n3, &v4, &n4);
			cloud_faces.push_back(int4(v1-1,v2-1,v3-1,v4-1)); // forms a face
			cloud_faces_normals.push_back(int4(n1-1,n2-1,n3-1,n4-1)); // forms a face
    	}

		#if 1
		for (int i=0; i < cloud_faces_normals.size(); i++) {
			int4& n = cloud_faces_normals[i];
			float4& n1 = cloud_normals[n.x];
			float4& n2 = cloud_normals[n.y];
			float4& n3 = cloud_normals[n.z];
			float4& n4 = cloud_normals[n.w];
		}
		#endif

		// push onto GPU
		pushCloudParticles(cloud_positions, cloud_normals);
	}

	//----------------------------------------------------------------------
    void CLOUD::addNewxyPlane(int np, bool scaled, vector<float4>& normals)
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
    void CLOUD::addHollowBall(int nn, float4 center, float radius_in, float radius_out, bool scaled, vector<float4>& normals)
    {
        float scale = 1.0f;
		#if 0
        if (scaled)
        {
            scale = sphp->simulation_scale;
        }
		#endif
		//printf("GEE inside addHollowBall, before addHollowSphere, scale= %f\n", scale);
		//printf("GEE inside addHollowBall, sphp->simulation_scale= %f\n", sphp->simulation_scale);
		//printf("spacing= %f\n", spacing);
		//printf("GEE addHollowSphere spacing = %f\n", spacing);

        vector<float4> sphere = addHollowSphere(nn, center, radius_in, radius_out, spacing/2., scale, normals);
        float4 velo(0, 0, 0, 0);
		printf("** addHollowBall: nb particles: %d\n", sphere.size());
		for (int i=0; i < sphere.size(); i++) {
			printf("%d, %f, %f, %f\n", i, sphere[i].x, sphere[i].y, sphere[i].z);
		}
		// pos of cloud in world coordinates
		// simulation coord = (world coord) * simulation_scale
        pushCloudParticles(sphere,normals);
//exit(0);
    }
	//----------------------------------------------------------------------
    void CLOUD::cloud_bitonic_sort()    // GE
    {
        timers["bitonic"]->start();
        try
        {
            int dir = 1;        // dir: direction
            //int batch = num;

			printf("before cloud_bitonic_sort, cloud_max_num= %d\n", cloud_max_num);
            int arrayLength = nlpo2(cloud_num);
            //printf("num: %d\n", num);
            printf("nlpo2(num): %d\n", arrayLength);
            //int arrayLength = max_num;
            //int batch = max_num / arrayLength;
            int batch = 1;


            //printf("about to try sorting\n");
            bitonic.Sort(batch, 
                        arrayLength,  //GE?? ???
                        dir,
                        &cl_cloud_sort_output_hashes,
                        &cl_cloud_sort_output_indices,
                        &cl_cloud_sort_hashes,
                        &cl_cloud_sort_indices );

        }
        catch (cl::Error er)
        {
            printf("ERROR(bitonic sort): %s(%s)\n", er.what(), oclErrorString(er.err()));
            exit(0);
        }

		printf("after cloud bitonic sort\n");

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


		// NOT SURE HOW THIS WORKS!! GE
        cl_cloud_sort_hashes.copyFromBuffer(cl_cloud_sort_output_hashes, 0, 0, cloud_num);
        cl_cloud_sort_indices.copyFromBuffer(cl_cloud_sort_output_indices, 0, 0, cloud_num);

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
        std::vector<unsigned int> sh = cl_cloud_sort_hashes.copyToHost(nbc);
        std::vector<unsigned int> si = cl_cloud_sort_indices.copyToHost(nbc);
        //std::vector<int> eci = cl_cell_indices_end.copyToHost(nbc);

    
        for(int i = 0; i < nbc; i++)
        {
            //printf("after[%d] %d eci: %d\n; ", i, sh[i], eci[i]);
            printf("sh[%d] %d si: %d\n ", i, sh[i], si[i]);
        }

#endif

        timers["bitonic"]->stop();
    }
	//----------------------------------------------------------------------

}; //end namespace

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


#define CLOUD_COLLISION 1

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

	//----------------------------------------------------------------------
    CLOUD::CLOUD(RTPS *psfr, SPHParams& sphp_, Buffer<GridParams>* cl_GridParams, Buffer<GridParams>* cl_GridParamsScaled, GridParams* grid_params, GridParams* grid_params_scaled, int max_nb_in_cloud) 
    {
		this->cl_GridParams = cl_GridParams;
		this->cl_GridParamsScaled = cl_GridParamsScaled;
		this->grid_params   = grid_params;
		this->grid_params_scaled   = grid_params_scaled;

		this->sphp = &sphp_;

        //store the particle system framework
        ps = psfr;
        settings = ps->settings;

		cloud_max_num = max_nb_in_cloud; // remove max_outer_num
		cloud_num = 0;
		// max_outer_particles defined in RTPSettings (used?)

		// I should be able to not specify this, but GPU restrictions ...
        resource_path = settings->GetSettingAs<string>("rtps_path");
        printf("resource path: %s\n", resource_path.c_str());

        //seed random
        srand ( time(NULL) );

		std::vector<CLOUDParams> vcparams(0);
		vcparams.push_back(cloudp);
		cl_cloudp = Buffer<CLOUDParams>(ps->cli, vcparams);

        spacing = settings->GetSettingAs<float>("Spacing");

		setupRigidBodyKinematics();

        setupTimers();
		setupStages();

		addCloud();

		cloud_omega = float4(10.,10.,100.,0.);
		cloud_cg    = float4(1.5, 2.5, 2.5, 0.);
		//cloud_cg = cloud_cg * sphp->simulation_scale;

		//printf("cloud_num = %d\n", cloud_num); exit(0);
        settings->SetSetting("Maximum Number of Cloud Particles", cloud_max_num);

		#if 0
		float4 f = float4(2.,3.,4.,5);
		float* ff = (float*) &f;
		printf("f = %f, %f %f, %f\n", ff[0], ff[1], ff[2], ff[3]);
		exit(0);
		#endif
    }

	//----------------------------------------------------------------------
    CLOUD::~CLOUD()
    {
    }

	//----------------------------------------------------------------------
	void CLOUD::permuteExecute()
	{
            cloud_permute.execute(
			    cloud_num,
                cl_position_u,
                cl_position_s,
                cl_normal_u, 
                cl_normal_s,
                cl_sort_indices,
                *cl_GridParams,
                clf_debug,
                cli_debug);
	}
	//----------------------------------------------------------------------
	void CLOUD::cellindicesExecute()
	{
		cellindices.execute(cloud_num,
                cl_sort_hashes,
                cl_sort_indices,
                cl_cell_indices_start,
                cl_cell_indices_end,
                //cl_sphp,
                *cl_GridParams,
                grid_params->nb_cells,
                clf_debug,
                cli_debug);
	}
	//----------------------------------------------------------------------
    void CLOUD::cloudVelocityExecute()
	{
		//printf("**** BEFORE velocity execute *****\n");
		//u.printDevArray(cl_position_s, "pos_s", 10, 10);
		//u.printDevArray(cl_velocity_s, "vel_s", 10, 10);

		float angle = 0.; // not used at this time

		velocity.execute(
					cloud_num,
                    settings->dt,  // should be time, not dt
                    angle,  // should be time, not dt
					cl_position_s,
					cl_velocity_s,
                    cloud_cg,
                    cloud_omega);

		//printf("**** AFTER velocity execute *****\n");
		//u.printDevArray(cl_position_s, "pos_s", 10, 10);
		//u.printDevArray(cl_velocity_s, "vel_s", 10, 10);
	}

	//----------------------------------------------------------------------
    void CLOUD::cloud_hash_and_sort()
    {
        timers["hash"]->start();
        hash.execute( cloud_num,
                cl_position_u,
                cl_sort_hashes,
                cl_sort_indices,
                *cl_GridParams,
                clf_debug,
                cli_debug);
        timers["hash"]->stop();

        cloud_bitonic_sort(); 
    }
	//----------------------------------------------------------------------
    void CLOUD::collision(Buffer<float4>& cl_sph_pos_s, Buffer<float4>& cl_sph_vel_s, 
	          Buffer<float4>& cl_sph_force_s, Buffer<SPHParams>& cl_sphp, int num_sph)
    {
		// NEED TIMER FOR POINT CLOUD COLLISIONS (GE)

		collision_cloud.execute(num_sph, cloud_num, 
			cl_sph_pos_s, 
			cl_sph_vel_s,  
			cl_position_s, 
			cl_normal_s,
			cl_velocity_s,
			cl_sph_force_s, // output

			cl_cell_indices_start,
			cl_cell_indices_end,

			cl_sphp,    // IS THIS CORRECT?
			//*cl_GridParams,
			*cl_GridParamsScaled,
			// debug
			clf_debug,
			cli_debug);
    }
	//----------------------------------------------------------------------

    void CLOUD::integrate()
    {
        timers["integrate"]->start();

		static int count=0;

		// CLOUD INTEGRATION

		// start the arm moving after x iterations
		if (count > 10) {

			// How to prevent the cloud from advecting INTO THE FLUID? 
			//printf("cloud euler, cloud_num= %d\n", cloud_num);
			// returns unsorted positions
            cloud_euler.execute(cloud_num,
                settings->dt,
                cl_position_u,
                cl_position_s,
                cl_normal_u,
                cl_normal_s,
                cl_velocity_u,
                cl_velocity_s,
                cl_sort_indices,
                *cl_sphp,
                //debug
                clf_debug,
                cli_debug);

		}
		count++;

		cl_position_u.copyToHost(cloud_positions);
		cl_normal_u.copyToHost(cloud_normals);

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
		std::fill(cloud_velocity.begin(), cloud_velocity.end(), avg_velocity);
		// END CLOUD

        //for reading back different values from the kernel
        //std::vector<float4> error_check(max_num);
        
		//grid_params->grid_max.print("xxx"); exit(0);
        float4 pmax = grid_params->grid_max + grid_params->grid_size;

        //std::fill(error_check.begin(), error_check.end(), float4(0.0f, 0.0f, 0.0f, 0.0f));

		printf("cloud prepare\n");

		//CLOUD BUFFERS
		if (cloud_max_num > 0) {
        	cl_position_u = Buffer<float4>(ps->cli, cloud_positions);
        	cl_position_s = Buffer<float4>(ps->cli, cloud_positions);
        	cl_normal_u   = Buffer<float4>(ps->cli, cloud_normals);
        	cl_normal_s   = Buffer<float4>(ps->cli, cloud_normals);
        	cl_velocity_u = Buffer<float4>(ps->cli, cloud_velocity);
        	cl_velocity_s = Buffer<float4>(ps->cli, cloud_velocity);
		}

		//cl_position_u.copyToHost(cloud_positions); printf("xx\n");exit(1);

        //setup debug arrays
        std::vector<float4> clfv(cloud_max_num);
        std::fill(clfv.begin(), clfv.end(),float4(0.0f, 0.0f, 0.0f, 0.0f));
        std::vector<int4> cliv(cloud_max_num);
        std::fill(cliv.begin(), cliv.end(),int4(0.0f, 0.0f, 0.0f, 0.0f));
        clf_debug = Buffer<float4>(ps->cli, clfv);
        cli_debug = Buffer<int4>(ps->cli, cliv);

        std::vector<unsigned int> cloud_keys(cloud_max_num);
        //to get around limits of bitonic sort only handling powers of 2

// DANGEROUS because could contain classes, etc.
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
        	cl_cell_indices_start  = Buffer<unsigned int>(ps->cli, gcells);
        	cl_cell_indices_end    = Buffer<unsigned int>(ps->cli, gcells);
        	cl_sort_indices        = Buffer<unsigned int>(ps->cli, cloud_keys);
        	cl_sort_hashes         = Buffer<unsigned int>(ps->cli, cloud_keys);
        	cl_sort_output_hashes  = Buffer<unsigned int>(ps->cli, cloud_keys);
        	cl_sort_output_indices = Buffer<unsigned int>(ps->cli, cloud_keys);
		}

		//printf("keys.size= %d\n", keys.size()); // 
		printf("cloud_keys.size= %d\n", cloud_keys.size()); // 8192
		printf("gcells.size= %d\n", gcells.size()); // 1729
		//exit(0);
     }
	 //----------------------------------------------------------------------
	void CLOUD::pushCloudParticles(vector<float4>& pos, vector<float4>& normals)
	{
		if ((cloud_num+pos.size()) > cloud_max_num) {
			printf("exceeded max number of cloud particles\n");
			exit(2); //GE
			return;
		}

        cl_position_u.copyToDevice(pos, cloud_num);
        cl_normal_u.copyToDevice(normals, cloud_num);

		cloud_num += pos.size();

		return;
	}
	//----------------------------------------------------------------------
	void CLOUD::readPointCloud(std::vector<float4>& cloud_positions, 
							   std::vector<float4>& cloud_normals,
							   std::vector<int4>&   cloud_faces,
							   std::vector<int4>&   cloud_faces_normals)
	{
		// REMOVE HARDCODING!!!
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

        vector<float4> sphere = addHollowSphere(nn, center, radius_in, radius_out, spacing/2., scale, normals);
        float4 velo(0, 0, 0, 0);
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

            //int arrayLength = nlpo2(cloud_num);
            int arrayLength = cloud_max_num;
            int batch = 1;

            //printf("about to try sorting\n");
            bitonic.Sort(batch, 
                        arrayLength,  //GE?? ???
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

		// NOT SURE HOW THIS WORKS!! GE
        cl_sort_hashes.copyFromBuffer(cl_sort_output_hashes, 0, 0, cloud_num);
        cl_sort_indices.copyFromBuffer(cl_sort_output_indices, 0, 0, cloud_num);

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

        timers["bitonic"]->stop();
    }
	//----------------------------------------------------------------------
    void CLOUD::updateCLOUDP()
	{
		cloudp.num = cloud_num; //settings->GetSettingAs<int>("Number of Cloud Particles");
		cloudp.max_num = cloud_max_num; // settings->GetSettingAs<int>("Maximum Number of Cloud Particles");
	}
	//----------------------------------------------------------------------
	void CLOUD::addCloud()
	{
		// should bounce off 
		float radius_in = 2.5;
		float radius_out = radius_in + .5;
		float4 center(1.6,1.6,2.7-radius_out, 0.0);
		//vector<float4> cloud_normals;
		bool scaled = true;

		// must be called after prepareSorted
		center = float4(2.5, 2.5, 0., 0.0);
		//center = float4(5.0, 2.5, 2.5, 0.0);
		//addHollowBall(2000, center, radius_in, radius_out, scaled, cloud_normals);
		//int nn = 4000;
    	//addNewxyPlane(nn, scaled, cloud_normals); 
		readPointCloud(cloud_positions, cloud_normals, cloud_faces, cloud_faces_normals);

        //calculate();
        updateCLOUDP(); cloudp.print(); // nb points corect

		//  ADD A SWITCH TO HANDLE CLOUD IF PRESENT
		// Must be called after a point cloud has been created. 
		if (cloud_num > 0) {
			collision_cloud = CollisionCloud(sph_source_dir, ps->cli, timers["ct_pgu"], cloud_max_num); 
		}


		cloud_positions.resize(cloud_positions.capacity());
		cloud_normals.resize(cloud_normals.capacity());
		// only needs to be done once if cloud not moving
		// ideally, cloud should be stored in vbos. 
        cl_position_u.copyToHost(cloud_positions);
	}
	//----------------------------------------------------------------------
	void CLOUD::setupRigidBodyKinematics()
	{
		// Ideally, change every time step, and update all points in 
		// the cloud on the GPU (in which routine?)
		avg_velocity = float4(3., 0., 0., 1.);
	}
	//----------------------------------------------------------------------
	void CLOUD::setupStages()
	{
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

		// Kernel errors when using CloudBitonic. Do not know why!
        bitonic = Bitonic<unsigned int>(common_source_dir, ps->cli );
        //bitonic = CloudBitonic<unsigned int>(common_source_dir, ps->cli );

        cellindices = CellIndices(common_source_dir, ps->cli, timers["ci_gpu"] );
        //permute = Permute( common_source_dir, ps->cli, timers["perm_gpu"] );
        cloud_permute = CloudPermute( common_source_dir, ps->cli, timers["perm_gpu"] );

		// ERROR in timer GE
		velocity = CloudVelocity( sph_source_dir, ps->cli, timers["per_gpu"] );

		// CLOUD Integrator
		// ADD Cloud timers later. 
		cloud_euler = CloudEuler(sph_source_dir, ps->cli, timers["euler_gpu"]);
#endif
	}

	//----------------------------------------------------------------------


}; //end namespace

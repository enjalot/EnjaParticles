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


#ifndef RTPS_CLOUD_H_INCLUDED
#define RTPS_CLOUD_H_INCLUDED

#ifdef WIN32
#define _USE_MATH_DEFINES
#include <cmath>
#endif

#include <string>

#include <RTPS.h>
#include <System.h>
#include <Kernel.h>
#include <opencl/Buffer.h>

#include <Domain.h>
//#include <CLOUDSettings.h>
#include <SPHSettings.h>

#include <Matrix.h>

#include <util.h>


//class OUTER;


#include <Hash.h>
#include <CloudBitonicSort.h>
#include <BitonicSort.h>
#include <CellIndices.h>
#include <CloudPermute.h> // contains CloudPermute
#include <CloudVelocity.h> 

#include <Permute.h> // contains CloudPermute
#include <sph/CloudEuler.h>

// sphp required
#include <sph/Collision_cloud.h>

//#include <timege.h>
#include <timer_eb.h>

#ifdef WIN32
    #if defined(rtps_EXPORTS)
        #define RTPS_EXPORT __declspec(dllexport)
    #else
        #define RTPS_EXPORT __declspec(dllimport)
	#endif 
#else
    #define RTPS_EXPORT
#endif

namespace rtps
{
    //using namespace sph;

	typedef struct CLOUDParams
    //Struct which gets passed to OpenCL routines
	{
		int num; // nb cloud points
		int max_num; // max nb cloud points
		void print() {
			printf("---- CLOUDParams----\n");
			printf("nb points: %d\n", num);
			printf("max nb points: %d\n", max_num);
		}
	}
#ifndef WIN32
	__attribute__((aligned(16)));
#else
		;
        #pragma pack(pop)
#endif



    //class RTPS_EXPORT CLOUD : public System
    class RTPS_EXPORT CLOUD 
    {
    public:
    	CLOUD(RTPS *psfr, SPHParams& sphp, Buffer<GridParams>* cl_GridParams, 
		    Buffer<GridParams>* cl_GridParamsScaled, 
			GridParams* grid_params, GridParams* grid_params_scaled, 
			int max_nb_in_cloud);

        ~CLOUD();



		//**** ROUTINES TO ADD CLOUDS *******

		// Generation of special clouds. Should really be in a 
		// Cloud generation class: CloudGeneration

    	void addHollowBall(int nn, float4 center, float radius_in, float radius_out, bool scaled, std::vector<float4>& normals);
        void addNewxyPlane(int np, bool scaled, vector<float4>& normals);

		void readPointCloud(std::vector<float4>& cloud_positions, 
							std::vector<float4>& cloud_normals,
						 	std::vector<int4>& cloud_faces,
						 	std::vector<int4>& cloud_faces_normals);



		//**** TIMERS *******
        EB::TimerList timers;
        int setupTimers();
        void printTimers();

    protected:
        //virtual void setRenderer();

    private:
        //the particle system framework
        RTPS* ps;
        RTPSettings* settings;

		CLOUDParams cloudp;
		SPHParams* sphp;

        GridParams* 		grid_params;
        GridParams* 		grid_params_scaled;
        float spacing; //Particle rest distance in world coordinates

        std::string sph_source_dir;

        Hash hash;
        CellIndices cellindices;
        CloudEuler cloud_euler;
        CloudPermute cloud_permute; // for generality, keep separate (GE)
		CloudVelocity velocity;
        CollisionCloud collision_cloud;

        //needs to be called when particles are added
        void prepareSorted();

		// POINT CLOUD ARRAYS
        std::vector<float4> cloud_positions;
        std::vector<float4> cloud_normals;
        std::vector<float4> cloud_velocity;
        std::vector<int4>   cloud_faces;
        std::vector<int4>   cloud_faces_normals;

		// ideally, the total point cloud is a collection of rigid
		// objects. For now, only a single rigid object
		float4				cloud_cg; // center of gravity
		float4				avg_velocity;
		float4				avg_angular_momentum;
		// quaternion (theta/2, rotation axis)
		float4				cloud_omega;  // rotation quaterion
		float 				cloud_rot_mat[3][3];

		Buffer<float4>		cl_position_u;
		Buffer<float4>		cl_position_s;
		Buffer<float4>		cl_velocity_u;
		Buffer<float4>		cl_velocity_s;
		Buffer<float4>		cl_normal_u;
		Buffer<float4>		cl_normal_s; // normalized for now
        Buffer<unsigned int>         cl_cell_indices_start;
        Buffer<unsigned int>         cl_cell_indices_end;
        Buffer<unsigned int>         cl_sort_hashes;
        Buffer<unsigned int>         cl_sort_indices;
        //Two arrays for bitonic sort (sort not done in place)
        //should be moved to within bitonic
        Buffer<unsigned int>         cl_sort_output_hashes;
        Buffer<unsigned int>         cl_sort_output_indices;

        //CloudBitonic<unsigned int> bitonic;
        Bitonic<unsigned int> bitonic;

        //Parameter structs
        Buffer<SPHParams>*    cl_sphp;
        Buffer<CLOUDParams>   cl_cloudp;
        Buffer<GridParams>*   cl_GridParams;
        Buffer<GridParams>*   cl_GridParamsScaled;

        Buffer<float4>        clf_debug;  //just for debugging cl files
        Buffer<int4>          cli_debug;  //just for debugging cl files

        //calculate the various parameters that depend on max_num of particles
        //void calculate();
        //copy the CLOUD parameter struct to the GPU
		void pushCloudParticles(vector<float4>& pos, vector<float4>& normals);

        //Permute permute;
        void hash_and_sort();
        void bitonic_sort();
        void cloud_bitonic_sort();   // GE


        std::string resource_path;
        std::string common_source_dir;

        int cloud_max_num;
        int cloud_num;  // USED FOR WHAT? 

		// GE
		vector<float4>& getCloudPoints() { return cloud_positions; }
		vector<float4>& getCloudNormals() { return cloud_normals; }

		Utils u;  // for debugging etc.

public:
    	void updateCLOUDP();
        void integrate();
    	void collision(Buffer<float4>& cl_pos_s, Buffer<float4>& cl_vel_s, 
	          Buffer<float4>& cl_force_s, Buffer<SPHParams>& cl_sphp, int num_sph);
        void cloud_hash_and_sort();  // GE
    	void cloudVelocityExecute();
		void cellindicesExecute();
		void permuteExecute();
		int setRenderer(Render* renderer) {
			this->renderer = renderer;
			printf("renderer= %ld, cloud_num = %d\n", renderer, cloud_num);
			this->renderer->setCloudData(cloud_positions, cloud_normals, cloud_faces, cloud_faces_normals, cloud_num);
		}
		void setSPHP(Buffer<SPHParams>* cl_sphp) {
			this->cl_sphp = cl_sphp;
		}
		void setGridParams(Buffer<GridParams>* cl_GridParams, GridParams* grid_params) {
			this->cl_GridParams = cl_GridParams;
			this->grid_params = grid_params;
		}

		void addCloud();
		void setupStages();
		void setupRigidBodyKinematics();

		int getCloudNum() { return cloud_num; }
		void setCloudNum(int cloud_num) { this->cloud_num = cloud_num; }
		int getMaxCloudNum() { return cloud_max_num; }

private:
		Render* renderer;
    };
}; // namespace
 
#endif

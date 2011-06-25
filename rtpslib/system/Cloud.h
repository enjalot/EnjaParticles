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
#include <Buffer.h>

#include <Domain.h>
//#include <CLOUDSettings.h>
#include <SPHSettings.h>


//class OUTER;


#include <Hash.h>
#include <CloudBitonicSort.h>
#include <CellIndices.h>
#include <CloudPermute.h> // contains CloudPermute

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

    //class RTPS_EXPORT CLOUD : public System
    class RTPS_EXPORT CLOUD 
    {
    public:
        //CLOUD(RTPS *ps, SPHParams& sphp, int nb_in_cloud=0);
    	CLOUD(RTPS *psfr, SPHParams& sphp, Buffer<GridParams>* cl_GridParams, GridParams* grid_params, int max_nb_in_cloud);

        ~CLOUD();

		// advance particles one iteration
        void update();

        //wrapper around IV.h addRect
        int addBox(int nn, float4 min, float4 max, bool scaled, float4 color=float4(1.0f, 0.0f, 0.0f, 1.0f));
        //wrapper around IV.h addSphere

		// Generation of special clouds. Should really be in a 
		// Cloud generation class: CloudGeneration

        void addBall(int nn, float4 center, float radius, bool scaled);
    	void addHollowBall(int nn, float4 center, float radius_in, float radius_out, bool scaled, std::vector<float4>& normals);
        void addNewxyPlane(int np, bool scaled, vector<float4>& normals);

		void readPointCloud(std::vector<float4>& cloud_positions, 
							std::vector<float4>& cloud_normals,
						 	std::vector<int4>& cloud_faces,
						 	std::vector<int4>& cloud_faces_normals);

        EB::TimerList timers;
        int setupTimers();
        void printTimers();
        void pushParticles(vector<float4> pos, float4 velo, float4 color=float4(1.0, 0.0, 0.0, 1.0));
        void pushParticles(vector<float4> pos, vector<float4> velo, float4 color=float4(1.0, 0.0, 0.0, 1.0));

        std::vector<float4> getDeletedPos();
        std::vector<float4> getDeletedVel();

    protected:
        //virtual void setRenderer();
    private:
        //the particle system framework
        RTPS* ps;
        RTPSettings* settings;

		CLOUDParams cloudp;
        GridParams* grid_params;
        GridParams grid_params_scaled;
        float spacing; //Particle rest distance in world coordinates

        std::string sph_source_dir;

        //needs to be called when particles are added
        void calculateCLOUDSettings();
        //void setupDomain();
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
		float4				avg_cloud_velocity;
		float4				avg_cloud_angular_momentum;
		// quaternion (theta/2, rotation axis)
		float4				cloud_omega;  // rotation quaterion
		float 				cloud_rot_mat[3][3];
		Buffer<float4>		cl_cloud_position_u;
		Buffer<float4>		cl_cloud_position_s;
		Buffer<float4>		cl_cloud_velocity_u;
		Buffer<float4>		cl_cloud_velocity_s;
		Buffer<float4>		cl_cloud_normal_u;
		Buffer<float4>		cl_cloud_normal_s; // normalized for now
        Buffer<unsigned int>         cl_cloud_cell_indices_start;
        Buffer<unsigned int>         cl_cloud_cell_indices_end;
        Buffer<unsigned int>         cl_cloud_sort_hashes;
        Buffer<unsigned int>         cl_cloud_sort_indices;
        //Two arrays for bitonic sort (sort not done in place)
        //should be moved to within bitonic
        Buffer<unsigned int>         cl_cloud_sort_output_hashes;
        Buffer<unsigned int>         cl_cloud_sort_output_indices;

        CloudBitonic<unsigned int> bitonic;

        //Parameter structs
        Buffer<SPHParams>*   cl_sphp;
        Buffer<GridParams>*  cl_GridParams;
        Buffer<GridParams>  cl_GridParamsScaled;

		//Parameter structs for point cloud
        Buffer<CLOUDParams>   cl_cloudp;

        Buffer<float4>      clf_debug;  //just for debugging cl files
        Buffer<int4>        cli_debug;  //just for debugging cl files

		//SPHParams* sphp;

        void updateGPU();

        //calculate the various parameters that depend on max_num of particles
        void calculate();
        //copy the CLOUD parameter struct to the GPU
        void updateCLOUDP();
		void pushCloudParticles(vector<float4>& pos, vector<float4>& normals);

        Hash hash;
        CellIndices cellindices;
        Permute permute;
        CloudPermute cloud_permute; // for generality, keep separate (GE)
        void hash_and_sort();
        void bitonic_sort();
        void cloud_bitonic_sort();   // GE
        //void collision();
        CollisionCloud collision_cloud;
        CloudEuler cloud_euler;

        std::string resource_path;
        std::string common_source_dir;

        int cloud_max_num;
        int cloud_num;  // USED FOR WHAT? 

		// GE
		vector<float4>& getCloudPoints() { return cloud_positions; }
		vector<float4>& getCloudNormals() { return cloud_normals; }

		int nb_in_cloud; // nb of points in cloud

		void printDevArray(Buffer<float4>& cl_array, char* msg, int nb_el, int nb_print);
		void printDevArray(Buffer<float>& cl_array, char* msg, int nb_el, int nb_print);
		void printDevArray(Buffer<int4>& cl_array, char* msg, int nb_el, int nb_print);
		void printDevArray(Buffer<int>& cl_array, char* msg, int nb_el, int nb_print);

public:
        void integrate();
    	void collision(Buffer<float4>& cl_pos_s, Buffer<float4>& cl_vel_s, 
	          Buffer<float4>& cl_force_s, Buffer<SPHParams>& cl_sphp, int num_sph);
        void cloud_hash_and_sort();  // GE
		void cellindicesExecute();
		void permuteExecute();
		int getCloudNum() { return cloud_num; }
		void setCloudNum(int cloud_num) { this->cloud_num = cloud_num; }
		int getMaxCloudNum() { return cloud_max_num; }
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

		#if 0
		void setSPHArrrays(
				cl_position_s, 
				cl_velocity_s,  
				cl_force_s, // output
		#endif

private:
		Render* renderer;
    };
}; // namespace
 
#endif

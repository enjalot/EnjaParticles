#ifndef RTPS_FLOCK_H_INCLUDED
#define RTPS_FLOCK_H_INCLUDED

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
#include <FLOCKSettings.h>


#include <flock/Prep.h>
#include <Hash.h>
#include <BitonicSort.h>
//#include <DataStructures.h>
#include <CellIndices.h>
#include <Permute.h>
//#include <Density.h>
//#include <Force.h>
//#include <Collision_wall.h>
//#include <Collision_triangle.h>
//#include <LeapFrog.h>
//#include <Lifetime.h>
#include <flock/ComputeRules.h>
#include <flock/AverageRules.h>

//#include "../util.h"
#include <Hose.h>

//#include <timege.h>
#include <timer_eb.h>

#include "../rtps_common.h"

// Added by GE, March 16, 2011
#include "boids.h"


namespace rtps
{
using namespace flock;

//----------------------------------------------------------------------
//keep track of the flock settings
/*typedef struct FLOCKSettings
{
    float simulation_scale;
    float particle_rest_distance;
    float smoothing_distance;
    float spacing;
} FLOCKSettings;

//----------------------------------------------------------------------
//pass parameters to OpenCL routines
#ifdef WIN32
#pragma pack(push,16)
#endif
typedef struct FLOCKParameters
{

    float4 grid_min;
    float4 grid_max;
    
    float rest_distance;
    float smoothing_distance;
    
    int num;
    int nb_vars;    // for combined variables (vars_sorted, etc.)
	int choice;     // which kind of calculation to invoke
    
    // Boids
    float min_dist;  // desired separation between boids
    float search_radius;
    float max_speed; 
    
    float w_sep;
    float w_align;
    float w_coh;

    void print() {
		printf("----- FLOCKParams ----\n");
		printf("min_dist: %f\n", min_dist);
		printf("search_radius: %f\n", search_radius);
		printf("max_speed: %f\n", max_speed);
	}
} FLOCKParameters
#ifndef WIN32
    __attribute__((aligned(16)));
#else
    ;
#pragma pack(pop,16)
#endif 
*/
//----------------------------------------------------------------------

class RTPS_EXPORT FLOCK : public System
{


public:
    FLOCK(RTPS *ps, int num);
    ~FLOCK();

    // update call for CPU and GPU
    void update();
    
    //wrapper around IV.h addRect
    int addBox(int nn, float4 min, float4 max, bool scaled, float4 color=float4(1., 0., 0., 1.));
    
    //wrapper around IV.h addSphere
    void addBall(int nn, float4 center, float radius, bool scaled);

    //wrapper around Hose.h 
    int addHose(int total_n, float4 center, float4 velocity, float radius, float4 color=float4(1.0, 0.0, 0.0, 1.0f));
    void updateHose(int index, float4 center, float4 velocity, float radius, float4 color=float4(1.0, 0.0, 0.0, 1.0f));
    void sprayHoses();


    virtual void render();
    
    void testDelete();
    
    // timers
    /*enum {
            TI_HASH=0, TI_BITONIC_SORT, TI_BUILD, TI_NEIGH, 
            TI_DENS, TI_FORCE, TI_EULER, TI_LEAPFROG, TI_UPDATE, TI_COLLISION_WALL,
            TI_COLLISION_TRI
         }; 
    GE::Time* timers[30];*/
    EB::TimerList timers;
    int setupTimers();
    void printTimers();

    void pushParticles(vector<float4> pos, float4 velo, float4 color=float4(1.0, 0.0, 0.0, 1.0));
    void pushParticles(vector<float4> pos, vector<float4> velo, float4 color=float4(1.0, 0.0, 0.0, 1.0));

protected:
    virtual void setRenderer();
    
private:
    //the particle system framework
    RTPS *ps;
    RTPSettings *settings;

	//Boids *boids;

    //FLOCKSettings   flock_settings;
    FLOCKParameters flock_params;
    GridParams      grid_params;
    GridParams      grid_params_scaled;
    float spacing; //Particle rest distance in world coordinates

    int nb_var;

    std::vector<float4> deleted_pos;
    std::vector<float4> deleted_vel;
    
    //keep track of hoses
    std::vector<Hose> hoses;   
    
    //needs to be called when particles are added
    void calculateFLOCKSettings();
    void setupDomain();
    void prepareSorted();
    
    //void pushParticles(vector<float4> pos);
    //void pushParticles(vector<float4> pos, float4 velo, float4 color=float4(1.0, 0.0, 0.0, 1.0));
    //void pushParticles(vector<float4> pos, vector<float4> velo, float4 color=float4(1.0, 0.0, 0.0, 1.0));

    // kernels
    //Kernel k_euler; 

    // kernels - neighbors
    //Kernel k_prep;
    //Kernel k_hash;
    //Kernel k_datastructures;
    //Kernel k_neighbors;

    //This should be in OpenCL classes
    Kernel k_scopy;

    std::vector<float4> positions;
    std::vector<float4> colors;
    std::vector<float4> velocities;
    std::vector<float4> veleval;
    
    //std::vector<float>  densities;
    //std::vector<float4> forces;
    //std::vector<float4> xflocks;
    std::vector<float4> separation;
    std::vector<float4> alignment;
    std::vector<float4> cohesion;

    Buffer<float4>      cl_position_u;
    Buffer<float4>      cl_position_s;
    Buffer<float4>      cl_color_u;
    Buffer<float4>      cl_color_s;
    Buffer<float4>      cl_velocity_u;
    Buffer<float4>      cl_velocity_s;
    Buffer<float4>      cl_veleval_u;
    Buffer<float4>      cl_veleval_s;
    
    //Buffer<float>       cl_density_s;
    //Buffer<float4>      cl_force_s;
    //Buffer<float4>      cl_xflock_s;
    Buffer<float4>      cl_separation_s;
    Buffer<float4>      cl_alignment_s;
    Buffer<float4>      cl_cohesion_s;

    //Neighbor Search related arrays
	//Buffer<float4> 	    cl_vars_sorted;
	//Buffer<float4> 	    cl_vars_unsorted;
	//Buffer<float4>   	cl_cells; // positions in Ian code
	Buffer<unsigned int> 		cl_cell_indices_start;
	Buffer<unsigned int> 		cl_cell_indices_end;
	//Buffer<int> 		cl_vars_sort_indices;
	Buffer<unsigned int> 		cl_sort_hashes;
	Buffer<unsigned int> 		cl_sort_indices;
	//Buffer<int> 		cl_unsort;
	//Buffer<int> 		cl_sort;

	
    //Two arrays for bitonic sort (sort not done in place)
	Buffer<unsigned int>         cl_sort_output_hashes;
	Buffer<unsigned int>         cl_sort_output_indices;

    Bitonic<unsigned int>        bitonic;
    
    //Parameter structs
    Buffer<FLOCKParameters>     cl_FLOCKParameters;
	Buffer<GridParams>          cl_GridParams;
	Buffer<GridParams>          cl_GridParamsScaled;
   
    Buffer<float4>  	clf_debug;  //just for debugging cl files
	Buffer<int4>		cli_debug;  //just for debugging cl files
    
    //index neighbors. Maximum of 50
	//Buffer<int> 		cl_index_neigh;
	

    
    //still in use?
    //Buffer<float4> cl_error_check;

    //these are defined in flock/ folder 
    //void loadEuler();
	//void ge_loadEuler();

    //Nearest Neighbors search related kernels
    //void loadPrep();
    //void loadHash();
    //void loadBitonicSort();
    //void loadDataStructures();
    //void loadNeighbors();

    //CPU functions
    void cpuComputeRules();
    void cpuAverageRules();
	//void ge_cpuEuler();

    void updateCPU();
    void updateGPU();

    // calculate the various parameters that depend on max_num of particles
    void calculate();

    //copy the FLOCK  parameter struct to the GPU
    void updateFLOCKP();

    //Nearest Neighbors search related functions
    Prep prep;
    void call_prep(int stage);
    Hash hash;
    //DataStructures datastructures;
    CellIndices cellindices;
    Permute permute;
    void hash_and_sort();
    void bitonic_sort();
    //Density density;
    //Force force;
    //void collision();
    //CollisionWall collision_wall;
    //CollisionTriangle collision_tri;
    ComputeRules computeRules;
    AverageRules averageRules;
    //void integrate();
    //LeapFrog leapfrog;
    //Euler euler;
    
    void integrate();

    //OpenCL helper functions, should probably be part of the OpenCL classes
    void loadScopy();
	void scopy(int n, cl_mem xsrc, cl_mem ydst); 


};

}

#endif

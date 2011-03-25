#ifndef RTPS_FLOCK_H_INCLUDED
#define RTPS_FLOCK_H_INCLUDED

#include <string>

#include "../RTPS.h"
#include "System.h"
#include "../opencl/Kernel.h"
#include "../opencl/Buffer.h"

// Added by GE, March 16, 2011
#include "boids.h"

#include "BitonicSort.h"

//#include "../util.h"
#include "../domain/Domain.h"

#include "timege.h"
#ifdef WIN32
    #if defined(rtps_EXPORTS)
        #define RTPS_EXPORT __declspec(dllexport)
    #else
        #define RTPS_EXPORT __declspec(dllimport)
	#endif 
#else
    #define RTPS_EXPORT
#endif


namespace rtps {


//----------------------------------------------------------------------
//keep track of the flock settings
typedef struct FLOCKSettings
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

//----------------------------------------------------------------------
class RTPS_EXPORT FLOCK : public System
{
public:
    FLOCK(RTPS *ps, int num);
    ~FLOCK();

    // update call for CPU and GPU
    void update();
    
    //wrapper around IV.h addRect
    int addBox(int nn, float4 min, float4 max, bool scaled);
    
    //wrapper around IV.h addSphere
    void addBall(int nn, float4 center, float radius, bool scaled);
	
    virtual void render();
    
    // timers
    enum {
            TI_HASH=0, TI_BITONIC_SORT, TI_BUILD, TI_NEIGH, 
            TI_DENS, TI_FORCE, TI_EULER, TI_LEAPFROG, TI_UPDATE, TI_COLLISION_WALL,
            TI_COLLISION_TRI
         }; 
    GE::Time* timers[30];
    int setupTimers();
    void printTimers();

protected:
    virtual void setRenderer();
    
private:
    //the particle system framework
    RTPS *ps;
	//Boids *boids;

    FLOCKSettings   flock_settings;
    FLOCKParameters flock_params;
    GridParams      grid_params;
    GridParams      grid_params_scaled;

    int nb_var;

    //needs to be called when particles are added
    void calculateFLOCKSettings();
    void setupDomain();
    void prepareSorted();
    void pushParticles(vector<float4> pos);

    // kernels
    Kernel k_euler; 

    // kernels - neighbors
    Kernel k_prep;
    Kernel k_hash;
    Kernel k_datastructures;
    Kernel k_neighbors;

    //This should be in OpenCL classes
    Kernel k_scopy;

    std::vector<float4> positions;
    std::vector<float4> colors;
    std::vector<float>  densities;
    std::vector<float4> forces;
    std::vector<float4> velocities;
    std::vector<float4> veleval;
    std::vector<float4> xflocks;

    Buffer<float4>      cl_position;
    Buffer<float4>      cl_color;
    Buffer<float>       cl_density;
    Buffer<float4>      cl_force;
    Buffer<float4>      cl_velocity;
    Buffer<float4>      cl_veleval;
    Buffer<float4>      cl_xflock;

    //Neighbor Search related arrays
	Buffer<float4> 	    cl_vars_sorted;
	Buffer<float4> 	    cl_vars_unsorted;
	Buffer<float4>   	cl_cells; // positions in Ian code
	Buffer<int> 		cl_cell_indices_start;
	Buffer<int> 		cl_cell_indices_end;
	Buffer<int> 		cl_vars_sort_indices;
	Buffer<int> 		cl_sort_hashes;
	Buffer<int> 		cl_sort_indices;
	Buffer<int> 		cl_unsort;
	Buffer<int> 		cl_sort;

	
    //Two arrays for bitonic sort (sort not done in place)
	Buffer<int>         cl_sort_output_hashes;
	Buffer<int>         cl_sort_output_indices;

    Bitonic<int>        bitonic;
    
    //Parameter structs
    Buffer<FLOCKParameters>     cl_FLOCKParameters;
	Buffer<GridParams>          cl_GridParams;
	Buffer<GridParams>          cl_GridParamsScaled;
   
    //index neighbors. Maximum of 50
	Buffer<int> 		cl_index_neigh;
	
    Buffer<float4>  	clf_debug;  //just for debugging cl files
	Buffer<int4>		cli_debug;  //just for debugging cl files
    
    //still in use?
    Buffer<float4> cl_error_check;

    //these are defined in flock/ folder 
    void loadEuler();
	void ge_loadEuler();

    //Nearest Neighbors search related kernels
    void loadPrep();
    void loadHash();
    void loadBitonicSort();
    void loadDataStructures();
    void loadNeighbors();

    //CPU functions
    void cpuEuler();
	void ge_cpuEuler();

    void updateCPU();
    void updateGPU();

    //copy the SPH parameter struct to the GPU
    void updateFLOCKP();

    //Nearest Neighbors search related functions
    void prep(int stage);
    void hash();
    void printHashDiagnostics();
    void bitonic_sort();
    void buildDataStructures();
    void neighborSearch(int choice);
    
    void integrate();

    //OpenCL helper functions, should probably be part of the OpenCL classes
    void loadScopy();
	void scopy(int n, cl_mem xsrc, cl_mem ydst); 
   
};

}

#endif

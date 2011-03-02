#ifndef RTPS_SPH_H_INCLUDED
#define RTPS_SPH_H_INCLUDED

#include <string>

#include <RTPS.h>
#include <System.h>
#include <Kernel.h>
#include <Buffer.h>

#include <BitonicSort.h>

//#include "../util.h"
#include <Domain.h>
#include <Hose.h>

#include <timege.h>


namespace rtps {

enum Integrator {EULER, LEAPFROG};

//pass parameters to OpenCL routines
typedef struct SPHParams
{
    float4 grid_min;
    float4 grid_max;
    float mass;
    float rest_distance;
    float smoothing_distance;
    float simulation_scale;
    float boundary_stiffness;
    float boundary_dampening;
    float boundary_distance;
    float EPSILON;
    float PI;       //delicious
    float K;        //gas constant
    float viscosity;
    float velocity_limit;
    float xsph_factor;



	float gravity; // -9.8 m/sec^2
    float friction_coef;
	float restitution_coef;
	float shear;
	float attraction;
	float spring;
	//float surface_threshold;

    
    //Kernel Coefficients
    float wpoly6_coef;
	float wpoly6_d_coef;
	float wpoly6_dd_coef; // laplacian
	float wspiky_coef;
	float wspiky_d_coef;
	float wspiky_dd_coef;
	float wvisc_coef;
	float wvisc_d_coef;
	float wvisc_dd_coef;

    int num;
    int nb_vars; // for combined variables (vars_sorted, etc.)
	int choice; // which kind of calculation to invoke
 
    
    void print() {
		printf("----- SPHParams ----\n");
		printf("simulation_scale: %f\n", simulation_scale);
		printf("friction_coef: %f\n", friction_coef);
		printf("restitution_coef: %f\n", restitution_coef);
		printf("damping: %f\n", boundary_dampening);
		printf("shear: %f\n", shear);
		printf("attraction: %f\n", attraction);
		printf("spring: %f\n", spring);
		printf("gravity: %f\n", gravity);
		printf("choice: %d\n", choice);
	}
} SPHParams __attribute__((aligned(16)));

//----------------------------------------------------------------------
// GORDON Datastructure for Fluid parameters.
// struct for fluid parameters
struct FluidParams
{
	float smoothing_length; // SPH radius
	float scale_to_simulation;
	//float mass;
	//float dt;
	float friction_coef;
	float restitution_coef;
	float damping;
	float shear;
	float attraction;
	float spring;
	float gravity; // -9.8 m/sec^2
	int   choice; // which kind of calculation to invoke

};



class SPH : public System
{
public:
    SPH(RTPS *ps, int num);
    ~SPH();

    void update();
    //wrapper around IV.h addRect
    int addBox(int nn, float4 min, float4 max, bool scaled);
    //wrapper around IV.h addSphere
    void addBall(int nn, float4 center, float radius, bool scaled);
    //wrapper around Hose.h 
    void addHose(int total_n, float4 center, float4 velocity, float radius, float spacing);
    void sprayHoses();

	virtual void render();
    
    void loadTriangles(std::vector<Triangle> triangles);
    


    enum {TI_HASH=0, TI_BITONIC_SORT, TI_BUILD, TI_NEIGH, 
          TI_DENS, TI_FORCE, TI_EULER, TI_LEAPFROG, TI_UPDATE, TI_COLLISION_WALL,
          TI_COLLISION_TRI
          }; //11
    GE::Time* timers[30];
    int setupTimers();
    void printTimers();


    void pushParticles(vector<float4> pos, float4 velo);
    
private:
    //the particle system framework
    RTPS *ps;

    SPHParams sphp;
    GridParams grid_params;
    GridParams grid_params_scaled;
    Integrator integrator;

    int nb_var;

    bool triangles_loaded; //keep track if we've loaded triangles yet
    
    //keep track of hoses
    std::vector<Hose> hoses;

    //needs to be called when particles are added
    void calculateSPHSettings();
    void setupDomain();
    void prepareSorted();
    //void popParticles();

    Kernel k_density, k_pressure, k_viscosity;
    Kernel k_collision_wall;
    Kernel k_collision_tri;
    Kernel k_euler, k_leapfrog;
    Kernel k_xsph;

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
    std::vector<float4> xsphs;

    Buffer<float4>      cl_position;
    Buffer<float4>      cl_color;
    Buffer<float>       cl_density;
    Buffer<float4>      cl_force;
    Buffer<float4>      cl_velocity;
    Buffer<float4>      cl_veleval;
    Buffer<float4>      cl_xsph;

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

    Buffer<Triangle>    cl_triangles;
	
    //Two arrays for bitonic sort (sort not done in place)
	Buffer<int>         cl_sort_output_hashes;
	Buffer<int>         cl_sort_output_indices;

    Bitonic<int> bitonic;
    
    //Parameter structs
    Buffer<SPHParams>   cl_SPHParams;
	Buffer<GridParams>  cl_GridParams;
	Buffer<GridParams>  cl_GridParamsScaled;
   
    //index neighbors. Maximum of 50
	Buffer<int> 		cl_index_neigh;
	
    Buffer<float4>  	clf_debug;  //just for debugging cl files
	Buffer<int4>		cli_debug;  //just for debugging cl files
    
    
    //still in use?
    Buffer<float4> cl_error_check;

    //these are defined in sph/ folder next to the kernels
    void loadDensity();
    void loadPressure();
    void loadViscosity();
    void loadXSPH();
    void loadCollision_wall();
    void loadCollision_tri();
    void loadEuler();
    void loadLeapFrog();

    //Nearest Neighbors search related kernels
    void loadPrep();
    void loadHash();
    void loadBitonicSort();
    void loadDataStructures();
    void loadNeighbors();

    //CPU functions
    void cpuDensity();
    void cpuPressure();
    void cpuViscosity();
    void cpuXSPH();
    void cpuCollision_wall();
    void cpuEuler();
    void cpuLeapFrog();

    void updateCPU();
    void updateGPU();

    //copy the SPH parameter struct to the GPU
    void SPH::updateSPHP();

    //Nearest Neighbors search related functions
    void prep(int stage);
    void hash();
    void printHashDiagnostics();
    void bitonic_sort();
    void buildDataStructures();
    void neighborSearch(int choice);
    void collision();
    void collide_triangles();
    void integrate();

    float Wpoly6(float4 r, float h);
    float Wspiky(float4 r, float h);
    float Wviscosity(float4 r, float h);

    //OpenCL helper functions, should probably be part of the OpenCL classes
    void loadScopy();
	void scopy(int n, cl_mem xsrc, cl_mem ydst); 
	
	//void sset_int(int n, int val, cl_mem xdst);
   
};



}

#endif

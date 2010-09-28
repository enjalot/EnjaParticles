#ifndef RTPS_GE_SPH_H_INCLUDED
#define RTPS_GE_SPH_H_INCLUDED

#include <string>

// in ge_datastructures/
// inlcludes RTPS, so recursion
//#include "datastructures.h"

#include "../RTPS.h"
#include "System.h"
#include "../opencl/Kernel.h"
#include "../opencl/Buffer.h"
#include "../opencl/BufferGE.h"
//#include "../opencl/BufferVBO.h"
//#include "../util.h"
#include "../particle/UniformGrid.h"

// Make sure it is same as in density.cl
#define DENS 0
#define POS 1
#define VEL 2
#define FOR 3

namespace rtps {

//class DataStructures;

//keep track of the fluid settings
typedef struct GE_SPHSettings
{
    float rest_density;
    float simulation_scale;
    float particle_mass;
    float particle_rest_distance;
    float smoothing_distance;
    float boundary_distance;
    float spacing;
    float grid_cell_size;

} GE_SPHSettings;

//-------------------------
// GORDON Datastructure for Grids. To be reconciled with Ian's
struct GridParams
{
    float4          grid_size;
    float4          grid_min;
    float4          grid_max;

    // number of cells in each dimension/side of grid
    float4          grid_res;

    float4          grid_delta;
    float4          grid_inv_delta;
	int				numParticles;
};

// GORDON Datastructure for Fluid parameters. To be reconciled with Ian's
// struct for fluid parameters
struct FluidParams
{
	float smoothing_length; // SPH radius
	float scale_to_simulation;
	float mass;
	float dt;
	float friction_coef;
	float restitution_coef;
	float damping;
	float shear;
	float attraction;
	float spring;
	float gravity; // -9.8 m/sec^2
};
//-------------------------

//pass parameters to OpenCL routines
typedef struct GE_SPHParams
{
    float4 grid_min; // changed by GE
    float4 grid_max;
    float grid_min_padding;     //float3s take up a float4 of space in OpenCL 1.0 and 1.1
    float grid_max_padding;
    float mass;
    float rest_distance;
    float smoothing_distance;
    float simulation_scale;
    float boundary_stiffness;
    float boundary_dampening;
    float boundary_distance;
    float EPSILON;
    float PI;       //delicious
    float K;        //speed of sound
} GE_SPHParams __attribute__((aligned(16)));

class GE_SPH : public System
{
public:
    GE_SPH(RTPS *ps, int num);
    ~GE_SPH();

    void update();
	//void setGEDataStructures(DataStructures* ds);

// BEGIN
// ADDED BY GORDON FOR TESTING of hash, sort, datastructures
// TO BE ADDED to System class
// MORE ARRAYS THAN NEEDED ...

	// Timers
	enum {TI_HASH=0, TI_SORT, TI_BUILD, TI_NEIGH, TI_DENS, TI_PRES, TI_EULER, TI_VISC, TI_UPDATE};
	GE::Time* ts_cl[20];   // ts_cl  is GE::Time**

	int nb_el;
	int nb_vars;
	int grid_size;

	//BufferGE<int>		cl_unsort_int;
	//BufferGE<int>		cl_sort_int;

	BufferGE<float4>* 	cl_vars_sorted;
	BufferGE<float4>* 	cl_vars_unsorted;
	BufferGE<float4>* 	cl_cells; // positions in Ian code
	BufferGE<int>* 		cl_cell_indices_start;
	BufferGE<int>* 		cl_cell_indices_end;
	BufferGE<int>* 		cl_vars_sort_indices;
	BufferGE<int>* 		cl_sort_hashes;
	BufferGE<int>* 		cl_sort_indices;
	BufferGE<int>* 		cl_unsort;
	BufferGE<int>* 		cl_sort;
	BufferGE<GridParams>*  cl_GridParams;
	BufferGE<FluidParams>* cl_FluidParams;

	BufferGE<float4>*	clf_debug;  //just for debugging cl files
	BufferGE<int4>*		cli_debug;  //just for debugging cl files

private:
	//DataStructures* ds;

public:
// Added by GE
	void hash();
	void sort(); //BufferGE<int>& key, BufferGE<int>& value);
	void setupArrays();
	void buildDataStructures();
	void neighbor_search();

private:
	void printSortDiagnostics();
	void prepareSortData();
	void printBuildDiagnostics();
	void printHashDiagnostics();

private:
    //the particle system framework
    RTPS *ps;

    GE_SPHSettings sph_settings;
    GE_SPHParams params;

    Kernel k_density, k_pressure, k_viscosity;
    Kernel k_collision_wall;
    Kernel k_euler;

	Kernel datastructures_kernel;
	Kernel hash_kernel;
	Kernel sort_kernel;
	Kernel step1_kernel;

    //Buffer<GE_SPHParams> cl_params;
    BufferGE<GE_SPHParams>* cl_params;


    std::vector<float4> positions;
    std::vector<float> densities;
    std::vector<float4> forces;
    std::vector<float4> velocities;

    Buffer<float4> cl_position;
    Buffer<float4> cl_color;
    Buffer<float> cl_density;
    Buffer<float4> cl_force;
    Buffer<float4> cl_velocity;
    
    Buffer<float4> cl_error_check;

    //these are defined in ge_sph/ folder next to the kernels
    void loadDensity();
    void loadPressure();
    void loadViscosity();
    void loadCollision_wall();
    void loadEuler();

	// loads kernel the first time, executes kernel every time
	void computeEuler(); //GE
	void computeDensity(); //GE
	void computePressure(); //GE
	void computeViscosity(); //GE

	// diagnostics, checking results of CPU and GPU code
	void checkDensity();

    void cpuDensity();

	void computeOnGPU();
	void computeOnCPU();
};

}

#endif

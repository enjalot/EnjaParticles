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
#include "../opencl/BufferVBO.h"
//#include "../util.h"
#include "../particle/UniformGrid.h"

#include "RadixSort.h"

// Make sure it is same as in density.cl
#define DENS 0
#define POS 1
#define VEL 2
#define FOR 3
#define SURF_TENS 4
#define COL 5
// gradient of color (COL)
#define NORM 6 

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
    float particle_radius;
    float boundary_distance;
    float spacing;
    float grid_cell_size;

	void print() {
		printf("----- GE_SPHSettings ----\n");
		printf("rest_density: %f\n", rest_density);
		printf("simulation_scale: %f\n", simulation_scale);
		printf("particle_mass: %f\n", particle_mass);
		printf("smoothing_distance: %f\n", smoothing_distance);
		printf("particle_radius: %f\n", particle_radius);
		printf("boundary_distance: %f\n", boundary_distance);
		printf("spacing: %f\n", spacing);
		printf("grid_cell_size: %f\n", grid_cell_size);
	}

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
	int				nb_vars; // for combined array

	void print() {
		printf("----- GridParms ----\n");
		grid_size.print("grid_size"); 
		grid_min.print("grid_min"); 
		grid_max.print("grid_max"); 
		grid_res.print("grid_res"); 
		grid_delta.print("grid_delta"); 
		grid_inv_delta.print("grid_inv_delta"); 
		printf("numParticles= %d\n", numParticles);
	}
};

// GORDON Datastructure for Fluid parameters. To be reconciled with Ian's
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

	void print() {
		printf("----- FluidParams ----\n");
		printf("scale_to_simulation: %f\n", scale_to_simulation);
		//printf("mass: %f\n", mass);
		//printf("dt: %f\n", dt);
		printf("friction_coef: %f\n", friction_coef);
		printf("restitution_coef: %f\n", restitution_coef);
		printf("damping: %f\n", damping);
		printf("shear: %f\n", shear);
		printf("attraction: %f\n", attraction);
		printf("spring: %f\n", spring);
		printf("gravity: %f\n", gravity);
		printf("choice: %d\n", choice);
	}
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
    float rest_density;
    float smoothing_distance;
    float particle_radius;
    float simulation_scale;
    float boundary_stiffness;
    float boundary_dampening;
    float boundary_distance;
    float EPSILON;
    float PI;       //delicious
    float K;        //speed of sound (what units?)
	float dt;

	void print()
	{
		printf("----- GE_SPHParams ----\n");
		grid_min.print("grid_min");
		grid_max.print("grid_max");
		printf("grid_min_padding= %f\n", grid_min_padding);
		printf("grid_max_padding= %f\n", grid_max_padding);
		printf("mass= %f\n", mass);
		printf("rest_distance= %f\n", rest_distance);
		printf("rest_density= %f\n", rest_density);
		printf("smoothing_distance= %f\n", smoothing_distance);
		printf("particle_radius= %f\n", particle_radius);
		printf("simulation_scale= %f\n", simulation_scale);
		printf("boundary_stiffness= %f\n", boundary_stiffness);
		printf("boundary_dampening= %f\n", boundary_dampening);
		printf("boundary_distance= %f\n", boundary_distance);
		printf("EPSILON= %f\n", EPSILON);
		printf("PI= %f\n", PI);
		printf("K= %f\n", K);
		printf("dt= %f\n", dt);
	}
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
	enum {TI_HASH=0, TI_RADIX_SORT, TI_BITONIC_SORT, TI_BUILD, TI_NEIGH, 
		  TI_DENS, TI_PRES, TI_EULER, TI_VISC, TI_UPDATE, TI_COLLISION_WALL, 
		  TI_COL, TI_COL_NORM};
	GE::Time* ts_cl[20];   // ts_cl  is GE::Time**

	int nb_el;
	int nb_vars;
	int grid_size;
	int4 nb_cells;

	//BufferGE<int>		cl_unsort_int;
	//BufferGE<int>		cl_sort_int;

	// Two arrays for bitonic sort (sort not done in place)
	BufferGE<int>* cl_sort_output_hashes;
	BufferGE<int>* cl_sort_output_indices;
	//BufferGE<int> cl_sort_output_hashes(ps->cli, nb_el);
	//BufferGE<int> cl_sort_output_indices(ps->cli, nb_el);

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
	RadixSort*   radixSort;

private:
	//DataStructures* ds;

public:
// Added by GE
	void hash();
	void radix_sort(); //BufferGE<int>& key, BufferGE<int>& value);
	void bitonic_sort(); //not in place, but keys/values
	void setupArrays();
	void buildDataStructures();
	void neighborSearch(int choice);

private:
	void printSortDiagnostics();
	//void printBiSortDiagnostics();
	void printBiSortDiagnostics(BufferGE<int>& cl_sort_output_hashes, BufferGE<int>& cl_sort_output_indices);
	void prepareSortData();
	void printBuildDiagnostics();
	void printHashDiagnostics();
	GE_SPHParams& getParams() {return params;}

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

    BufferGE<GE_SPHParams>* cl_params;


    std::vector<float4> positions;
    std::vector<float> densities;
    std::vector<float4> forces;
    std::vector<float4> velocities;

    BufferVBO<float4>* cl_position;
    BufferVBO<float4>* cl_color;
    BufferGE<float>*   cl_density;
    BufferGE<float4>*  cl_force;
    BufferGE<float4>*  cl_velocity;
    
    BufferGE<float4>* cl_error_check;

    //these are defined in ge_sph/ folder next to the kernels
    void loadDensity();
    void loadPressure();
    void loadViscosity();
    void loadCollision_wall();
    void loadEuler();

	// loads kernel the first time, executes kernel every time
	void computeCollisionWall(); //GE
	void computeEuler(); //GE
	void computeDensity(); //GE
	void computePressure(); //GE
	void computeViscosity(); //GE

	// diagnostics, checking results of CPU and GPU code
	void checkDensity();

    void cpuDensity();

	void computeOnGPU(int nb_sub_iter);
	void computeOnCPU();

	void computeCellStartEndGPU();
	void computeCellStartEndCPU();

	void printGPUDiagnostics();
};

}

#endif

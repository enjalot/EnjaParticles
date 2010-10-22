#ifndef _DATASTRUCTURES_H_
#define _DATASTRUCTURES_H_

// for cl_float4, etc.
#include <CL/cl_platform.h>

#include <timege.h>
#include <GL/glew.h>
#include <string>
#include "../opencl/BufferGE.h"
#include "../util.h"
#include "../structs.h"
#include "../opencl/Kernel.h"
#include "../oclRadixSortKeysValues/inc/RadixSort.h"

#include "../RTPS.h"
//class RTPS;

namespace rtps {

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

//-------------------------------------------
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

//----------------------------------------------------------------------
class DataStructures
{
private:
// BEGIN
// ADDED BY GORDON FOR TESTING of hash, sort, datastructures
	int nb_el, nb_vars, grid_size;
	BufferGE<float4> cl_vars_sorted;
	BufferGE<float4> cl_vars_unsorted;
	BufferGE<int> cl_cell_indices_start;
	BufferGE<int> cl_cell_indices_end;
	BufferGE<int> cl_vars_sort_indices;
	//BufferGE<int> cl_unsort_int;
	//BufferGE<int> cl_sort_int;
	BufferGE<int> cl_sort_hashes;
	BufferGE<int> cl_sort_indices;
	BufferGE<GridParams> cl_GridParams;
	BufferGE<FluidParams> cl_FluidParams;
	BufferGE<float4> cl_cells;
	BufferGE<int> cl_unsort;
	BufferGE<int> cl_sort;

    Kernel sort_kernel;     //
    Kernel hash_kernel;     //
    Kernel datastructures_kernel;     //
    Kernel step1_kernel;     //

	int err;

	// GET THESE FROM ANOTHER CLASS
    //opencl
    //std::vector<cl::Device> devices;
    //cl::Context context;
    //cl::CommandQueue queue;
    //cl::Event event;

	RTPS* ps;

public:
	enum {TI_HASH=0, TI_SORT, TI_BUILD, TI_NEIGH, TI_DENS, TI_PRES, TI_EULER};
	GE::Time* ts_cl[10];

	int getNbVars();

public:
	DataStructures(RTPS* ps);
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
};

}
#endif

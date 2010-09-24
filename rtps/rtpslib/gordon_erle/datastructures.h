#ifndef _DATASTRUCTURES_H_
#define _DATASTRUCTURES_H_

#include "../opencl/Buffer.h"

namespace rtps {

struct GridParams
{
    float4          grid_size;
    float4          grid_min;
    float4          grid_max;

    // number of cells in each dimension/side of grid
    float4          grid_res;

    float4          grid_delta;
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
	//CL cl;  // GE CL library
	int nb_el, nb_vars, grid_size;
	Buffer cl_vars_sorted<float4>;
	Buffer cl_vars_unsorted<float4>;
	Buffer cl_cell_indices_start<int>;
	Buffer cl_cell_indices_end<int>;
	Buffer cl_sort_hashes<int>;
	Buffer cl_sort_indices<int>;
	Buffer cl_GridParams<GridParams>;
	Buffer cl_FluidParams<FluidParams>;
	Buffer cl_cells<int>;
	Buffer cl_unsort<int>;
	Buffer cl_sort<int>;
	std::vector<cl_uint> sort_indices;
	std::vector<cl_uint> sort_hashes;
	std::vector<cl_float4> vars_sorted; 
	std::vector<cl_float4> vars_unsorted; 
	std::vector<cl_uint> cell_indices_start;
	std::vector<cl_uint> cell_indices_end;
	std::vector<cl_float4> cells;
	std::vector<int> sort_int;
	std::vector<int> unsort_int;
	//GridParams gp;
	//FluidParams fp;

public:
	hash();
	sort();
	setupArrays();
	buildDataStructures();
};

}
#endif

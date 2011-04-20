#ifndef _CL_FLOCK_STRUCTS_H_
#define _CL_FLOCK_STRUCTS_H_

#include "../cl_common/cl_structs.h"

//Struct which gets passed to OpenCL routines
typedef struct FLOCKParameters
{
    // use it later
    float mass;
   
    // simulation settings 
    float simulation_scale;
    float rest_distance;
    float smoothing_distance;
//    float spacing;
    
    // grid dimensions for boundary conditions
    float4 grid_min;
    float4 grid_max;
    
    // CL parameters 
    int num;
    int nb_vars;    // for combined variables (vars_sorted, etc.)
	int choice;     // which kind of calculation to invoke
    int max_num;

    // Boids parameters
    float min_dist;  // desired separation between boids
    float search_radius;
    float max_speed; 
    
    // Boid rules' weights
    float w_sep;
    float w_align;
    float w_coh;
} FLOCKParameters;

// Will be local variable
// used to output multiple variables per point
typedef struct Boid 
{
	float4 separation;
	float4 alignment;  
	float4 cohesion;
	float4 acceleration;
	float4 color;
    int num_flockmates;
    int num_nearestFlockmates;
} Boid;

/*
//----------------------------------------------------------------------
struct GridParams
{
    float4          grid_size;
    float4          grid_min;
    float4          grid_max;
    float4          bnd_min;
    float4          bnd_max;

    // number of cells in each dimension/side of grid
    float4          grid_res;
    float4          grid_delta;

    int nb_cells;
};

struct FLOCKParameters
{

    float4 grid_min;
    float4 grid_max;
    
    float rest_distance;
    float smoothing_distance;
    
    int num;
    int nb_vars; // for combined variables (vars_sorted, etc.)
	int choice; // which kind of calculation to invoke
    
    // Boids
    float min_dist;  // desired separation between boids
    float search_radius;
    float max_speed; 
    
    float w_sep;
    float w_align;
    float w_coh;
};
*/

#endif

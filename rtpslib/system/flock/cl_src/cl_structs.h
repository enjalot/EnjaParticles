#ifndef _CL_FLOCK_STRUCTS_H_
#define _CL_FLOCK_STRUCTS_H_

#include "../cl_common/cl_structs.h"

//Struct which gets passed to OpenCL routines
typedef struct FLOCKParameters
{
    // simulation settings 
    float simulation_scale;
    float rest_distance;
    float smoothing_distance;
    
    // Boids parameters
    float min_dist;  // desired separation between boids
    float search_radius;
    float max_speed; 
    float ang_vel;

    // Boid rules' weights
    float w_sep;
    float w_align;
    float w_coh;
    float w_goal;
    float w_avoid;
    float w_wander;
    float w_leadfoll;

    // Boid rule's settings
    float slowing_distance;
    int leader_index;

    int num;
    int max_num;
} FLOCKParameters;

// Will be local variable used to output multiple variables per point
typedef struct Boid 
{
	float4 separation;
	float4 alignment;  
	float4 cohesion;
	float4 goal;
	float4 avoid;
	float4 leaderfollowing;
	float4 color;
    int num_flockmates;
    int num_nearestFlockmates;
} Boid;

#endif

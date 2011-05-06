#ifndef RTPS_FLOCKSETTINGS_H_INCLUDED
#define RTPS_FLOCKSETTINGS_H_INCLUDED

#include <stdlib.h>
#include <string>
#include <map>
#include <iostream>
#include <stdio.h>
#include <sstream>

#include <structs.h>
#include <Buffer.h>
#include <Domain.h>

namespace rtps 
{

#ifdef WIN32
#pragma pack(push,16)
#endif
//Struct which gets passed to OpenCL routines
typedef struct FLOCKParameters
{
    // use it later
    float mass;
   
    // simulation settings 
    float simulation_scale;
    float rest_distance;
    float smoothing_distance;
    
    // CL parameters 
    //int nb_vars;    // for combined variables (vars_sorted, etc.)
	//int choice;     // which kind of calculation to invoke
    //int max_num;

    // Boids parameters
    float min_dist;  // desired separation between boids
    float search_radius;
    float max_speed; 
    
    // Boid rules' weights
    float w_sep;
    float w_align;
    float w_coh;

    int num;
} FLOCKParameters
#ifndef WIN32
    __attribute__((aligned(16)));
#else
    ;
#pragma pack(pop,16)
#endif
    
}

#endif

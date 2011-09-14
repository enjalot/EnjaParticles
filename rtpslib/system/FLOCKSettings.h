/****************************************************************************************
* Real-Time Particle System - An OpenCL based Particle system developed to run on modern GPUs. Includes SPH fluid simulations.
* version 1.0, September 14th 2011
* 
* Copyright (C) 2011 Ian Johnson, Andrew Young, Gordon Erlebacher, Myrna Merced, Evan Bollig
* 
* This software is provided 'as-is', without any express or implied
* warranty.  In no event will the authors be held liable for any damages
* arising from the use of this software.
* 
* Permission is granted to anyone to use this software for any purpose,
* including commercial applications, and to alter it and redistribute it
* freely, subject to the following restrictions:
* 
* 1. The origin of this software must not be misrepresented; you must not
* claim that you wrote the original software. If you use this software
* in a product, an acknowledgment in the product documentation would be
* appreciated but is not required.
* 2. Altered source versions must be plainly marked as such, and must not be
* misrepresented as being the original software.
* 3. This notice may not be removed or altered from any source distribution.
****************************************************************************************/


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
} FLOCKParameters
#ifndef WIN32
    __attribute__((aligned(16)));
#else
    ;
#pragma pack(pop,16)
#endif
    
}

#endif

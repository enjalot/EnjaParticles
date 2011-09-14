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


#ifndef RTPS_FORCEFIELD_H_INCLUDED
#define RTPS_FORCEFIELD_H_INCLUDED

//#include "../rtps_common.h"
#include "structs.h"
namespace rtps
{

enum FFType{ATTRACTOR, REPELER};

//keep track of the fluid settings
#ifdef WIN32
#pragma pack(push,16)
#endif
typedef struct ForceField
{
    float4 center;
    float radius;
    float max_force;
    float f;        //memory padding for opencl
    float ff;
    //FFType type;
    //unsigned int type;
    //unsigned int padd;

    ForceField(){};
    //ForceField(float4 center, float radius, float max_force, unsigned int type, unsigned int padd)
    ForceField(float4 center, float radius, float max_force)
    {
        this->center = center;
        this->radius = radius;
        this->max_force = max_force;
        this->f = 0;
        this->ff = 0;
        //this->type = type;
        //this->padd = padd;
    }

} ForceField 
#ifndef WIN32
__attribute__((aligned(16)));
#else
;
#pragma pack(pop)
#endif



}


#endif

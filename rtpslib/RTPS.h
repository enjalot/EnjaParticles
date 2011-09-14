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


#ifndef RTPS_RTPS_H_INCLUDED
#define RTPS_RTPS_H_INCLUDED

#include <vector>

//System API
#include "system/System.h"

//OpenCL API
#include "opencl/CLL.h"

//initial value API
//TODO probably shouldn't be included here
#include "domain/IV.h"

//settings class to configure the framework
#include "RTPSettings.h"

//defines useful structs like float3 and float4
#include "structs.h"

//defines a few handy utility functions
//TODO should not be included here
#include "util.h"

#ifdef WIN32
    #if defined(rtps_EXPORTS)
        #define RTPS_EXPORT __declspec(dllexport)
    #else
        #define RTPS_EXPORT __declspec(dllimport)
	#endif 
#else
    #define RTPS_EXPORT
#endif

namespace rtps
{

    class RTPS_EXPORT RTPS
    
    {
    public:
        //default constructor
        RTPS();
        //Setup CL, Render, initial values and System based on settings
        RTPS(RTPSettings *s);
        RTPS(RTPSettings *s, CL* _cli);

        ~RTPS();

        void Init();

        //Keep track of settings
        RTPSettings *settings;
        
        //OpenCL abstraction instance
        //TODO shouldn't be public
        CL *cli;

        //will be instanciated as a specific subclass like SPH or Boids
        //TODO shouldn't be public? right now we expose various methods from the system
        System *system;
        System *system_outer; // GE: meant for ghost and other non-fluid particles
        //std::vector<System> systems;

        //initial value helper
        //IV iv;

        void update();
        void render();

        void printTimers();

    private:
        bool cl_managed;
        
    };
}

#endif

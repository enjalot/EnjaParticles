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


#ifndef RTPS_SIMPLE_H_INCLUDED
#define RTPS_SIMPLE_H_INCLUDED

#include <string>

#include "RTPS.h"
#include "System.h"
#include "ForceField.h"
#include "Kernel.h"
#include "Buffer.h"

#ifdef WIN32
    #if defined(rtps_EXPORTS)
        #define RTPS_EXPORT __declspec(dllexport)
    #else
        #define RTPS_EXPORT __declspec(dllimport)
	#endif 
#else
    #define RTPS_EXPORT
#endif
//#include "../util.h"

namespace rtps
{


    class RTPS_EXPORT Simple : public System
    {
    public:
        Simple(RTPS *ps, int num);
        ~Simple();

        void update();

        bool forcefields_enabled;
        int max_forcefields;

        //the particle system framework
        RTPS *ps;

        std::vector<float4> positions;
        std::vector<float4> colors;
        std::vector<float4> velocities;
        std::vector<float4> forces;
        std::vector<ForceField> forcefields;


        Kernel k_forcefield;
        Kernel k_euler;

        Buffer<float4> cl_position;
        Buffer<float4> cl_color;
        Buffer<float4> cl_force;
        Buffer<float4> cl_velocity;
        Buffer<ForceField> cl_forcefield;


        void loadForceField();
        void loadForceFields(std::vector<ForceField> ff);
        void loadEuler();

        void cpuForceField();
        void cpuEuler();


    };

}

#endif

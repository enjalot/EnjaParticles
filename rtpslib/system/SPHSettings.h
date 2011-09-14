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


#ifndef RTPS_SPHSETTINGS_H_INCLUDED
#define RTPS_SPHSETTINGS_H_INCLUDED

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
	typedef struct SPHParams
    {
        float mass;
        float rest_distance;
        float smoothing_distance;
        float simulation_scale;
        
        //dynamic params
        float boundary_stiffness;
        float boundary_dampening;
        float boundary_distance;
        float K;        //gas constant
        
        float viscosity;
        float velocity_limit;
        float xsph_factor;
        float gravity; // -9.8 m/sec^2

        float friction_coef;
        //next 4 not used at the moment
        float restitution_coef;
        float shear;
        float attraction;

        float spring;
        //float surface_threshold;
        //constants
        float EPSILON;
        float PI;       //delicious
        //Kernel Coefficients
        float wpoly6_coef;
        
        float wpoly6_d_coef;
        float wpoly6_dd_coef; // laplacian
        float wspiky_coef;
        float wspiky_d_coef;

        float wspiky_dd_coef;
        float wvisc_coef;
        float wvisc_d_coef;
        float wvisc_dd_coef;


        //CL parameters
        int num;
        int nb_vars; // for combined variables (vars_sorted, etc.)
        int choice; // which kind of calculation to invoke
        int max_num;

		//CL parameter, cloud
        int cloud_num; // nb cloud points
        int max_cloud_num;


        void print()
        {
            printf("----- SPHParams ----\n");
            printf("mass: %f\n", mass);
            printf("rest distance: %f\n", rest_distance);
            printf("smoothing distance: %f\n", smoothing_distance);
            printf("simulation_scale: %f\n", simulation_scale);
            printf("--------------------\n");

            /*
            printf("friction_coef: %f\n", friction_coef);
            printf("restitution_coef: %f\n", restitution_coef);
            printf("damping: %f\n", boundary_dampening);
            printf("shear: %f\n", shear);
            printf("attraction: %f\n", attraction);
            printf("spring: %f\n", spring);
            printf("gravity: %f\n", gravity);
            printf("choice: %d\n", choice);
            */
        }
    } SPHParams
#ifndef WIN32
	__attribute__((aligned(16)));
#else
		;
        #pragma pack(pop)
#endif


    enum Integrator
    {
        EULER, LEAPFROG
    };


}

#endif

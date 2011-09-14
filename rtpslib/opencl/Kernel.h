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


#ifndef RTPS_KERNEL_H_INCLUDED
#define RTPS_KERNEL_H_INCLUDED
/*
 * The Kernel class abstracts the OpenCL Kernel class
 * by providing some convenience methods
 *
 * For now we build one program per kernel. In the future
 * we will make it possible to make several kernels per program
 *
 * we pass in an OpenCL instance  to the constructor 
 * which manages the underlying context and queues
 */

#include <string>
#include <stdio.h>

#include "CLL.h"
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


    class RTPS_EXPORT Kernel
    {
    public:
        Kernel()
        {
            cli = NULL;
        };
        Kernel(CL *cli, std::string source, std::string name);
        Kernel(CL *cli, cl::Program program, std::string name);

        //we will want to access buffers by name when going accross systems
        std::string name;
        std::string source;

        CL *cli;
        //we need to build a program to have a kernel
        cl::Program program;

        //the actual OpenCL kernel object
        cl::Kernel kernel;

        template <class T> void setArg(int arg, T val);
        void setArgShared(int arg, int nb_bytes);

        //execute the kernel and return the time it took in milliseconds using GPU timer
        //assumes null range for worksize offset and local worksize
        float execute(int ndrange);
        //later we will make more execute routines to give more options
        float execute(int ndrange, int workgroup_size);

    };

    template <class T> void Kernel::setArg(int arg, T val)
    {
        try
        {
            kernel.setArg(arg, val);
        }
        catch (cl::Error er)
        {
            printf("ERROR: %s(%s)\n", er.what(), oclErrorString(er.err()));
        }

    }



}

#endif


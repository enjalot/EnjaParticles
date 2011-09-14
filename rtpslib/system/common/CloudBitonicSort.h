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


#ifndef RTPS_CLOUD_BITONIC_SORT_H
#define RTPS_CLOUD_BITONIC_SORT_H

#include "CLL.h"
#include "../opencl/Kernel.h"
#include "../opencl/Buffer.h"
//#include "../opencl/Scopy.h"

#ifndef uint
#define uint unsigned int
#endif

static const uint CLOUD_LOCAL_SIZE_LIMIT = 512U;

namespace rtps
{

template <class T>
class CloudBitonic
{
public:
    CloudBitonic(){ cli=NULL; };
    //create an OpenCL buffer from existing data
    CloudBitonic( std::string source_dir, CL *cli );

    int Sort(int batch, int arrayLength, int dir,
                Buffer<T> *dstkey, Buffer<T> *dstval, 
                Buffer<T> *srckey, Buffer<T> *srcval);
    void loadKernels(std::string source_dir);



private:
    Kernel k_bitonicSortLocal, k_bitonicSortLocal1;
    Kernel k_bitonicMergeLocal, k_bitonicMergeGlobal;
    /*
    Buffer<T> *cl_srckey;
    Buffer<T> *cl_srcval;
    Buffer<T> *cl_dstkey;
    Buffer<T> *cl_dstval;
    */

    CL *cli;

};

#include "CloudBitonicSort.cpp"

}

#endif

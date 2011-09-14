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


#ifndef IV_H_INCLUDE
#define IV_H_INCLUDE

//Initial Value functions
#include <vector>
#include "../structs.h"
#include <vector>
#include "../rtps_common.h"
using namespace std;

namespace rtps
{

    RTPS_EXPORT vector<float4> addRect(int num, float4 min, float4 max, float spacing, float scale);
    RTPS_EXPORT vector<float4> addSphere(int num, float4 center, float radius, float spacing, float scale);
    RTPS_EXPORT std::vector<float4> addDisc(int num, float4 center, float4 u, float4 v, float radius, float spacing);
    RTPS_EXPORT std::vector<float4> addDiscRandom(int num, float4 center, float4 vel, float4 u, float4 v, float radius, float spacing);

    RTPS_EXPORT void addCube(int num, float4 min, float4 max, float spacing, float scale, std::vector<float4>& rvec);

    RTPS_EXPORT vector<float4> addRandRect(int num, float4 min, float4 max, float spacing, float scale, float4 dmin, float4 dmax);
    RTPS_EXPORT vector<float4> addRandArrangement(int num, float scale, float4 dmin, float4 dmax);
    RTPS_EXPORT vector<float4> addRandSphere(int num, float4 center, float radius, float spacing, float scale, float4 dmin, float4 dmax);
	RTPS_EXPORT std::vector<float4> addHollowSphere(int nn, float4 center, float radius_in, float radius_out, float spacing, float scale, std::vector<float4>& normals);
    RTPS_EXPORT std::vector<float4> addxyPlane(int num, float4 min, float4 max, float spacing, float scale, float zlevel, std::vector<float4>& normals);
}

#endif

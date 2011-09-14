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


// FastSqrt.cpp
//


#include "FastSqrt.h"
#include <math.h>


typedef union FastSqrtUnion FastSqrtUnion, *PFastSqrtUnion, &RFastSqrtUnion;

union FastSqrtUnion
{
	Float  float_val;
	UInt32 uint_val;
};


UInt32 FastSqrtTable[0x10000];


Void BuildSqrtTable()
{
	Int i;
	FastSqrtUnion squareroot;

	for (i = 0; i <= 0x7FFF; i++)
	{
		squareroot.uint_val  = (i << 8) | (0x7F << 23);
		squareroot.float_val = (Float) sqrt(squareroot.float_val);

		FastSqrtTable[i + 0x8000] = (squareroot.uint_val & 0x7FFFFF);

		squareroot.uint_val  = (i << 8) | (0x80 << 23);
		squareroot.float_val = (Float) sqrt(squareroot.float_val);

		FastSqrtTable[i] = (squareroot.uint_val & 0x7FFFFF);
	}
}
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


// FastTrigonometry.h
//


#ifndef _3D_FASTTRIGONOMETRY_H
#define _3D_FASTTRIGONOMETRY_H


#include "Types.h"
#include "Math\Vector\Vector 3.h"


ForceInline Float __fastcall SinCos(Float x, PFloat cosval)
{
	x; cosval;						// Just for this shit "unreferenced formal parameter" warning

	__asm
	{
		fld      dword ptr [x]
		fsincos
		fstp     dword ptr [cosval]
	}
}

ForceInline Void __fastcall SinCosVec(PVector3 vec, PVector3 sinvalues, PVector3 cosvalues)
{
	vec; sinvalues; cosvalues;		// Just for this shit "unreferenced formal parameter" warning

	__asm
	{
		mov      eax, vec
		mov      ecx, sinvalues
		mov      edx, cosvalues

		fld      dword ptr [eax]
		fsincos
		fstp     dword ptr [ecx]
		fstp     dword ptr [edx]

		fld      dword ptr [eax+4]
		fsincos
		fstp     dword ptr [ecx+4]
		fstp     dword ptr [edx+4]

		fld      dword ptr [eax+8]
		fsincos
		fstp     dword ptr [ecx+8]
		fstp     dword ptr [edx+8]
	}
}

ForceInline Float __fastcall CoTan(Float x)
{
	x;								// Just for this shit "unreferenced formal parameter" warning

    __asm
	{
		fld      dword ptr [x]
		fptan
		fdivrp   st(1), st
	}
}


#endif
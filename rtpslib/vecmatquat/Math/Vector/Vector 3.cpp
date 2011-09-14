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


// Vector 3.cpp
//


#include "Vector 3.h"
#include "Math\FastSqrt.h"
#include <math.h>


#define VECTOR_COMPARE_EPSILON       0.0005


Bool VecIsValid(RVector3 vec)
{
    if ((vec.m_X * vec.m_X) < 0.0f)
		return FALSE;
	if ((vec.m_Y * vec.m_Y) < 0.0f)
		return FALSE;
	if ((vec.m_Z * vec.m_Z) < 0.0f)
		return FALSE;

	return TRUE;
}

Bool VecIsNormalized(RVector3 vec)
{
	if (fabs(Vec3LengthSquared(vec) - 1.0f) < (VECTOR_COMPARE_EPSILON * VECTOR_COMPARE_EPSILON))
		return TRUE;
	return FALSE;
}

Float VecNormalize(RVector3 vec)
{
    Float fLength = FastSqrt(Vec3LengthSquared(vec));
	
	if (fLength == 0.0f)
		return 0.0f;

	Float fOneOverLength = 1.0f / fLength;
	
	vec.m_X *= fOneOverLength;
	vec.m_Y *= fOneOverLength;
	vec.m_Z *= fOneOverLength;

	return fLength;
}

Float VecLength(RVector3 vec)
{
    return (FastSqrt(Vec3LengthSquared(vec)));
}

Float VecDistanceBetween(RVector3 v1, RVector3 v2)
{
	Vector3 tmp;
	Vec3Sub(v1, v2, tmp);

    return (FastSqrt(Vec3LengthSquared(tmp)));
}

Bool VecCompare(RVector3 v1, RVector3 v2, Float fTolerance)
{
	if (fabs(v1.m_X - v2.m_X) > fTolerance)
		return FALSE;
	if (fabs(v1.m_Y - v2.m_Y) > fTolerance)
		return FALSE;
	if (fabs(v1.m_Z - v2.m_Z) > fTolerance)
		return FALSE;
	return TRUE;
}
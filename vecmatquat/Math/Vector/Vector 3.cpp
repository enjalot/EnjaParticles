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
// Vector 4.cpp
//


#include "Vector 4.h"
#include "Math\FastSqrt.h"
#include <math.h>


#define VECTOR_COMPARE_EPSILON       0.0005


Bool VecIsValid(RVector4 vec)
{
    if ((vec.m_X * vec.m_X) < 0.0f)
		return FALSE;
	if ((vec.m_Y * vec.m_Y) < 0.0f)
		return FALSE;
	if ((vec.m_Z * vec.m_Z) < 0.0f)
		return FALSE;
	if ((vec.m_W * vec.m_W) < 0.0f)
		return FALSE;

	return TRUE;
}

Bool VecIsNormalized(RVector4 vec)
{
	if (fabs(Vec4LengthSquared(vec) - 1.0f) < (VECTOR_COMPARE_EPSILON * VECTOR_COMPARE_EPSILON))
		return TRUE;
	return FALSE;
}

Float VecNormalize(RVector4 vec)
{
    Float fLength = FastSqrt(Vec4LengthSquared(vec));
	
	if (fLength == 0.0f)
		return 0.0f;

	Float fOneOverLength = 1.0f / fLength;
	
	vec.m_X *= fOneOverLength;
	vec.m_Y *= fOneOverLength;
	vec.m_Z *= fOneOverLength;
	vec.m_W *= fOneOverLength;

	return fLength;
}

Float VecLength(RVector4 vec)
{
    return (FastSqrt(Vec4LengthSquared(vec)));
}

Float VecDistanceBetween(RVector4 v1, RVector4 v2)
{
	Vector4 tmp;
	Vec4Sub(v1, v2, tmp);

    return (FastSqrt(Vec4LengthSquared(tmp)));
}

Bool VecCompare(RVector4 v1, RVector4 v2, Float fTolerance)
{
	if (fabs(v1.m_X - v2.m_X) > fTolerance)
		return FALSE;
	if (fabs(v1.m_Y - v2.m_Y) > fTolerance)
		return FALSE;
	if (fabs(v1.m_Z - v2.m_Z) > fTolerance)
		return FALSE;
	if (fabs(v1.m_W - v2.m_W) > fTolerance)
		return FALSE;
	return TRUE;
}
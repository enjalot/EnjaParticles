// Vector 3.cpp
//


#include "Vector 2.h"
#include "Math\FastSqrt.h"
#include <math.h>


#define VECTOR_COMPARE_EPSILON       0.0005


Bool VecIsValid(RVector2 vec)
{
    if ((vec.m_X * vec.m_X) < 0.0f)
		return FALSE;
	if ((vec.m_Y * vec.m_Y) < 0.0f)
		return FALSE;

	return TRUE;
}

Bool VecIsNormalized(RVector2 vec)
{
	if (fabs(Vec2LengthSquared(vec) - 1.0f) < (VECTOR_COMPARE_EPSILON * VECTOR_COMPARE_EPSILON))
		return TRUE;
	return FALSE;
}

Float VecNormalize(RVector2 vec)
{
    Float fLength = FastSqrt(Vec2LengthSquared(vec));
	
	if (fLength == 0.0f)
		return 0.0f;

	Float fOneOverLength = 1.0f / fLength;
	
	vec.m_X *= fOneOverLength;
	vec.m_Y *= fOneOverLength;

	return fLength;
}

Float VecLength(RVector2 vec)
{
    return (FastSqrt(Vec2LengthSquared(vec)));
}

Float VecDistanceBetween(RVector2 v1, RVector2 v2)
{
	Vector2 tmp;
	Vec2Sub(v1, v2, tmp);

    return (FastSqrt(Vec2LengthSquared(tmp)));
}

Bool VecCompare(RVector2 v1, RVector2 v2, Float fTolerance)
{
	if (fabs(v1.m_X - v2.m_X) > fTolerance)
		return FALSE;
	if (fabs(v1.m_Y - v2.m_Y) > fTolerance)
		return FALSE;
	return TRUE;
}
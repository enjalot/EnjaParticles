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


// Quaternion.cpp
//


#include "Quaternion.h"
#include "Math\FastSqrt.h"
#include "Math\FastTrigonometry.h"
#include <math.h>
#include <assert.h>


#define QUAT_UNIT_TOLERANCE				0.001  
#define QUAT_QZERO_TOLERANCE			0.00001 
#define QUAT_TRACE_QZERO_TOLERANCE		0.1
#define QUAT_AA_QZERO_TOLERANCE			0.0001
#define QUAT_ZERO_EPSILON				0.0001
#define QUAT_EPSILON					0.0001


Bool QuatIsValid(RQuaternion quat)
{
	if ((quat.m_W * quat.m_W) < 0.0f)
		return FALSE;
	if ((quat.m_X * quat.m_X) < 0.0f)
		return FALSE;
	if ((quat.m_Y * quat.m_Y) < 0.0f)
		return FALSE;
	if ((quat.m_Z * quat.m_Z) < 0.0f)
		return FALSE;

	return TRUE;
}

Bool QuatIsUnit(RQuaternion quat)
{
    if (fabs(QuatLengthSquared(quat) - 1.0f) < (QUAT_UNIT_TOLERANCE * QUAT_UNIT_TOLERANCE))
		return TRUE;
	return FALSE;
}

Float QuatNormalize(RQuaternion quat)
{
    Float length = FastSqrt(QuatLengthSquared(quat));

	if (fabs(length) < QUAT_QZERO_TOLERANCE)
		return 0.0f;

	Float oneoverlength = 1.0f / length;

	QuatScale(quat, oneoverlength);
	
	return length;
}

Bool QuatCompare(RQuaternion q1, RQuaternion q2, Float fTolerance)
{
	if (fabs(q1.m_X - q2.m_X) > fTolerance)
		return FALSE;
	if (fabs(q1.m_Y - q2.m_Y) > fTolerance)
		return FALSE;
	if (fabs(q1.m_Z - q2.m_Z) > fTolerance)
		return FALSE;
	if (fabs(q1.m_W - q2.m_W) > fTolerance)
		return FALSE;
	return TRUE;
}

Void QuatSetFromAxisAngle(RQuaternion quat, RVector3 axis, Float theta)
{
	theta = SinCos(theta * 0.5f, &quat.m_W);
	quat.m_X = axis.m_X * theta;
	quat.m_Y = axis.m_Y * theta;
	quat.m_Z = axis.m_Z * theta;
}

Bool QuatGetAxisAngle(RQuaternion quat, RVector3 axis, PFloat theta)
{	
	Float OneOverSinTheta;
	Float HalfTheta = acosf(quat.m_W);
	
	if (HalfTheta > QUAT_QZERO_TOLERANCE)
	{
		OneOverSinTheta = 1.0f / sinf(HalfTheta);
		axis.m_X = OneOverSinTheta * quat.m_X;
		axis.m_Y = OneOverSinTheta * quat.m_Y;
		axis.m_Z = OneOverSinTheta * quat.m_Z;
		*theta = 2.0f * HalfTheta;

		return TRUE;
	}
	else
	{
		Vec3Clear(axis);
		*theta = 0.0f;

		return FALSE;
	}
}

Void QuatFromMatrix(RMatrix4x4 matrix, RQuaternion quat)
{
	Float trace, s;

	trace = matrix.f11 + matrix.f22 + matrix.f33;
	if (trace > 0.0f)
	{
		s = FastSqrt(trace + 1.0f);
		quat.m_W = s * 0.5f;
		s = 0.5f / s;

		quat.m_X = (matrix.f32 - matrix.f23) * s;
		quat.m_Y = (matrix.f13 - matrix.f31) * s;
		quat.m_Z = (matrix.f21 - matrix.f12) * s;
	}
	else
	{
		Int biggest;
		enum {A,E,I};
		if (matrix.f11 > matrix.f22)
		{
			if (matrix.f33 > matrix.f11)
				biggest = I;	
			else
				biggest = A;
		}
		else
		{
			if (matrix.f33 > matrix.f11)
				biggest = I;
			else
				biggest = E;
		}

		switch (biggest)
		{
			case A:
				s = FastSqrt(matrix.f11 - (matrix.f22 + matrix.f33) + 1.0f);
				if (s > QUAT_TRACE_QZERO_TOLERANCE)
				{
					quat.m_X = s * 0.5f;
					s = 0.5f / s;
					quat.m_W = (matrix.f32 - matrix.f23) * s;
					quat.m_Y = (matrix.f12 + matrix.f21) * s;
					quat.m_Z = (matrix.f13 + matrix.f31) * s;
					break;
				}
				// I
				s = FastSqrt(matrix.f33 - (matrix.f11 + matrix.f22) + 1.0f);
				if (s > QUAT_TRACE_QZERO_TOLERANCE)
				{
					quat.m_Z = s * 0.5f;
					s = 0.5f / s;
					quat.m_W = (matrix.f21 - matrix.f12) * s;
					quat.m_X = (matrix.f31 + matrix.f13) * s;
					quat.m_Y = (matrix.f32 + matrix.f23) * s;
					break;
				}
				// E
				s = FastSqrt(matrix.f22 - (matrix.f33 + matrix.f11) + 1.0f);
				if (s > QUAT_TRACE_QZERO_TOLERANCE)
				{
					quat.m_Y = s * 0.5f;
					s = 0.5f / s;
					quat.m_W = (matrix.f13 - matrix.f31) * s;
					quat.m_Z = (matrix.f23 + matrix.f32) * s;
					quat.m_X = (matrix.f21 + matrix.f12) * s;
					break;
				}
				break;

			case E:
				s = FastSqrt(matrix.f22 - (matrix.f33 + matrix.f11) + 1.0f);
				if (s > QUAT_TRACE_QZERO_TOLERANCE)
				{
					quat.m_Y = s * 0.5f;
					s = 0.5f / s;
					quat.m_W = (matrix.f13 - matrix.f31) * s;
					quat.m_Z = (matrix.f23 + matrix.f32) * s;
					quat.m_X = (matrix.f21 + matrix.f12) * s;
					break;
				}
				// I
				s = FastSqrt(matrix.f33 - (matrix.f11 + matrix.f22) + 1.0f);
				if (s > QUAT_TRACE_QZERO_TOLERANCE)
				{
					quat.m_Z = s * 0.5f;
					s = 0.5f / s;
					quat.m_W = (matrix.f21 - matrix.f12) * s;
					quat.m_X = (matrix.f31 + matrix.f13) * s;
					quat.m_Y = (matrix.f32 + matrix.f23) * s;
					break;
				}
				// A
				s = FastSqrt(matrix.f11 - (matrix.f22 + matrix.f33) + 1.0f);
				if (s > QUAT_TRACE_QZERO_TOLERANCE)
				{
					quat.m_X = s * 0.5f;
					s = 0.5f / s;
					quat.m_W = (matrix.f32 - matrix.f23) * s;
					quat.m_Y = (matrix.f12 + matrix.f21) * s;
					quat.m_Z = (matrix.f13 + matrix.f31) * s;
					break;
				}
				break;

			case I:
				s = FastSqrt(matrix.f33 - (matrix.f11 + matrix.f22) + 1.0f);
				if (s > QUAT_TRACE_QZERO_TOLERANCE)
				{
					quat.m_Z = s * 0.5f;
					s = 0.5f / s;
					quat.m_W = (matrix.f21 - matrix.f12) * s;
					quat.m_X = (matrix.f31 + matrix.f13) * s;
					quat.m_Y = (matrix.f32 + matrix.f23) * s;
					break;
				}
				// A
				s = FastSqrt(matrix.f11 - (matrix.f22 + matrix.f33) + 1.0f);
				if (s > QUAT_TRACE_QZERO_TOLERANCE)
				{
					quat.m_X = s * 0.5f;
					s = 0.5f / s;
					quat.m_W = (matrix.f32 - matrix.f23) * s;
					quat.m_Y = (matrix.f12 + matrix.f21) * s;
					quat.m_Z = (matrix.f13 + matrix.f31) * s;
					break;
				}
				// E
				s = FastSqrt(matrix.f22 - (matrix.f33 + matrix.f11) + 1.0f);
				if (s > QUAT_TRACE_QZERO_TOLERANCE)
				{
					quat.m_Y = s * 0.5f;
					s = 0.5f / s;
					quat.m_W = (matrix.f13 - matrix.f31) * s;
					quat.m_Z = (matrix.f23 + matrix.f32) * s;
					quat.m_X = (matrix.f21 + matrix.f12) * s;
					break;
				}
				break;

			default:
				assert(0);
		}
	}
}

Void QuatToMatrix(RQuaternion quat, RMatrix4x4 matrix)
{
	Float X2,  Y2,  Z2;
	Float XX2, YY2, ZZ2;
	Float XY2, XZ2, XW2;
	Float YZ2, YW2, ZW2;

	X2  = 2.0f * quat.m_X;
	XX2 = X2   * quat.m_X;
	XY2 = X2   * quat.m_Y;
	XZ2 = X2   * quat.m_Z;
	XW2 = X2   * quat.m_W;

	Y2  = 2.0f * quat.m_Y;
	YY2 = Y2   * quat.m_Y;
	YZ2 = Y2   * quat.m_Z;
	YW2 = Y2   * quat.m_W;
	
	Z2  = 2.0f * quat.m_Z;
	ZZ2 = Z2   * quat.m_Z;
	ZW2 = Z2   * quat.m_W;
	
	matrix.f11 = 1.0f - YY2 - ZZ2;
	matrix.f12 = XY2  - ZW2;
	matrix.f13 = XZ2  + YW2;

	matrix.f21 = XY2  + ZW2;
	matrix.f22 = 1.0f - XX2 - ZZ2;
	matrix.f23 = YZ2  - XW2;

	matrix.f31 = XZ2  - YW2;
	matrix.f32 = YZ2  + XW2;
	matrix.f33 = 1.0f - XX2 - YY2;

	matrix.f14 = matrix.f24 = matrix.f34 = matrix.f14 = matrix.f42 = matrix.f43 = 0.0f;
	matrix.f44 = 1.0f;
}

Void QuatSlerp(RQuaternion q1, RQuaternion q2, Float t, RQuaternion qdst)
{
	Float omega, cosom, sinom, Scale1, Scale2;
	Quaternion qtmp;

	cosom = QuatMagnitude(q1, q2);
	if (cosom < 0)
	{
		cosom = -cosom;
		QuatNegCopy(q2, qtmp);
	}
	else
	{
		qtmp = q2;
	}
			

	if ((1.0f - cosom) > QUAT_EPSILON)
	{
		omega  = acosf(cosom);
		sinom  = sinf(omega);
		Scale1 = sinf((1.0f - t) * omega) / sinom;
		Scale2 = sinf(t * omega) / sinom;
	}
	else
	{
		Scale1 = 1.0f - t;
		Scale2 = t;
	}

	QuatDualScaleAdd(q1, Scale1, qtmp, Scale2, qdst);
}

Void QuatSlerpNotShortest(RQuaternion q1, RQuaternion q2, Float t, RQuaternion qdst)
{
	Float omega, cosom, sinom, Scale1, Scale2;

	cosom =	QuatMagnitude(q1, q2);
	if ((1.0f + cosom) > QUAT_EPSILON)
	{
		if ((1.0f - cosom) > QUAT_EPSILON)
		{
			omega = acosf(cosom);
			sinom = sinf(omega);
			if (sinom < QUAT_EPSILON)
			{
				Scale1 = 1.0f - t;
				Scale2 = t;
			}
			else
			{
				Scale1 = sinf((1.0f - t) * omega) / sinom;
				Scale2 = sinf(t * omega) / sinom;
			}
		}
		else
		{
			Scale1 = 1.0f - t;
			Scale2 = t;
		}
		
		QuatDualScaleAdd(q1, Scale1, q2, Scale2, qdst);
	}
	else
	{
		qdst.m_X = -q1.m_Y; 
		qdst.m_Y =  q1.m_X;
		qdst.m_Z = -q1.m_W;
		qdst.m_W =  q1.m_Z;
		
		Scale1 = sinf((1.0f - t) * (Float) CONST_PI_OVER_2);
		Scale2 = sinf(t * (Float) CONST_PI_OVER_2);

		QuatDualScaleAdd(q1, Scale1, qdst, Scale2, qdst);
	}
}

Void QuatMultiply(RQuaternion q1, RQuaternion q2, RQuaternion qdst)
{
	Quaternion qtmp;

	qtmp.m_W = ((q1.m_W * q2.m_W) - (q1.m_X * q2.m_X) - (q1.m_Y * q2.m_Y) - (q1.m_Z * q2.m_Z));
	qtmp.m_X = ((q1.m_W * q2.m_X) + (q1.m_X * q2.m_W) + (q1.m_Y * q2.m_Z) - (q1.m_Z * q2.m_Y));
	qtmp.m_Y = ((q1.m_W * q2.m_Y) - (q1.m_X * q2.m_Z) + (q1.m_Y * q2.m_W) + (q1.m_Z * q2.m_X));
	qtmp.m_Z = ((q1.m_W * q2.m_Z) + (q1.m_X * q2.m_Y) - (q1.m_Y * q2.m_X) + (q1.m_Z * q2.m_W));

	qdst = qtmp;
}

Void QuatRotateVector(RQuaternion quat, RVector3 vecin, RVector3 vecout)
{
	Quaternion qtmp;

	qtmp.m_W = (-(quat.m_X * vecin.m_X) - (quat.m_Y * vecin.m_Y) - (quat.m_Z * vecin.m_Z));
	qtmp.m_X = ( (quat.m_W * vecin.m_X) + (quat.m_Y * vecin.m_Z) - (quat.m_Z * vecin.m_Y));
	qtmp.m_Y = ( (quat.m_W * vecin.m_Y) - (quat.m_X * vecin.m_Z) + (quat.m_Z * vecin.m_X));
	qtmp.m_Z = ( (quat.m_W * vecin.m_Z) + (quat.m_X * vecin.m_Y) - (quat.m_Y * vecin.m_X));

	vecout.m_X = ((qtmp.m_Z * vecin.m_Y) - (qtmp.m_W * vecin.m_X) - (qtmp.m_Y * vecin.m_Z));
	vecout.m_Y = ((qtmp.m_X * vecin.m_Z) - (qtmp.m_W * vecin.m_Y) - (qtmp.m_Z * vecin.m_X));
	vecout.m_Z = ((qtmp.m_Y * vecin.m_X) - (qtmp.m_W * vecin.m_Z) - (qtmp.m_X * vecin.m_Y));
}

Void QuatLn(RQuaternion quat, RQuaternion qdst)
{
	Quaternion qtmp;

	if (quat.m_W < 0.0f)
	{
		QuatNegCopy(quat, qtmp);
	}
	else
	{
		qtmp = quat;
	}

	Float theta = acosf(qtmp.m_W);

	if (theta < QUAT_ZERO_EPSILON)
	{
		qdst.m_W = 0.0f;
		qdst.m_X = qtmp.m_X;
		qdst.m_Y = qtmp.m_Y;
		qdst.m_Z = qtmp.m_Z;
	}
	else
	{
		theta /= sinf(theta);

		qdst.m_W = 0.0f;
		qdst.m_X = theta * qtmp.m_X;
		qdst.m_Y = theta * qtmp.m_Y;
		qdst.m_Z = theta * qtmp.m_Z;
	}
}

Void QuatExp(RQuaternion quat, RQuaternion qdst)
{
	Float theta = FastSqrt(QuatLengthSquared(quat));
	Float SinThetaOverTheta;
	if (theta > QUAT_ZERO_EPSILON)
	{
		SinThetaOverTheta = SinCos(theta, &qdst.m_W) / theta;
	}
	else
	{
		qdst.m_W = cosf(theta);
		SinThetaOverTheta = 1.0f;
	}

	qdst.m_X = SinThetaOverTheta * quat.m_X;
	qdst.m_Y = SinThetaOverTheta * quat.m_Y;
	qdst.m_Z = SinThetaOverTheta * quat.m_Z;
}
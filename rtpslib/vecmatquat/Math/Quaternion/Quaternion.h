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


// Quaternion.h
//


#ifndef _3D_QUATERNION_H
#define _3D_QUATERNION_H


#include "Types.h"
#include "Math\Vector\Vector 3.h"
#include "Math\Matrix\Matrix 4x4.h"


typedef struct Quaternion Quaternion, *PQuaternion, &RQuaternion;

struct Quaternion
{
	Float m_W;
	Float m_X;
	Float m_Y;
	Float m_Z;
};


#define QuatClear(quat)								{ (quat).m_X = (quat).m_Y = (quat).m_Z = 0.0f; (quat).m_W = 1.0f; }
#define QuatSet(quat, w, x, y, z)					{ (quat).m_X = (x); (quat).m_Y = (y); (quat).m_Z = (z); (quat).m_W = (w); }
#define QuatSetVec(quat, w, vec)					{ QuatSet(quat, (vec).m_X, (vec).m_Y, (vec).m_Z, w); }
#define QuatMagnitude(q1, q2)						( ((q1).m_X * (q2).m_X) + ((q1).m_Y * (q2).m_Y) + ((q1).m_Z * (q2).m_Z) + ((q1).m_W * (q2).m_W))
#define QuatLengthSquared(quat)						( QuatMagnitude(quat, quat))
#define QuatNegCopy(q, qinv)						{ (qinv).m_X = -(q).m_X; (qinv).m_Y = -(q).m_Y; (qinv).m_Z = -(q).m_Z; (qinv).m_W = -(q).m_W; }
#define QuatNeg(quat)								{ QuatNegCopy(quat, quat); }
#define QuatDualScaleAdd(q1, q1s, q2, q2s, qdst)	{ (qdst).m_X = ((q1).m_X * (q1s)) + ((q2).m_X * (q2s)); (qdst).m_Y = ((q1).m_Y * (q1s)) + ((q2).m_Y * (q2s)); (qdst).m_Z = ((q1).m_Z * (q1s)) + ((q2).m_Z * (q2s)); (qdst).m_W = ((q1).m_W * (q1s)) + ((q2).m_W * (q2s)); }
#define QuatScale(quat, scale)						{ (quat).m_X *= scale; (quat).m_Y *= scale; (quat).m_Z *= scale; (quat).m_W *= scale; }
#define QuatInverseCopy(q, qdst)					{ (q).m_X = -(qdst).m_X; (q).m_Y = -(qdst).m_Y; (q).m_Z = -(qdst).m_Z; }
#define QuatInverse(quat)							{ QuatInverseCopy(quat, quat); }
#define QuatAdd(q1, q2, qdst)						{ (qdst).m_X = (q1).m_X + (q2).m_X; (qdst).m_Y = (q1).m_Y + (q2).m_Y; (qdst).m_Z = (q1).m_Z + (q2).m_Z; (qdst).m_W = (q1).m_W + (q2).m_W; }
#define QuatSub(q1, q2, qdst)						{ (qdst).m_X = (q1).m_X - (q2).m_X; (qdst).m_Y = (q1).m_Y - (q2).m_Y; (qdst).m_Z = (q1).m_Z - (q2).m_Z; (qdst).m_W = (q1).m_W - (q2).m_W; }
#define QuatCopy(quat, qdst)						{ (qdst) = (quat); }
#define QuatSwap(a, b)								{ Quaternion tmp = (a); (a) = (b); (b) = tmp; }

Bool QuatIsValid(RQuaternion quat);
Bool QuatIsUnit(RQuaternion quat);
Float QuatNormalize(RQuaternion quat);
Bool QuatCompare(RQuaternion q1, RQuaternion q2, Float fTolerance);
Void QuatSetFromAxisAngle(RQuaternion quat, RVector3 axis, Float theta);
Bool QuatGetAxisAngle(RQuaternion quat, RVector3 axis, PFloat theta);
Void QuatFromMatrix(RMatrix4x4 matrix, RQuaternion quat);
Void QuatToMatrix(RQuaternion quat, RMatrix4x4 matrix);
Void QuatSlerp(RQuaternion q1, RQuaternion q2, Float t, RQuaternion qdst);
Void QuatSlerpNotShortest(RQuaternion q1, RQuaternion q2, Float t, RQuaternion qdst);
Void QuatMultiply(RQuaternion q1, RQuaternion q2, RQuaternion qdst);
Void QuatRotateVector(RQuaternion quat, RVector3 vecin, RVector3 vecout);
Void QuatLn(RQuaternion quat, RQuaternion qdst);
Void QuatExp(RQuaternion quat, RQuaternion qdst);


class CQuaternion : public Quaternion
{
public:
	ForceInline CQuaternion()
	{ QuatClear(*this); }
	ForceInline CQuaternion(Float w, Float x, Float y, Float z)
	{ QuatSet(*this, x, y, z, w); }
	ForceInline CQuaternion(Float w, RVector3 vec)
	{ QuatSetVec(*this, w, vec); }

public:
	ForceInline Void Clear()
	{ QuatClear(*this); }

	ForceInline Void Set(Float w, Float x, Float y, Float z)
	{ QuatSet(*this, w, x, y, z); }

	ForceInline Void SetVec(Float w, RVector3 vec)
	{ QuatSetVec(*this, w, vec); }

	ForceInline Float Magnitude(RQuaternion q2)
	{ return (QuatMagnitude(*this, q2)); }

	ForceInline Float LengthSquared()
	{ return (QuatLengthSquared(*this)); }

	ForceInline Void NegCopy(RQuaternion qdst)
	{ QuatNegCopy(*this, qdst); }

	ForceInline Void Neg()
	{ QuatNeg(*this); }

	ForceInline Void DualScaleAdd(Float q1scale, RQuaternion q2, Float q2scale, RQuaternion qdst)
	{ QuatDualScaleAdd(*this, q1scale, q2, q2scale, qdst); }

	ForceInline Void Scale(Float scale)
	{ QuatScale(*this, scale); }

	ForceInline Void InverseCopy(RQuaternion qdst)
	{ QuatInverseCopy(*this, qdst); }

	ForceInline Void Inverse()
	{ QuatInverse(*this); }

	ForceInline Void Add(RQuaternion q2, RQuaternion qdst)
	{ QuatAdd(*this, q2, qdst); }
	ForceInline Void Sub(RQuaternion q2, RQuaternion qdst)
	{ QuatSub(*this, q2, qdst); }

	ForceInline Void Copy(RQuaternion qdst)
	{ QuatCopy(*this, qdst); }
	
	ForceInline Void Swap(RQuaternion b)
	{ QuatSwap(*(PQuaternion) this, b); }

	ForceInline Bool IsValid()
	{ return (QuatIsValid(*this)); }

	ForceInline Bool IsUnit()
	{ return (QuatIsUnit(*this)); }

	ForceInline Float Normalize()
	{ return (QuatNormalize(*this)); }

	ForceInline Bool Compare(RQuaternion q2, Float tolerance)
	{ return (QuatCompare(*this, q2, tolerance)); }
	
	ForceInline Void SetFromAxisAngle(RVector3 axis, Float theta)
	{ QuatSetFromAxisAngle(*this, axis, theta); }
	ForceInline Void GetAxisAngle(RVector3 axis, PFloat theta)
	{ QuatGetAxisAngle(*this, axis, theta); }

	ForceInline Void FromMatrix(RMatrix4x4 matrix)
	{ QuatFromMatrix(matrix, *this); }

	ForceInline Void ToMatrix(RMatrix4x4 mdst)
	{ QuatToMatrix(*this, mdst); }

	ForceInline Void Slerp(RQuaternion q2, Float t, RQuaternion qdst)
	{ QuatSlerp(*this, q2, t, qdst); }
	ForceInline Void SlerpNotShortest(RQuaternion q2, Float t, RQuaternion qdst)
	{ QuatSlerpNotShortest(*this, q2, t, qdst); }

	ForceInline Void Multiply(RQuaternion q2, RQuaternion qdst)
	{ QuatMultiply(*this, q2, qdst); }

	ForceInline Void RotateVector(RVector3 vecin, RVector3 vecout)
	{ QuatRotateVector(*this, vecin, vecout); }

	ForceInline Void Ln(RQuaternion qdst)
	{ QuatLn(*this, qdst); }

	ForceInline Void Exp(RQuaternion qdst)
	{ QuatExp(*this, qdst); }

public:
	ForceInline operator PQuaternion ()
	{ return ((PQuaternion) this); }
	ForceInline operator *= (RQuaternion right)
	{ Multiply(right, *this); }
};


#endif
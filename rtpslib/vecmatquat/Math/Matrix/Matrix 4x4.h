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


// Matrix 4x4.h
//


#ifndef _3D_MATRIX4X4_H
#define _3D_MATRIX4X4_H


#include "Types.h"
#include "Math\Vector\Vector 3.h"


typedef struct Matrix4x4 Matrix4x4, *PMatrix4x4, &RMatrix4x4;

struct Matrix4x4
{
    Float f11, f21, f31, f41;
	Float f12, f22, f32, f42;
	Float f13, f23, f33, f43;
	Float f14, f24, f34, f44;
};


//  Matrix order:
//
//  _11  _12  _13  _14
//  _21  _22  _23  _24
//  _31  _32  _33  _34
//  _41  _42  _43  _44


extern const Matrix4x4 IdentityMatrix;

#define MatrixSetIdentity(m)					{ (m) = IdentityMatrix; }
#define MatrixSetScaling(m, x, y, z)			{ (m).f11 = (x); (m).f22 = (y); (m).f33 = (z); (m).f12 = (m).f13 = (m).f14 = (m).f21 = (m).f23 = (m).f24 = (m).f31 = (m).f32 = (m).f34 = (m).f41 = (m).f42 = (m).f43 = 0.0f; (m).f44 = 1.0f; }
#define MatrixSetScalingVec(m, vec)				{ MatrixSetScaling(m, (vec).m_X, (vec).m_Y, (vec).m_Z); }
#define MatrixSetTranslation(m, x, y, z)		{ (m).f14 = (x); (m).f24 = (y); (m).f34 = (z); (m).f11 = (m).f22 = (m).f33 = (m).f44 = 1.0f; (m).f12 = (m).f13 = (m).f21 = (m).f23 = (m).f31 = (m).f32 = (m).f41 = (m).f42 = (m).f43 = 0.0f; }
#define MatrixSetTranslationVec(m, vec)			{ MatrixSetTranslation(m, (vec).m_X, (vec).m_Y, (vec).m_Z); }
#define MatrixSetLeftUpIn(m, left, up, in)		{ (m).f11 = -(left).m_X; (m).f12 = -(left).m_Y; (m).f13 = -(left).m_Z; (m).f21 = (up).m_X;	(m).f22 = (up).m_Y;	(m).f23 = (up).m_Z; (m).f31 = -(in).m_X; (m).f32 = -(in).m_Y; (m).f33 = -(in).m_Z; (m).f14 = (m).f24 = (m).f34 = (m).f41 = (m).f42 = (m).f43 = 0.0f; (m).f44 = 1.0f; }
#define MatrixGetLeftVec(m, vecleft)			{ (vecleft).m_X = -(m).f11; (vecleft).m_Y = -(m).f21; (vecleft).m_Z = -(m).f31; }
#define MatrixGetUpVec(m, vecup)				{ (vecup).m_X = (m).f12; (vecup).m_Y = (m).f22; (vecup).m_Z = (m).f32; }
#define MatrixGetInVec(m, vecin)				{ (vecin).m_X = -(m).f13; (vecin).m_Y = -(m).f23; (vecin).m_Z = -(m).f33; }
#define MatrixGetLeftUpIn(m, left, up, in)		{ MatrixGetLeftVec(m, left); MatrixGetUpVec(m, up); MatrixGetInVec(m, in); }
#define MatrixGetTranspose(m, mtrans)			{ (mtrans).f11 = (m).f11; (mtrans).f12 = (m).f21; (mtrans).f13 = (m).f31;	(mtrans).f14 = -(m).f14; (mtrans).f21 = (m).f12; (mtrans).f22 = (m).f22; (mtrans).f23 = (m).f32; (mtrans).f24 = -(m).f24; (mtrans).f31 = (m).f13; (mtrans).f32 = (m).f23; (mtrans).f33 = (m).f33; (mtrans).f34 = -(m).f34; (mtrans).f41 = (m).f41; (mtrans).f42 = (m).f42; (mtrans).f43 = (m).f43; (mtrans).f44 = -(m).f44; }
#define MatrixTranslate(m, x, y, z)				{ (m).f14 += (x); (m).f24 += (y); (m).f34 += (z); }
#define MatrixTranslateVec(m, vec)				{ MatrixTranslate(m, (vec).m_X, (vec).m_Y, (vec).m_Z); }
#define MatrixScale(m, x, y, z)					{ (m).f11 *= (x); (m).f12 *= (x); (m).f13 *= (x); (m).f14 *= (x); (m).f21 *= (y); (m).f22 *= (y); (m).f23 *= (y); (m).f24 *= (y); (m).f31 *= (z); (m).f32 *= (z); (m).f33 *= (z); (m).f34 *= (z); }
#define MatrixScaleVec(m, vec)					{ MatrixScale(m, (vec).m_X, (vec).m_Y, (vec).m_Z); }
#define MatrixRotateVector(m, vecin, vecout)	{ (vecout).m_X = ((vecin).m_X * (m).f11) + ((vecin).m_Y * (m).f12) + ((vecin).m_Z * (m).f13); (vecout).m_Y = ((vecin).m_X * (m).f21) + ((vecin).m_Y * (m).f22) + ((vecin).m_Z * (m).f23); (vecout).m_Z = ((vecin).m_X * (m).f31) + ((vecin).m_Y * (m).f32) + ((vecin).m_Z * (m).f33); }
#define MatrixTransformVector(m, vecin, vecout)	{ (vecout).m_X = ((vecin).m_X * (m).f11) + ((vecin).m_Y * (m).f12) + ((vecin).m_Z * (m).f13) + (m).f14; (vecout).m_Y = ((vecin).m_X * (m).f21) + ((vecin).m_Y * (m).f22) + ((vecin).m_Z * (m).f23) + (m).f24; (vecout).m_Z = ((vecin).m_X * (m).f31) + ((vecin).m_Y * (m).f32) + ((vecin).m_Z * (m).f33) + (m).f34; }
#define MatrixProjectVector(m, vecin, vecout)	{ Float rhw = 1.0f / (((vecin).m_X * (m).f41) + ((vecin).m_Y * (m).f42) + ((vecin).m_Z * (m).f43) + (m).f44); (vecout).m_X = (((vecin).m_X * (m).f11) + ((vecin).m_Y * (m).f12) + ((vecin).m_Z * (m).f13) + (m).f14) * rhw; (vecout).m_Y = (((vecin).m_X * (m).f21) + ((vecin).m_Y * (m).f22) + ((vecin).m_Z * (m).f23) + (m).f24) * rhw; (vecout).m_Z = (((vecin).m_X * (m).f31) + ((vecin).m_Y * (m).f32) + ((vecin).m_Z * (m).f33) + (m).f34) * rhw; }
#define MatrixCopy(m, mdst)						{ (mdst) = (m); }
#define MatrixSwap(a, b)						{ Matrix4x4 tmp = (a); (a) = (b); (b) = tmp; }

Bool MatrixIsValid(RMatrix4x4 matrix);
Bool MatrixIsOrthonormal(RMatrix4x4 matrix);
Bool MatrixIsOrthogonal(RMatrix4x4 matrix);
Void MatrixOrthonormalize(RMatrix4x4 matrix);
Void MatrixSetXRotation(RMatrix4x4 matrix, Float m_XAngle);
Void MatrixSetYRotation(RMatrix4x4 matrix, Float m_YAngle);
Void MatrixSetZRotation(RMatrix4x4 matrix, Float m_ZAngle);
Void MatrixSetXYZRotation(RMatrix4x4 matrix, Float m_XAngle, Float m_YAngle, Float m_ZAngle);
Void MatrixSetRotationScalingTranslation(RMatrix4x4 matrix, RVector3 vRotation, RVector3 vScaling, RVector3 vTranslation);
Bool MatrixSetViewMatrix(RMatrix4x4 matrix, RVector3 vFrom, RVector3 vView, RVector3 vWorldUp);
Void MatrixSetOrthogonalMatrix(RMatrix4x4 matrix, Float fLeft, Float fRight, Float fBottom, Float fTop, Float fNear, Float fFar);
Void MatrixSetOrthogonal2DMatrix(RMatrix4x4 matrix, Float fLeft, Float fRight, Float fBottom, Float fTop);
Void MatrixSetFrustumMatrix(RMatrix4x4 matrix, Float fLeft, Float fRight, Float fBottom, Float fTop, Float fNear, Float fFar);
Void MatrixSetPerspectiveMatrix(RMatrix4x4 matrix, Float fFov, Float fAspect, Float fNear, Float fFar);
Void MatrixAdd(RMatrix4x4 m1, RMatrix4x4 m2, RMatrix4x4 mdest);
Void MatrixMultiply(RMatrix4x4 m1, RMatrix4x4 m2, RMatrix4x4 mdest);
Void MatrixRotateX(RMatrix4x4 matrix, Float m_XAngle);
Void MatrixRotateY(RMatrix4x4 matrix, Float m_YAngle);
Void MatrixRotateZ(RMatrix4x4 matrix, Float m_ZAngle);
Void MatrixRotateXYZ(RMatrix4x4 matrix, Float m_XAngle, Float m_YAngle, Float m_ZAngle);
Void MatrixGetXYZRotationAngles(RMatrix4x4 matrix, RVector3 vAngles);
Void MatrixRotateVectorArray(RMatrix4x4 matrix, PVector3 vIn, PVector3 vOut, UInt uiNumVecs);
Void MatrixTransformVectorArray(RMatrix4x4 matrix, PVector3 vIn, PVector3 vOut, UInt uiNumVecs);
Void MatrixProjectVectorArray(RMatrix4x4 matrix, PVector3 vIn, PVector3 vOut, UInt uiNumVecs);


class CMatrix4x4 : public Matrix4x4
{
public:
	ForceInline CMatrix4x4()
	{ SetIdentity(); }

public:
	ForceInline Void SetIdentity()		
	{ MatrixSetIdentity(*(PMatrix4x4) this); }
	
	ForceInline Void SetScaling(Float x, Float y, Float z)
	{ MatrixSetScaling(*this, x, y, z); }
	ForceInline Void SetScaling(RVector3 vec)
	{ MatrixSetScalingVec(*this, vec); }

	ForceInline Void SetTranslation(Float x, Float y, Float z)
	{ MatrixSetTranslation(*this, x, y, z); }
	ForceInline Void SetTranslation(RVector3 vec)
	{ MatrixSetTranslationVec(*this, vec); }

	ForceInline Void SetLeftUpIn(RVector3 left, RVector3 up, RVector3 in)
	{ MatrixSetLeftUpIn(*this, left, up, in); }

	ForceInline Void GetLeftUpIn(RVector3 left, RVector3 up, RVector3 in)
	{ MatrixGetLeftUpIn(*this, left, up, in); }

	ForceInline Void GetLeftVec(RVector3 left)
	{ MatrixGetLeftVec(*this, left); }
	ForceInline Void GetUpVec(RVector3 up)
	{ MatrixGetUpVec(*this, up); }
	ForceInline Void GetInVec(RVector3 in)
	{ MatrixGetInVec(*this, in); }

	ForceInline Void GetTranspose(RMatrix4x4 mdst)
	{ MatrixGetTranspose(*this, mdst); }

	ForceInline Void Translate(Float x, Float y, Float z)
	{ MatrixTranslate(*this, x, y, z); }
	ForceInline Void Translate(RVector3 vec)
	{ MatrixTranslateVec(*this, vec); }

	ForceInline Void Scale(Float x, Float y, Float z)
	{ MatrixScale(*this, x, y, z); }
	ForceInline Void Scale(RVector3 vec)
	{ MatrixScaleVec(*this, vec); }

	ForceInline Void RotateVector(RVector3 vecin, RVector3 vecout)
	{ MatrixRotateVector(*this, vecin, vecout); }
	ForceInline Void TransformVector(RVector3 vecin, RVector3 vecout)
	{ MatrixTransformVector(*this, vecin, vecout); }
	ForceInline Void ProjectVector(RVector3 vecin, RVector3 vecout)
	{ MatrixProjectVector(*this, vecin, vecout); }

	ForceInline Void Copy(RMatrix4x4 mdst)
	{ MatrixCopy(*this, mdst); }

	ForceInline Void Swap(RMatrix4x4 m2)
	{ MatrixSwap(*(PMatrix4x4) this, m2); }

	ForceInline Bool IsValid()
	{ return (MatrixIsValid(*this)); }
	ForceInline Bool IsOrthonormal()
	{ return (MatrixIsOrthonormal(*this)); }
	ForceInline Bool IsOrthogonal()
	{ return (MatrixIsOrthogonal(*this)); }

	ForceInline Void Orthonormalize()
	{ MatrixOrthonormalize(*this); }

	ForceInline Void SetXRotation(Float x)
	{ MatrixSetXRotation(*this, x); }
	ForceInline Void SetYRotation(Float y)
	{ MatrixSetYRotation(*this, y); }
	ForceInline Void SetZRotation(Float z)
	{ MatrixSetZRotation(*this, z); }

	ForceInline Void SetRotationScalingTranslation(RVector3 rotation, RVector3 scaling, RVector3 translation)
	{ MatrixSetRotationScalingTranslation(*this, rotation, scaling, translation); }

	ForceInline Void SetViewMatrix(RVector3 from, RVector3 view, RVector3 worldup)
	{ MatrixSetViewMatrix(*this, from, view, worldup); }

	ForceInline Void SetOrthogonalMatrix(Float left, Float right, Float bottom, Float top, Float znear, Float zfar)
	{ MatrixSetOrthogonalMatrix(*this, left, right, bottom, top, znear, zfar); }
	
	ForceInline Void SetOrthogonal2DMatrix(Float left, Float right, Float bottom, Float top)
	{ MatrixSetOrthogonal2DMatrix(*this, left, right, bottom, top); }

	ForceInline Void SetFrustumMatrix(Float left, Float right, Float bottom, Float top, Float znear, Float zfar)
	{ MatrixSetFrustumMatrix(*this, left, right, bottom, top, znear, zfar); }

	ForceInline Void SetPerspectiveMatrix(Float fov, Float aspect, Float znear, Float zfar)
	{ MatrixSetPerspectiveMatrix(*this, fov, aspect, znear, zfar); }

	ForceInline Void Add(RMatrix4x4 m2, RMatrix4x4 mdst)
	{ MatrixAdd(*this, m2, mdst); }

	ForceInline Void Multiply(RMatrix4x4 m2, RMatrix4x4 mdst)
	{ MatrixMultiply(*this, m2, mdst); }

	ForceInline Void RotateX(Float x)
	{ MatrixRotateX(*this, x); }
	ForceInline Void RotateY(Float y)
	{ MatrixRotateY(*this, y); }
	ForceInline Void RotateZ(Float z)
	{ MatrixRotateZ(*this, z); }

	ForceInline Void RotateXYZ(Float x, Float y, Float z)
	{ MatrixRotateXYZ(*this, x, y, z); }
	ForceInline Void RotateXYZ(RVector3 vec)
	{ MatrixRotateXYZ(*this, vec.m_X, vec.m_Y, vec.m_Z); }

	ForceInline Void GetXYZRotationAngles(RVector3 angles)
	{ MatrixGetXYZRotationAngles(*this, angles); }

	ForceInline Void RotateVectorArray(PVector3 vecin, PVector3 vecout, UInt count)
	{ MatrixRotateVectorArray(*this, vecin, vecout, count); }
	ForceInline Void TransformVectorArray(PVector3 vecin, PVector3 vecout, UInt count)
	{ MatrixTransformVectorArray(*this, vecin, vecout, count); }
	ForceInline Void ProjectVectorArray(PVector3 vecin, PVector3 vecout, UInt count)
	{ MatrixProjectVectorArray(*this, vecin, vecout, count); }

public:
	ForceInline operator PMatrix4x4()
	{ return ((PMatrix4x4) this); }
	ForceInline operator += (RMatrix4x4 right)
	{ Add(right, *this); }
	ForceInline operator *= (RMatrix4x4 right)
	{ Multiply(right, *this); }
};


#endif
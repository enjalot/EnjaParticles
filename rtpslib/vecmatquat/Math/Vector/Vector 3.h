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


// Vector 3.h
//


#ifndef _3D_VECTOR3D_H
#define _3D_VECTOR3D_H


#include "Types.h"


typedef struct Vector3 Vector3, *PVector3, &RVector3;

struct Vector3
{
	Float m_X;
	Float m_Y;
	Float m_Z;
};


#define Vec3Set(vec, x, y, z)				{ (vec).m_X = (x); (vec).m_Y = (y); (vec).m_Z = (z); }
#define Vec3Clear(a)						{ (a).m_X = (a).m_Y = (a).m_Z = 0.0f; }
#define Vec3DotProduct(a, b)				(((a).m_X * (b).m_X) + ((a).m_Y * (b).m_Y) + ((a).m_Z * (b).m_Z))
#define Vec3LengthSquared(a)				(Vec3DotProduct((a), (a)))
#define Vec3Add(a, b, res)					{ (res).m_X = (a).m_X + (b).m_X; (res).m_Y = (a).m_Y + (b).m_Y; (res).m_Z = (a).m_Z + (b).m_Z; }
#define Vec3Sub(a, b, res)					{ (res).m_X = (a).m_X - (b).m_X; (res).m_Y = (a).m_Y - (b).m_Y; (res).m_Z = (a).m_Z - (b).m_Z; }
#define Vec3Invert(a)						{ (a).m_X = -(a).m_X; (a).m_Y = -(a).m_Y; (a).m_Z = -(a).m_Z; }
#define Vec3Scale(a, factor)				{ (a).m_X *= (factor); (a).m_Y *= (factor); (a).m_Z *= (factor); }
#define Vec3Copy(src, dst)					{ (dst) = (src); }
#define Vec3CrossProduct(a, b, dst)			{ (dst).m_X = ((a).m_Y * (b).m_Z) - ((a).m_Z * (b).m_Y); (dst).m_Y = ((a).m_Z * (b).m_X) - ((a).m_X * (b).m_Z); (dst).m_Z = ((a).m_X * (b).m_Y) - ((a).m_Y * (b).m_X); }
#define Vec3Normal(a, b, c, dst)			{ Vector3 vdif1, vdif2;	Vec3Sub(b, a, vdif1); Vec3Sub(c, a, vdif2); Vec3CrossProduct(vdif1, vdif2, dst); }
#define Vec3Swap(a, b)						{ Vector3 tmp = (a); (a) = (b); (b) = tmp; }
#define Vec3DistBetweenSquared(a, b, dist)	{ Vector3 tmp; Vec3Sub((a), (b), tmp); dist = Vec3LengthSquared(tmp); }
#define Vec3AddScaled(a, b, bscale, dst)	{ (dst).m_X = (a).m_X + ((b).m_X * (bscale)); (dst).m_Y = (a).m_Y + ((b).m_Y * (bscale)); (dst).m_Z = (a).m_Z + ((b).m_Z * (bscale)); }
#define Vec3SubScaled(a, b, bscale, dst)	{ (dst).m_X = (a).m_X - ((b).m_X * (bscale)); (dst).m_Y = (a).m_Y - ((b).m_Y * (bscale)); (dst).m_Z = (a).m_Z - ((b).m_Z * (bscale)); }

Bool  Vec3IsValid(RVector3 vec);
Bool  Vec3IsNormalized(RVector3 vec);
Float Vec3Normalize(RVector3 vec);
Float Vec3Length(RVector3 vec);
Float Vec3DistanceBetween(RVector3 v1, RVector3 v2);
Bool  Vec3Compare(RVector3 v1, RVector3 v2, Float fTolerance);


class CVector3 : public Vector3
{
public:
	ForceInline CVector3()
	{ Vec3Clear(*this); }
	ForceInline CVector3(Float x, Float y, Float z)
	{ Vec3Set(*this, x, y, z); }

public:
	ForceInline Void Set(Float x, Float y, Float z)
	{ Vec3Set(*this, x, y, z); }
	
	ForceInline Void Clear()
	{ Vec3Clear(*this); }

	ForceInline Float DotProduct(RVector3 v2)
	{ return (Vec3DotProduct(*this, v2)); }

	ForceInline Float LengthSquared()
	{ return (Vec3LengthSquared(*this)); }

	ForceInline Void Add(RVector3 v2, RVector3 vdst)
	{ Vec3Add(*this, v2, vdst); }

	ForceInline Void Sub(RVector3 v2, RVector3 vdst)
	{ Vec3Sub(*this, v2, vdst); }

	ForceInline Void Invert()
	{ Vec3Invert(*this); }

	ForceInline Void Scale(Float factor)
	{ Vec3Scale(*this, factor); }

	ForceInline Void Copy(RVector3 vdst)
	{ Vec3Copy(*this, vdst); }

	ForceInline Void CrossProduct(RVector3 v2, RVector3 vdst)
	{ Vec3CrossProduct(*this, v2, vdst);	}

	ForceInline Void Normal(RVector3 v2, RVector3 v3, RVector3 vdst)
	{ Vec3Normal(*this, v2, v3, vdst); }

	ForceInline Void Swap(RVector3 v2)
	{ Vec3Swap(*(PVector3) this, v2); }

	ForceInline Float DistBetweenSquared(RVector3 v2)
	{ Float distsq; Vec3DistBetweenSquared(*this, v2, distsq); return distsq; }

	ForceInline Void AddScaled(RVector3 v2, Float v2scale, RVector3 vdst)
	{ Vec3AddScaled(*this, v2, v2scale, vdst); }
	ForceInline Void SubScaled(RVector3 v2, Float v2scale, RVector3 vdst)
	{ Vec3SubScaled(*this, v2, v2scale, vdst); }

	ForceInline Bool IsValid()
	{ return (Vec3IsValid(*this)); }

	ForceInline Bool IsNormalized()
	{ return (Vec3IsNormalized(*this)); }

	ForceInline Float Normalize()
	{ return (Vec3Normalize(*this)); }

	ForceInline Float Length()
	{ return (Vec3Length(*this)); }

	ForceInline Float DistanceBetween(RVector3 v2)
	{ return (Vec3DistanceBetween(*this, v2)); }

	ForceInline Bool Compare(RVector3 v2, Float tolerance)
	{ return (Vec3Compare(*this, v2, tolerance)); }

public:
	ForceInline operator PVector3 ()
	{ return ((PVector3) this); }
	ForceInline operator += (RVector3 right)
	{ Add(right, *this); }
	ForceInline operator -= (RVector3 right)
	{ Sub(right, *this); }
	ForceInline operator *= (Float right)
	{ Scale(right); }
};


#endif
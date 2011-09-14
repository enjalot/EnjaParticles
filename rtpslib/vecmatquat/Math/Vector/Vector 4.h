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


// Vector 4.h
//


#ifndef _3D_VECTOR4D_H
#define _3D_VECTOR4D_H


#include "Types.h"


typedef struct Vector4 Vector4, *PVector4, &RVector4;

struct Vector4
{
	Float m_X;
	Float m_Y;
	Float m_Z;
	Float m_W;
};


#define Vec4Set(vec, x, y, z, w)			{ (vec).m_X = (x); (vec).m_Y = (y); (vec).m_Z = (z); (vec).m_W = (w); }
#define Vec4Clear(a)						{ (a).m_X = (a).m_Y = (a).m_Z = (a).m_W = 0.0f; }
#define Vec4DotProduct(a, b)				(((a).m_X * (b).m_X) + ((a).m_Y * (b).m_Y) + ((a).m_Z * (b).m_Z) + ((a).m_W * (a).m_W))
#define Vec4LengthSquared(a)				(Vec4DotProduct((a), (a)))
#define Vec4Add(a, b, res)					{ (res).m_X = (a).m_X + (b).m_X; (res).m_Y = (a).m_Y + (b).m_Y; (res).m_Z = (a).m_Z + (b).m_Z; (res).m_W = (a).m_W + (b).m_W; }
#define Vec4Sub(a, b, res)					{ (res).m_X = (a).m_X - (b).m_X; (res).m_Y = (a).m_Y - (b).m_Y; (res).m_Z = (a).m_Z - (b).m_Z; (res).m_W = (a).m_W - (b).m_W; }
#define Vec4Invert(a)						{ (a).m_X = -(a).m_X; (a).m_Y = -(a).m_Y; (a).m_Z = -(a).m_Z; (a).m_W = -(a).m_W; }
#define Vec4Scale(a, factor)				{ (a).m_X *= (factor); (a).m_Y *= (factor); (a).m_Z *= (factor); (a).m_W *= (factor); }
#define Vec4Copy(src, dst)					{ (dst) = (src); }
#define Vec4Swap(a, b)						{ Vector4 tmp = (a); (a) = (b); (b) = tmp; }
#define Vec4DistBetweenSquared(a, b, dist)	{ Vector4 tmp; Vec4Sub((a), (b), tmp); dist = Vec4LengthSquared(tmp); }
#define Vec4AddScaled(a, b, bscale, dst)	{ (dst).m_X = (a).m_X + ((b).m_X * (bscale)); (dst).m_Y = (a).m_Y + ((b).m_Y * (bscale)); (dst).m_Z = (a).m_Z + ((b).m_Z * (bscale)); (dst).m_W = (a).m_W + ((b).m_W * (bscale)); }
#define Vec4SubScaled(a, b, bscale, dst)	{ (dst).m_X = (a).m_X - ((b).m_X * (bscale)); (dst).m_Y = (a).m_Y - ((b).m_Y * (bscale)); (dst).m_Z = (a).m_Z - ((b).m_Z * (bscale)); (dst).m_W = (a).m_W - ((b).m_W * (bscale)); }

Bool  Vec4IsValid(RVector4 vec);
Bool  Vec4IsNormalized(RVector4 vec);
Float Vec4Normalize(RVector4 vec);
Float Vec4Length(RVector4 vec);
Float Vec4DistanceBetween(RVector4 v1, RVector4 v2);
Bool  Vec4Compare(RVector4 v1, RVector4 v2, Float fTolerance);


class CVector4 : public Vector4
{
public:
	ForceInline CVector4()
	{ Vec4Clear(*this); }
	ForceInline CVector4(Float x, Float y, Float z, Float w)
	{ Vec4Set(*this, x, y, z, w); }

public:
	ForceInline Void Set(Float x, Float y, Float z, Float w)
	{ Vec4Set(*this, x, y, z, w); }
	
	ForceInline Void Clear()
	{ Vec4Clear(*this); }

	ForceInline Float DotProduct(RVector4 v2)
	{ return (Vec4DotProduct(*this, v2)); }

	ForceInline Float LengthSquared()
	{ return (Vec4LengthSquared(*this)); }

	ForceInline Void Add(RVector4 v2, RVector4 vdst)
	{ Vec4Add(*this, v2, vdst); }

	ForceInline Void Sub(RVector4 v2, RVector4 vdst)
	{ Vec4Sub(*this, v2, vdst); }

	ForceInline Void Invert()
	{ Vec4Invert(*this); }

	ForceInline Void Scale(Float factor)
	{ Vec4Scale(*this, factor); }

	ForceInline Void Copy(RVector4 vdst)
	{ Vec4Copy(*this, vdst); }

	ForceInline Void Swap(RVector4 v2)
	{ Vec4Swap(*(PVector4) this, v2); }

	ForceInline Float DistBetweenSquared(RVector4 v2)
	{ Float distsq; Vec4DistBetweenSquared(*this, v2, distsq); return distsq; }

	ForceInline Void AddScaled(RVector4 v2, Float v2scale, RVector4 vdst)
	{ Vec4AddScaled(*this, v2, v2scale, vdst); }
	ForceInline Void SubScaled(RVector4 v2, Float v2scale, RVector4 vdst)
	{ Vec4SubScaled(*this, v2, v2scale, vdst); }

	ForceInline Bool IsValid()
	{ return (Vec4IsValid(*this)); }

	ForceInline Bool IsNormalized()
	{ return (Vec4IsNormalized(*this)); }

	ForceInline Float Normalize()
	{ return (Vec4Normalize(*this)); }

	ForceInline Float Length()
	{ return (Vec4Length(*this)); }

	ForceInline Float DistanceBetween(RVector4 v2)
	{ return (Vec4DistanceBetween(*this, v2)); }

	ForceInline Bool Compare(RVector4 v2, Float tolerance)
	{ return (Vec4Compare(*this, v2, tolerance)); }

public:
	ForceInline operator PVector4 ()
	{ return ((PVector4) this); }
	ForceInline operator += (RVector4 right)
	{ Add(right, *this); }
	ForceInline operator -= (RVector4 right)
	{ Sub(right, *this); }
	ForceInline operator *= (Float right)
	{ Scale(right); }
};


#endif
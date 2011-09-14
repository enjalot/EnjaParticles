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


#ifndef _3D_VECTOR2D_H
#define _3D_VECTOR2D_H


#include "Types.h"


typedef struct Vector2 Vector2, *PVector2, &RVector2;

struct Vector2
{
	Float m_X;
	Float m_Y;
};


#define Vec2Set(vec, x, y)					{ (vec).m_X = (x); (vec).m_Y = (y); }
#define Vec2Clear(a)						{ (a).m_X = (a).m_Y = 0.0f; }
#define Vec2DotProduct(a, b)				(((a).m_X * (b).m_X) + ((a).m_Y * (b).m_Y))
#define Vec2LengthSquared(a)				(Vec2DotProduct((a), (a)))
#define Vec2Add(a, b, res)					{ (res).m_X = (a).m_X + (b).m_X; (res).m_Y = (a).m_Y + (b).m_Y; }
#define Vec2Sub(a, b, res)					{ (res).m_X = (a).m_X - (b).m_X; (res).m_Y = (a).m_Y - (b).m_Y; }
#define Vec2Invert(a)						{ (a).m_X = -(a).m_X; (a).m_Y = -(a).m_Y; }
#define Vec2Scale(a, factor)				{ (a).m_X *= (factor); (a).m_Y *= (factor); }
#define Vec2Copy(src, dst)					{ (dst) = (src); }
#define Vec2Swap(a, b)						{ Vector2 tmp = (a); (a) = (b); (b) = tmp; }
#define Vec2DistBetweenSquared(a, b, dist)	{ Vector2 tmp; Vec2Sub((a), (b), tmp); dist = Vec2LengthSquared(tmp); }
#define Vec2AddScaled(a, b, bscale, dst)	{ (dst).m_X = (a).m_X + ((b).m_X * (bscale)); (dst).m_Y = (a).m_Y + ((b).m_Y * (bscale)); }
#define Vec2SubScaled(a, b, bscale, dst)	{ (dst).m_X = (a).m_X - ((b).m_X * (bscale)); (dst).m_Y = (a).m_Y - ((b).m_Y * (bscale)); }

Bool  Vec2IsValid(RVector2 vec);
Bool  Vec2IsNormalized(RVector2 vec);
Float Vec2Normalize(RVector2 vec);
Float Vec2Length(RVector2 vec);
Float Vec2DistanceBetween(RVector2 v1, RVector2 v2);
Bool  Vec2Compare(RVector2 v1, RVector2 v2, Float fTolerance);


class CVector2 : public Vector2
{
public:
	ForceInline CVector2()
	{ Vec2Clear(*this); }
	ForceInline CVector2(Float x, Float y)
	{ Vec2Set(*this, x, y); }

public:
	ForceInline Void Set(Float x, Float y)
	{ Vec2Set(*this, x, y); }
	
	ForceInline Void Clear()
	{ Vec2Clear(*this); }

	ForceInline Float DotProduct(RVector2 v2)
	{ return (Vec2DotProduct(*this, v2)); }

	ForceInline Float LengthSquared()
	{ return (Vec2LengthSquared(*this)); }

	ForceInline Void Add(RVector2 v2, RVector2 vdst)
	{ Vec2Add(*this, v2, vdst); }

	ForceInline Void Sub(RVector2 v2, RVector2 vdst)
	{ Vec2Sub(*this, v2, vdst); }

	ForceInline Void Invert()
	{ Vec2Invert(*this); }

	ForceInline Void Scale(Float factor)
	{ Vec2Scale(*this, factor); }

	ForceInline Void Copy(RVector2 vdst)
	{ Vec2Copy(*this, vdst); }

	ForceInline Void Swap(RVector2 v2)
	{ Vec2Swap(*(PVector2) this, v2); }

	ForceInline Float DistBetweenSquared(RVector2 v2)
	{ Float distsq; Vec2DistBetweenSquared(*this, v2, distsq); return distsq; }

	ForceInline Void AddScaled(RVector2 v2, Float v2scale, RVector2 vdst)
	{ Vec2AddScaled(*this, v2, v2scale, vdst); }
	ForceInline Void SubScaled(RVector2 v2, Float v2scale, RVector2 vdst)
	{ Vec2SubScaled(*this, v2, v2scale, vdst); }

	ForceInline Bool IsValid()
	{ return (Vec2IsValid(*this)); }

	ForceInline Bool IsNormalized()
	{ return (Vec2IsNormalized(*this)); }

	ForceInline Float Normalize()
	{ return (Vec2Normalize(*this)); }

	ForceInline Float Length()
	{ return (Vec2Length(*this)); }

	ForceInline Float DistanceBetween(RVector2 v2)
	{ return (Vec2DistanceBetween(*this, v2)); }

	ForceInline Bool Compare(RVector2 v2, Float tolerance)
	{ return (Vec2Compare(*this, v2, tolerance)); }

public:
	ForceInline operator PVector2 ()
	{ return ((PVector2) this); }
	ForceInline operator += (RVector2 right)
	{ Add(right, *this); }
	ForceInline operator -= (RVector2 right)
	{ Sub(right, *this); }
	ForceInline operator *= (Float right)
	{ Scale(right); }
};


#endif
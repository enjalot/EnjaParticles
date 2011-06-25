/*******************************************************************************
 File:				Quat.inline.h

 Author: 			Gaspard Petit (gaspardpetit@gmail.com)
 Last Revision:		March 14, 2007
 
 This code may be reused without my permission as long as credits are given to
 the original author.  If you find bugs, please send me a note...
*******************************************************************************/
#ifndef __MK_GEOMETRY_QUAT_INLINE__
#define __MK_GEOMETRY_QUAT_INLINE__


//==============================================================================
//	EXTERNAL DECLARATIONS
//==============================================================================
#include "Quat.h"
#include <cmath>

//==============================================================================
//	CLASS Quat
//==============================================================================
//------------------------------------------------------------------------------
//
template<class T>
Quat<T>::Quat()
:
	w(),
	x(),
	y(),
	z()
{}

//------------------------------------------------------------------------------
//
template<class T>
Quat<T>::Quat(T inW, T inX, T inY, T inZ)
:
	w(inW),
	x(inX),
	y(inY),
	z(inZ)
{}

//------------------------------------------------------------------------------
//
template<class T>
Quat<T>::Quat(const Point<T,3> &axis, T angle)
:
	w(),
	x(),
	y(),
	z()
{
	float sinAngleOver2 = sin(angle/2);
	w = cos(angle/2);
	x = reinterpret_cast<const T*>(&axis)[0]*sinAngleOver2;
	y = reinterpret_cast<const T*>(&axis)[1]*sinAngleOver2;
	z = reinterpret_cast<const T*>(&axis)[2]*sinAngleOver2;
}

//------------------------------------------------------------------------------
//
template<class T>
Quat<T>::Quat(const Matrix<T, 1, 4> &rhs)
:
	w(reinterpret_cast<const T*>(&rhs)[0]),
	x(reinterpret_cast<const T*>(&rhs)[1]),
	y(reinterpret_cast<const T*>(&rhs)[2]),
	z(reinterpret_cast<const T*>(&rhs)[3])
{}

//------------------------------------------------------------------------------
//
template<class T>
Quat<T>& Quat<T>::operator = (const Matrix<T, 1, 4> &rhs)
{
	w = reinterpret_cast<const T*>(&rhs)[0];
	x = reinterpret_cast<const T*>(&rhs)[1];
	y = reinterpret_cast<const T*>(&rhs)[2];
	z = reinterpret_cast<const T*>(&rhs)[3];
	return *this;
}

//------------------------------------------------------------------------------
//
template<class T>
Quat<T>::Quat(const Quat<T> &rhs)
:
	w(rhs.w),
	x(rhs.x),
	y(rhs.y),
	z(rhs.z)
{}

//------------------------------------------------------------------------------
//
template<class T>
Quat<T>& Quat<T>::operator = (const Quat<T> &rhs)
{
	w = rhs.w;
	x = rhs.x;
	y = rhs.y;
	z = rhs.z;
	return *this;
}

//------------------------------------------------------------------------------
//
template<class T>
bool Quat<T>::operator == (const Quat<T> &rhs) const
{
	return (rhs.w == w) && (rhs.x == x) && (rhs.y == y) && (rhs.z == z);
}

//------------------------------------------------------------------------------
//
template<class T>
bool Quat<T>::operator != (const Quat<T> &rhs) const
{
	return (rhs.w != w) || (rhs.x != x) || (rhs.y != y) || (rhs.z != z);
}

//------------------------------------------------------------------------------
//
template<class T>
bool Quat<T>::operator < (const Quat<T> &rhs) const
{
	return (w < rhs.w) || ((w == rhs.w) && ((x < rhs.x) || ((x == rhs.x) && ((y < rhs.y) || ((y == rhs.y) && (z < rhs.z))))));
}

//------------------------------------------------------------------------------
//
template<class T>
bool Quat<T>::operator <= (const Quat<T> &rhs) const
{
	return (w < rhs.w) || ((w == rhs.w) && ((x < rhs.x) || ((x == rhs.x) && ((y < rhs.y) || ((y == rhs.y) && (z <= rhs.z))))));
}

//------------------------------------------------------------------------------
//
template<class T>
bool Quat<T>::operator > (const Quat<T> &rhs) const
{
	return (w > rhs.w) || ((w == rhs.w) && ((x > rhs.x) || ((x == rhs.x) && ((y > rhs.y) || ((y == rhs.y) && (z > rhs.z))))));
}

//------------------------------------------------------------------------------
//
template<class T>
bool Quat<T>::operator >= (const Quat<T> &rhs) const
{
	return (w > rhs.w) || ((w == rhs.w) && ((x > rhs.x) || ((x == rhs.x) && ((y > rhs.y) || ((y == rhs.y) && (z >= rhs.z))))));
}

//------------------------------------------------------------------------------
//
template<class T>
const T& Quat<T>::operator [] (int index) const
{
	return reinterpret_cast<const T*>(this)[index];
}

//------------------------------------------------------------------------------
//
template<class T>
Quat<T>::operator T*()
{
	return reinterpret_cast<T*>(this);
}

//------------------------------------------------------------------------------
//
template<class T>
Quat<T>::operator const T*() const
{
	return reinterpret_cast<const T*>(this);
}

//------------------------------------------------------------------------------
//
template<class T>
Quat<T> Quat<T>::operator + (const Quat<T> &rhs) const
{
	return Quat(x+rhs.x, y+rhs.y, z+rhs.z, w+rhs.w);
}

//------------------------------------------------------------------------------
//
template<class T>
Quat<T> Quat<T>::operator - (const Quat<T> &rhs) const
{
	return Quat(x-rhs.x, y-rhs.y, z-rhs.z, w-rhs.w);
}

//------------------------------------------------------------------------------
//
template<class T>
Quat<T> Quat<T>::operator * (const Quat<T> &rhs) const
{
	return Quat(
		w*rhs.w - x*rhs.x - y*rhs.y - z*rhs.z,
		w*rhs.x + x*rhs.w + y*rhs.z - z*rhs.y,
		w*rhs.y + y*rhs.w + z*rhs.x - x*rhs.z,
		w*rhs.z + z*rhs.w + x*rhs.y - y*rhs.x
	);
}

//------------------------------------------------------------------------------
//
template<class T>
Quat<T> Quat<T>::operator * (const T &rhs) const
{
	return Quat(w*rhs, x*rhs, y*rhs, z*rhs);
}

//------------------------------------------------------------------------------
//
template<class T>
Quat<T> Quat<T>::operator / (const T &rhs) const
{
	return Quat(w/rhs, x/rhs, y/rhs, z/rhs);
}

//------------------------------------------------------------------------------
//
template<class T>
Quat<T>& Quat<T>::operator += (const Quat<T> &rhs)
{
	w+= rhs.w; x += rhs.x; y += rhs.y; z += rhs.z;
	return *this;
}

//------------------------------------------------------------------------------
//
template<class T>
Quat<T>& Quat<T>::operator -= (const Quat<T> &rhs)
{
	w -= rhs.w; x -= rhs.x; y -= rhs.y; z -= rhs.z;
	return *this;
}

//------------------------------------------------------------------------------
//
template<class T>
Quat<T>& Quat<T>::operator *= (const T &rhs)
{
	*this = *this * rhs;

	return *this;
}

//------------------------------------------------------------------------------
//
template<class T>
Quat<T>& Quat<T>::operator /= (const T &rhs)
{
	w /= rhs; x /= rhs; y /= rhs; z /= rhs;
	return *this;
}

//------------------------------------------------------------------------------
//
template<class T>
Quat<T>& Quat<T>::operator - ()
{
	x = -x, y = -y; z = -z; w = -w;
	return *this;
}

//------------------------------------------------------------------------------
//
template<class T>
float Quat<T>::length()
{
	return sqrt(length2());
}

//------------------------------------------------------------------------------
//
template<class T>
float Quat<T>::length2()
{
	return w*w+x*x+y*y+z*z;
}

//------------------------------------------------------------------------------
//
template<class T>
Quat<T>& Quat<T>::scale(float newLength)
{
	float s = newLength/length();
	w *= s;	x *= s, y *= s; z *= s;
	return *this;
}

//------------------------------------------------------------------------------
//
template<class T>
Quat<T>& Quat<T>::normalize()
{
	return scale(1.0f);
}

//------------------------------------------------------------------------------
//
template<class T>
Quat<T> Quat<T>::conjugate()
{
	return Quat(w, -x, -y, -z);
}

//------------------------------------------------------------------------------
//
template<class T>
Quat<T> Quat<T>::unitInverse()	//	assumes we have a unit quaternion
{
	return conjugate();
}

//------------------------------------------------------------------------------
//
template<class T>
Quat<T> Quat<T>::inverse()
{
	return conjugate()/length2();
}

//------------------------------------------------------------------------------
//
template<class T>
void Quat<T>::to4x4Matrix(Matrix<T, 4, 4> *outMatrix)
{
	// avoid depending on Matrix.h
	T* m = reinterpret_cast<T*>(outMatrix);
	
	float xx = x*x;	float xy = x*y;
	float xz = x*z;	float xw = x*w;

	float yy = y*y;	float yz = y*z;
	float yw = y*w;

	float zz = z*z;	float zw = z*w;

	m[0*4+0] = 1-2*(yy+zz); m[1*4+0] =   2*(xy-zw); m[2*4+0] =   2*(xz+yw); m[3*4+0] = 0;
	m[0*4+1] =   2*(xy+zw); m[1*4+1] = 1-2*(xx+zz); m[2*4+1] =   2*(yz-xw); m[3*4+1] = 0;
	m[0*4+2] =   2*(xz-yw); m[1*4+2] =   2*(yz+xw); m[2*4+2] = 1-2*(xx+yy); m[3*4+2] = 0;
	m[0*4+3] =           0; m[1*4+3] =           0; m[2*4+3] =           0; m[3*4+3] = 1;
}

//------------------------------------------------------------------------------
//
template<class T>
void Quat<T>::to3x3Matrix(Matrix<T, 3, 3> *outMatrix)
{
	// avoid depending on Matrix.h
	T* m = reinterpret_cast<T*>(outMatrix);

	float xx = x*x;	float xy = x*y;
	float xz = x*z;	float xw = x*w;

	float yy = y*y;	float yz = y*z;
	float yw = y*w;

	float zz = z*z;	float zw = z*w;

	m[0*3+0] = 1-2*(yy+zz); m[1*3+0] =   2*(xy-zw); m[2*3+0] =   2*(xz+yw);
	m[0*3+1] =   2*(xy+zw); m[1*3+1] = 1-2*(xx+zz); m[2*3+1] =   2*(yz-xw);
	m[0*3+2] =   2*(xz-yw); m[1*3+2] =   2*(yz+xw); m[2*3+2] = 1-2*(xx+yy);
}

//------------------------------------------------------------------------------
//
template<class T>
static inline Quat<T> operator * (const T &lhs, const Quat<T> &rhs)
{
	return rhs * lhs;
}

#endif // __MK_GEOMETRY_QUAT_INLINE__

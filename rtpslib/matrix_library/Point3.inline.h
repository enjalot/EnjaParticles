/*******************************************************************************
 File:				Point3.inline.h

 Author: 			Gaspard Petit (gaspardpetit@gmail.com)
 Last Revision:		March 14, 2007
 
 This code may be reused without my permission as long as credits are given to
 the original author.  If you find bugs, please send me a note...
*******************************************************************************/
#ifndef __MK_GEOMETRY_POINT3_INLINE__
#define __MK_GEOMETRY_POINT3_INLINE__


//==============================================================================
//	EXTERNAL DECLARATIONS
//==============================================================================
#include "Point3.h"


//------------------------------------------------------------------------------
//
template<class T>
Point<T, 3>::Point()
:
	x(),
	y(),
	z()
{}

//------------------------------------------------------------------------------
//
template<class T>
template<class U>
Point<T, 3>::Point(const U *array)
:
	x(array[0]),
	y(array[1]),
	z(array[2])
{}

//------------------------------------------------------------------------------
//
template<class T>
Point<T, 3>::Point(T inX, T inY, T inZ)
:
	x(inX),
	y(inY),
	z(inZ)
{}

//------------------------------------------------------------------------------
//
template<class T>
Point<T, 3>::Point(const Point<T, 1> &inX, const Point<T, 1> &inY, const Point<T, 1> &inZ)
:
	x(inX.x),
	y(inY.x),
	z(inZ.x)
{}

//------------------------------------------------------------------------------
//
template<class T>
Point<T, 3>::Point(const T &inX, const Point<T, 1> &inY, const Point<T, 1> &inZ)
:
	x(inX),
	y(inY.x),
	z(inZ.x)
{}

//------------------------------------------------------------------------------
//
template<class T>
Point<T, 3>::Point(const Point<T, 1> &inX, const T &inY, const Point<T, 1> &inZ)
:
	x(inX.x),
	y(inY),
	z(inZ.x)
{}

//------------------------------------------------------------------------------
//
template<class T>
Point<T, 3>::Point(const Point<T, 1> &inX, const Point<T, 1> &inY, const T &inZ)
:
	x(inX.x),
	y(inY.x),
	z(inZ)
{}

//------------------------------------------------------------------------------
//
template<class T>
Point<T, 3>::Point(const T &inX, const T &inY, const Point<T, 1> &inZ)
:
	x(inX),
	y(inY),
	z(inZ.x)
{}

//------------------------------------------------------------------------------
//
template<class T>
Point<T, 3>::Point(const Point<T, 1> &inX, const T &inY, const T &inZ)
:
	x(inX.x),
	y(inY),
	z(inZ)
{}

//------------------------------------------------------------------------------
//
template<class T>
Point<T, 3>::Point(const Point<T,2> &inXY, const T &inZ)
:
	x(inXY.x),
	y(inXY.y),
	z(inZ)
{}

//------------------------------------------------------------------------------
//
template<class T>
Point<T, 3>::Point(const Point<T,2> &inXY, const Point<T, 1> &inZ)
:
	x(inXY.x),
	y(inXY.y),
	z(inZ.x)
{}

//------------------------------------------------------------------------------
//
template<class T>
Point<T, 3>::Point(const T &inX, const Point<T, 2> &inYZ)
:
	x(inX),
	y(inYZ.x),
	z(inYZ.y)
{}

//------------------------------------------------------------------------------
//
template<class T>
Point<T, 3>::Point(const Point<T, 1> &inX, const Point<T, 2> &inYZ)
:
	x(inX.x),
	y(inYZ.x),
	z(inYZ.y)
{}

//------------------------------------------------------------------------------
//
template<class T>
Point<T, 3>::Point(T inVal)
:
	x(inVal),
	y(inVal),
	z(inVal)
{}

//------------------------------------------------------------------------------
//
template<class T>
Point<T, 3>::Point(const Matrix<T, 1, 3> &rhs)
:
	x(reinterpret_cast<const T*>(&rhs)[0]),
	y(reinterpret_cast<const T*>(&rhs)[1]),
	z(reinterpret_cast<const T*>(&rhs)[2])
{}

//------------------------------------------------------------------------------
//
template<class T>
template<class U>
Point<T, 3>::Point(const Point<U,3> &rhs)
:
	x(rhs.x),
	y(rhs.y),
	z(rhs.z)
{}

//------------------------------------------------------------------------------
//
template<class T>
Point<T, 3>& Point<T, 3>::operator = (const Matrix<T, 1, 3> &rhs)
{
	x = reinterpret_cast<const T*>(&rhs)[0];
	y = reinterpret_cast<const T*>(&rhs)[1];
	z = reinterpret_cast<const T*>(&rhs)[2];
	return *this;
}

//------------------------------------------------------------------------------
//
template<class T>
Point<T, 3>& Point<T, 3>::operator = (const Point &rhs)
{
	x = rhs.x;
	y = rhs.y;
	z = rhs.z;
	return *this;
}

//------------------------------------------------------------------------------
//
template<class T>
bool Point<T, 3>::operator == (const Point &rhs) const
{
	return (rhs.x == x) && (rhs.y == y) && (rhs.z == z);
}

//------------------------------------------------------------------------------
//
template<class T>
bool Point<T, 3>::operator != (const Point &rhs) const
{
	return (rhs.x != x) || (rhs.y != y) || (rhs.z != z);
}

//------------------------------------------------------------------------------
//
template<class T>
bool Point<T, 3>::operator < (const Point &rhs) const
{
	return (x < rhs.x) || ((x == rhs.x) && ((y < rhs.y) || ((y == rhs.y) && (z < rhs.z))));
}

//------------------------------------------------------------------------------
//
template<class T>
bool Point<T, 3>::operator <= (const Point &rhs) const
{
	return (x < rhs.x) || ((x == rhs.x) && ((y < rhs.y) || ((y == rhs.y) && (z <= rhs.z))));
}

//------------------------------------------------------------------------------
//
template<class T>
bool Point<T, 3>::operator > (const Point &rhs) const
{
	return (x > rhs.x) || ((x == rhs.x) && ((y > rhs.y) || ((y == rhs.y) && (z > rhs.z))));
}

//------------------------------------------------------------------------------
//
template<class T>
bool Point<T, 3>::operator >= (const Point &rhs) const
{
	return (x > rhs.x) || ((x == rhs.x) && ((y > rhs.y) || ((y == rhs.y) && (z >= rhs.z))));
}

//------------------------------------------------------------------------------
//
template<class T>
const T& Point<T, 3>::operator [] (int index) const
{
	return reinterpret_cast<const T*>(this)[index];
}

//------------------------------------------------------------------------------
//
template<class T>
T& Point<T, 3>::operator [] (int index)
{
	return reinterpret_cast<T*>(this)[index];
}


//------------------------------------------------------------------------------
//
template<class T>
T* Point<T, 3>::ptr()
{
	return reinterpret_cast<T*>(this);
}

//------------------------------------------------------------------------------
//
template<class T>
const T* Point<T, 3>::ptr() const
{
	return reinterpret_cast<const T*>(this);
}

//------------------------------------------------------------------------------
//
template<class T>
Point<T, 3> Point<T, 3>::operator + (const Point<T, 3> &rhs) const
{
	return Point<T, 3>(x+rhs.x, y+rhs.y, z+rhs.z);
}

//------------------------------------------------------------------------------
//
template<class T>
Point<T, 3> Point<T, 3>::operator - (const Point<T, 3> &rhs) const
{
	return Point<T, 3>(x-rhs.x, y-rhs.y, z-rhs.z);
}

//------------------------------------------------------------------------------
//
template<class T>
T Point<T, 3>::operator * (const Point<T, 3> &rhs) const
{
	return x*rhs.x + y*rhs.y + z*rhs.z;
}

//------------------------------------------------------------------------------
//
template<class T>
Point<T, 3> Point<T, 3>::operator ^ (const Point<T, 3> &rhs) const
{
	return Point<T, 3>(y*rhs.z-z*rhs.y, z*rhs.x-x*rhs.z, x*rhs.y-y*rhs.x);
}

//------------------------------------------------------------------------------
//
template<class T>
Point<T, 3> Point<T, 3>::operator * (const T &rhs) const
{
	return Point<T, 3>(x*rhs, y*rhs, z*rhs);
}

//------------------------------------------------------------------------------
//
template<class T>
Point<T, 3> Point<T, 3>::operator / (const T &rhs) const
{
	return Point<T, 3>(x/rhs, y/rhs, z/rhs);
}

//------------------------------------------------------------------------------
//
template<class T>
Point<T, 3>& Point<T, 3>::operator += (const Point<T, 3> &rhs)
{
	x += rhs.x; y += rhs.y; z += rhs.z;
	return *this;
}

//------------------------------------------------------------------------------
//
template<class T>
Point<T, 3>& Point<T, 3>::operator -= (const Point<T, 3> &rhs)
{
	x -= rhs.x; y -= rhs.y; z -= rhs.z;
	return *this;
}

//------------------------------------------------------------------------------
//
template<class T>
Point<T, 3>& Point<T, 3>::operator *= (const T &rhs)
{
	x *= rhs; y *= rhs; z *= rhs;
	return *this;
}

//------------------------------------------------------------------------------
//
template<class T>
Point<T, 3>& Point<T, 3>::operator /= (const T &rhs)
{
	x /= rhs; y /= rhs; z /= rhs;
	return *this;
}

//------------------------------------------------------------------------------
//
template<class T>
Point<T, 3>& Point<T, 3>::operator - ()
{
	x = -x, y = -y; z = -z;
	return *this;
}

//------------------------------------------------------------------------------
//
template<class T>
float Point<T, 3>::length() const
{
	return sqrt(length2());
}

//------------------------------------------------------------------------------
//
template<class T>
float Point<T, 3>::length2() const
{
	return x*x+y*y+z*z;
}

//------------------------------------------------------------------------------
//
template<class T>
void Point<T, 3>::scale(float newLength)
{
	*this = scaled(newLength);
}

//------------------------------------------------------------------------------
//
template<class T>
Point<T, 3> Point<T, 3>::scaled(float newLength) const
{
	float scale = newLength/length();
	return *this * scale;
}

//------------------------------------------------------------------------------
//
template<class T>
void Point<T, 3>::normalize()
{
	scale(1.0);
}

//------------------------------------------------------------------------------
//
template<class T>
Point<T, 3> Point<T, 3>::normalized() const
{
	return scaled(1.0f);
}

//------------------------------------------------------------------------------
//
template<class T>
Matrix<T, 1, 3>& Point<T, 3>::asTMatrix()
{
	return *(Matrix<T, 1, 3>*)this;
}

//------------------------------------------------------------------------------
//
template<class T>
const Matrix<T, 1, 3>& Point<T, 3>::asTMatrix() const
{
	return *(Matrix<T, 1, 3>*)this;
}

//------------------------------------------------------------------------------
//
template<class T>
Matrix<T, 3, 1>& Point<T, 3>::asMatrix()
{
	return *(Matrix<T, 3, 1>*)this;
}

//------------------------------------------------------------------------------
//
template<class T>
const Matrix<T, 3, 1>& Point<T, 3>::asMatrix() const
{
	return *(Matrix<T, 3, 1>*)this;
}

//------------------------------------------------------------------------------
//
template<class T>
bool Point<T, 3>::hasNan() const
{
	return std::isnan(x) || std::isnan(y) || std::isnan(z);
}

//------------------------------------------------------------------------------
//
template<class T>
bool Point<T, 3>::hasInf() const
{
	return std::isinf(x) || std::isinf(y) || std::isinf(z);
}


//------------------------------------------------------------------------------
//
template<class T>
static inline Point<T,3> operator * (const T &lhs, const Point<T,3> &rhs)
{
	return rhs * lhs;
}


//------------------------------------------------------------------------------
//
template<class T>
static inline Point<T,3> operator * (const Matrix<T, 3, 3> &lhs, const Point<T,3> &rhs)
{
	return lhs * rhs.asTMatrix();
}

#endif // __MK_GEOMETRY_POINT3_INLINE__

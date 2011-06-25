/*******************************************************************************
 File:				Point2.inline.h

 Author: 			Gaspard Petit (gaspardpetit@gmail.com)
 Last Revision:		March 14, 2007
 
 This code may be reused without my permission as long as credits are given to
 the original author.  If you find bugs, please send me a note...
*******************************************************************************/
#ifndef __MK_GEOMETRY_POINT2_INLINE__
#define __MK_GEOMETRY_POINT2_INLINE__

//==============================================================================
//	EXTERNAL DECLARATIONS
//==============================================================================
#include "Point2.h"
#include <cmath>


//------------------------------------------------------------------------------
//
template<class T>
Point<T, 2>::Point()
:
	x(),
	y()
{}

//------------------------------------------------------------------------------
//
template<class T>
Point<T, 2>::Point(T inX, T inY)
:
	x(inX),
	y(inY)
{}

//------------------------------------------------------------------------------
//
template<class T>
Point<T, 2>::Point(const Point<T, 1> &inX, const Point<T, 1> &inY)
:
	x(inX.x),
	y(inY.x)
{}

//------------------------------------------------------------------------------
//
template<class T>
Point<T, 2>::Point(const T &inX, const Point<T, 1> &inY)
:
	x(inX),
	y(inY.x)
{}

//------------------------------------------------------------------------------
//
template<class T>
Point<T, 2>::Point(const Point<T, 1> &inX, const T &inY)
:
	x(inX.x),
	y(inY)
{}

//------------------------------------------------------------------------------
//
template<class T>
Point<T, 2>::Point(T inVal)
:
	x(inVal),
	y(inVal)
{}

//------------------------------------------------------------------------------
//
template<class T>
Point<T, 2>::Point(const Matrix<T, 1, 2> &rhs)
:
	x(reinterpret_cast<const T*>(&rhs)[0]),
	y(reinterpret_cast<const T*>(&rhs)[1])
{}

//------------------------------------------------------------------------------
//
template<class T>
Point<T, 2>::Point(const Point &rhs)
:
	x(rhs.x),
	y(rhs.y)
{}

//------------------------------------------------------------------------------
//
template<class T>
Point<T, 2>& Point<T, 2>::operator = (const Point<T, 2> &rhs)
{
	x = rhs.x;
	y = rhs.y;
	return *this;
}

//------------------------------------------------------------------------------
//
template<class T>
Point<T, 2>& Point<T, 2>::operator = (const Matrix<T, 1, 2> &rhs)
{
	x = reinterpret_cast<const T*>(&rhs)[0];
	y = reinterpret_cast<const T*>(&rhs)[1];
	return *this;
}

//------------------------------------------------------------------------------
//
template<class T>
bool Point<T, 2>::operator == (const Point<T, 2> &rhs) const
{
	return (rhs.x == x) && (rhs.y == y);
}

//------------------------------------------------------------------------------
//
template<class T>
bool Point<T, 2>::operator != (const Point<T, 2> &rhs) const
{
	return (rhs.x != x) || (rhs.y != y);
}

//------------------------------------------------------------------------------
//
template<class T>
bool Point<T, 2>::operator < (const Point<T, 2> &rhs) const
{
	return (x < rhs.x) || ((x == rhs.x) && (y < rhs.y));
}

//------------------------------------------------------------------------------
//
template<class T>
bool Point<T, 2>::operator <= (const Point<T, 2> &rhs) const
{
	return (x < rhs.x) || ((x == rhs.x) && (y <= rhs.y));
}

//------------------------------------------------------------------------------
//
template<class T>
bool Point<T, 2>::operator > (const Point<T, 2> &rhs) const
{
	return (x > rhs.x) || ((x == rhs.x) && (y > rhs.y));
}

//------------------------------------------------------------------------------
//
template<class T>
bool Point<T, 2>::operator >= (const Point<T, 2> &rhs) const
{
	return (x > rhs.x) || ((x == rhs.x) && (y >= rhs.y));
}

//------------------------------------------------------------------------------
//
template<class T>
const T& Point<T, 2>::operator [] (int index) const
{
	return reinterpret_cast<const T*>(this)[index];
}

//------------------------------------------------------------------------------
//
template<class T>
T& Point<T, 2>::operator [] (int index)
{
	return reinterpret_cast<T*>(this)[index];
}

//------------------------------------------------------------------------------
//
template<class T>
T* Point<T, 2>::ptr()
{
	return reinterpret_cast<T*>(this);
}

//------------------------------------------------------------------------------
//
template<class T>
const T* Point<T, 2>::ptr() const
{
	return reinterpret_cast<const T*>(this);
}

//------------------------------------------------------------------------------
//
template<class T>
Point<T, 2> Point<T, 2>::operator + (const Point<T, 2> &rhs) const
{
	return Point<T, 2>(x+rhs.x, y+rhs.y);
}

//------------------------------------------------------------------------------
//
template<class T>
Point<T, 2> Point<T, 2>::operator - (const Point<T, 2> &rhs) const
{
	return Point<T, 2>(x-rhs.x, y-rhs.y);
}

//------------------------------------------------------------------------------
//
template<class T>
T Point<T, 2>::operator * (const Point<T, 2> &rhs) const
{
	return x*rhs.x + y*rhs.y;
}

//------------------------------------------------------------------------------
//
template<class T>
T Point<T, 2>::operator ^ (const Point<T, 2> &rhs) const
{
	return x*rhs.y - y*rhs.x;
}

//------------------------------------------------------------------------------
//
template<class T>
Point<T, 2> Point<T, 2>::operator * (const T &rhs) const
{
	return Point<T, 2>(x*rhs, y*rhs);
}

//------------------------------------------------------------------------------
//
template<class T>
Point<T, 2> Point<T, 2>::operator / (const T &rhs) const
{
	return Point<T, 2>(x/rhs, y/rhs);
}

//------------------------------------------------------------------------------
//
template<class T>
Point<T, 2>& Point<T, 2>::operator += (const Point<T, 2> &rhs)
{
	x += rhs.x; y += rhs.y;
	return *this;
}

//------------------------------------------------------------------------------
//
template<class T>
Point<T, 2>& Point<T, 2>::operator -= (const Point<T, 2> &rhs)
{
	x -= rhs.x; y -= rhs.y;
	return *this;
}

//------------------------------------------------------------------------------
//
template<class T>
Point<T, 2>& Point<T, 2>::operator *= (const T &rhs)
{
	x *= rhs; y *= rhs;
	return *this;
}

//------------------------------------------------------------------------------
//
template<class T>
Point<T, 2>& Point<T, 2>::operator /= (const T &rhs)
{
	x /= rhs; y /= rhs;
	return *this;
}

//------------------------------------------------------------------------------
//
template<class T>
Point<T, 2>& Point<T, 2>::operator - ()
{
	x = -x, y = -y;
	return *this;
}

//------------------------------------------------------------------------------
//
template<class T>
Point<T, 2>& Point<T, 2>::operator ~ ()
{
	*this = perpendicular();
	return *this;
}

//------------------------------------------------------------------------------
//
template<class T>
float Point<T, 2>::length() const
{
	return sqrt(length2());
}

//------------------------------------------------------------------------------
//
template<class T>
float Point<T, 2>::length2() const
{
	return x*x+y*y;
}

//------------------------------------------------------------------------------
//
template<class T>
void Point<T, 2>::scale(float newLength)
{
	*this = scaled(newLength);
}

//------------------------------------------------------------------------------
//
template<class T>
Point<T, 2> Point<T, 2>::scaled(float newLength) const
{
	T scale = newLength/length();
	return *this * scale;
}


//------------------------------------------------------------------------------
//
template<class T>
void Point<T, 2>::normalize()
{
	scale(1.0f);
}

//------------------------------------------------------------------------------
//
template<class T>
Point<T, 2> Point<T, 2>::normalized() const
{
	return scaled(1.0f);
}

//------------------------------------------------------------------------------
//
template<class T>
Point<T, 2> Point<T, 2>::perpendicular() const
{
	return Point<T, 2>(-y, x);
}

//------------------------------------------------------------------------------
//
template<class T>
Matrix<T, 1, 2>& Point<T, 2>::asTMatrix()
{
	return *(Matrix<T, 1, 2>*)this;
}

//------------------------------------------------------------------------------
//
template<class T>
const Matrix<T, 1, 2>& Point<T, 2>::asTMatrix() const
{
	return *(Matrix<T, 1, 2>*)this;
}

//------------------------------------------------------------------------------
//
template<class T>
Matrix<T, 2, 1>& Point<T, 2>::asMatrix()
{
	return *(Matrix<T, 2, 1>*)this;
}

//------------------------------------------------------------------------------
//
template<class T>
const Matrix<T, 2, 1>& Point<T, 2>::asMatrix() const
{
	return *(Matrix<T, 2, 1>*)this;
}

//------------------------------------------------------------------------------
//
template<class T>
bool Point<T, 2>::hasNan() const
{
	return std::isnan(x) || std::isnan(y);
}

//------------------------------------------------------------------------------
//
template<class T>
bool Point<T, 2>::hasInf() const
{
	return std::isinf(x) || std::isinf(y);
}

//------------------------------------------------------------------------------
//
template<class U, class T>
static inline Point<T, 2> operator * (const U &lhs, const Point<T, 2> &rhs)
{
	return rhs * lhs;
}

#endif // __MK_GEOMETRY_POINT2_INLINE__

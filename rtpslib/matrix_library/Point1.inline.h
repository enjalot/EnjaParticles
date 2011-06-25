/*******************************************************************************
 File:				Point1.inline.h

 Author: 			Gaspard Petit (gaspardpetit@gmail.com)
 Last Revision:		March 14, 2007
 
 This code may be reused without my permission as long as credits are given to
 the original author.  If you find bugs, please send me a note...
*******************************************************************************/
#ifndef __MK_GEOMETRY_POINT1_INLINE__
#define __MK_GEOMETRY_POINT1_INLINE__

//==============================================================================
//	EXTERNAL DECLARATIONS
//==============================================================================
#include "Point1.h"
#include <cmath>

//==============================================================================
//	CLASS Point1
//==============================================================================

//------------------------------------------------------------------------------
//
template<class T>
Point<T, 1>::Point()
:
	x()
{}

//------------------------------------------------------------------------------
//
template<class T>
Point<T, 1>::Point(T inX)
:
	x(inX)
{}

//------------------------------------------------------------------------------
//
template<class T>
Point<T, 1>::Point(const Matrix<T, 1, 1> &rhs)
:
	x(*reinterpret_cast<const T*>(&rhs))
{}

//------------------------------------------------------------------------------
//
template<class T>
Point<T, 1>::Point(const Point<T, 1> &rhs)
:
	x(rhs.x)
{}

//------------------------------------------------------------------------------
//
template<class T>
Point<T, 1>& Point<T, 1>::operator = (const Point<T, 1> &rhs)
{
	x = rhs.x;
	return *this;
}

//------------------------------------------------------------------------------
//
template<class T>
Point<T, 1>& Point<T, 1>::operator = (const Matrix<T, 1, 1> &rhs)
{
	x = *reinterpret_cast<const T*>(&rhs);
	return *this;
}

//------------------------------------------------------------------------------
//
template<class T>
bool Point<T, 1>::operator == (const Point<T, 1> &rhs) const
{
	return (rhs.x == x);
}

//------------------------------------------------------------------------------
//
template<class T>
bool Point<T, 1>::operator != (const Point<T, 1> &rhs) const
{
	return (rhs.x != x);
}

//------------------------------------------------------------------------------
//
template<class T>
bool Point<T, 1>::operator < (const Point<T, 1> &rhs) const
{
	return (x < rhs.x);
}

//------------------------------------------------------------------------------
//
template<class T>
bool Point<T, 1>::operator <= (const Point<T, 1> &rhs) const
{
	return (x < rhs.x);
}

//------------------------------------------------------------------------------
//
template<class T>
bool Point<T, 1>::operator > (const Point<T, 1> &rhs) const
{
	return (x > rhs.x);
}

//------------------------------------------------------------------------------
//
template<class T>
bool Point<T, 1>::operator >= (const Point<T, 1> &rhs) const
{
	return (x > rhs.x);
}

//------------------------------------------------------------------------------
//
template<class T>
const T& Point<T, 1>::operator [] (int index) const
{
	return reinterpret_cast<const T*>(this)[index];
}

//------------------------------------------------------------------------------
//
template<class T>
T& Point<T, 1>::operator [] (int index)
{
	return reinterpret_cast<T*>(this)[index];
}

//------------------------------------------------------------------------------
//
template<class T>
T* Point<T, 1>::ptr()
{
	return reinterpret_cast<T*>(this);
}

//------------------------------------------------------------------------------
//
template<class T>
const T* Point<T, 1>::ptr() const
{
	return reinterpret_cast<const T*>(this);
}

//------------------------------------------------------------------------------
//
template<class T>
Point<T, 1> Point<T, 1>::operator + (const Point<T, 1> &rhs) const
{
	return Point<T, 1>(x+rhs.x);
}

//------------------------------------------------------------------------------
//
template<class T>
Point<T, 1> Point<T, 1>::operator - (const Point<T, 1> &rhs) const
{
	return Point<T, 1>(x-rhs.x);
}

//------------------------------------------------------------------------------
//
template<class T>
T Point<T, 1>::operator * (const Point<T, 1> &rhs) const
{
	return x*rhs.x;
}

//------------------------------------------------------------------------------
//
template<class T>
Point<T, 1> Point<T, 1>::operator * (const T &rhs) const
{
	return Point<T, 1>(x*rhs);
}

//------------------------------------------------------------------------------
//
template<class T>
Point<T, 1> Point<T, 1>::operator / (const T &rhs) const
{
	return Point<T, 1>(x/rhs);
}

//------------------------------------------------------------------------------
//
template<class T>
Point<T, 1>& Point<T, 1>::operator += (const Point<T, 1> &rhs)
{
	x += rhs.x;
	return *this;
}

//------------------------------------------------------------------------------
//
template<class T>
Point<T, 1>& Point<T, 1>::operator -= (const Point<T, 1> &rhs)
{
	x -= rhs.x;
	return *this;
}

//------------------------------------------------------------------------------
//
template<class T>
Point<T, 1>& Point<T, 1>::operator *= (const T &rhs)
{
	x *= rhs;
	return *this;
}

//------------------------------------------------------------------------------
//
template<class T>
Point<T, 1>& Point<T, 1>::operator /= (const T &rhs)
{
	x /= rhs;
	return *this;
}

//------------------------------------------------------------------------------
//
template<class T>
Point<T, 1>& Point<T, 1>::operator - ()
{
	x = -x;
	return *this;
}

//------------------------------------------------------------------------------
//
template<class T>
T Point<T, 1>::length() const
{
	return x;
}

//------------------------------------------------------------------------------
//
template<class T>
T Point<T, 1>::length2() const
{
	return x*x;
}

//------------------------------------------------------------------------------
//
template<class T>
Matrix<T, 1, 1>& Point<T, 1>::asMatrix()
{
	return *(Matrix<T, 1, 1>*)this;
}

//------------------------------------------------------------------------------
//
template<class T>
const Matrix<T, 1, 1>& Point<T, 1>::asMatrix() const
{
	return *(Matrix<T, 1, 1>*)this;
}

//------------------------------------------------------------------------------
//
template<class T>
bool Point<T, 1>::hasNan() const
{
	return std::isnan(x);
}

//------------------------------------------------------------------------------
//
template<class T>
bool Point<T, 1>::hasInf() const
{
	return std::isinf(x);
}

//------------------------------------------------------------------------------
//
template<class U, class T>
static inline Point<T, 1> operator * (const U &lhs, const Point<T, 1> &rhs)
{
	return rhs * lhs;
}

#endif // __MK_GEOMETRY_POINT1_INLINE__

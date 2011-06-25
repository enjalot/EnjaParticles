/*******************************************************************************
 File:				Point.inline.h

 Author: 			Gaspard Petit (gaspardpetit@gmail.com)
 Last Revision:		March 14, 2007
 
 This code may be reused without my permission as long as credits are given to
 the original author.  If you find bugs, please send me a note...
*******************************************************************************/
#ifndef __MK_GEOMETRY_POINT_INLINE__
#define __MK_GEOMETRY_POINT_INLINE__

//==============================================================================
//	EXTERNAL DECLARATIONS
//==============================================================================
#include "Point.h"
#include <string>

//==============================================================================
//	CLASS Point
//==============================================================================

//------------------------------------------------------------------------------
//
template<class T, int SIZE>
int Point<T, SIZE>::size()
{
	return SIZE;
}

//------------------------------------------------------------------------------
//
template<class T, int SIZE>
Point<T, SIZE>::Point()
{}

//------------------------------------------------------------------------------
//
template<class T, int SIZE>
Point<T, SIZE>::Point(const T *ptData)
{
	memcpy(coords, ptData, sizeof(T)*SIZE);
}

//------------------------------------------------------------------------------
//
template<class T, int SIZE>
Point<T, SIZE>::Point(const Matrix<T, 1, SIZE> &rhs)
{
	memcpy(coords, &rhs, sizeof(T)*SIZE);
}

//------------------------------------------------------------------------------
//
template<class T, int SIZE>
Point<T, SIZE>::Point(const Point<T, SIZE> &rhs)
{
	memcpy(coords, rhs.coords, sizeof(T)*SIZE);
}

//------------------------------------------------------------------------------
//
template<class T, int SIZE>
Point<T, SIZE>& Point<T, SIZE>::operator = (const Matrix<T, 1, SIZE> &rhs)
{
	memcpy(coords, &rhs, sizeof(T)*SIZE);
	return *this;
}

//------------------------------------------------------------------------------
//
template<class T, int SIZE>
Point<T, SIZE>& Point<T, SIZE>::operator = (const Point<T, SIZE> &rhs)
{
	memcpy(coords, rhs.coords, sizeof(T)*SIZE);
	return *this;
}

//------------------------------------------------------------------------------
//
template<class T, int SIZE>
bool Point<T, SIZE>::operator == (const Point<T, SIZE> &rhs) const
{
	return memcmp(coords, rhs.coords, sizeof(T)*SIZE) == 0;
}

//------------------------------------------------------------------------------
//
template<class T, int SIZE>
bool Point<T, SIZE>::operator != (const Point<T, SIZE> &rhs) const
{
	return memcmp(coords, rhs.coords, sizeof(T)*SIZE) != 0;
}

//------------------------------------------------------------------------------
//
template<class T, int SIZE>
bool Point<T, SIZE>::operator < (const Point<T, SIZE> &rhs) const
{
	for (int i = 0; i < SIZE; ++i)
	{
		if ((*this)[i] < rhs[i])		return true;
		else if ((*this)[i] > rhs[i])	return false;
	}
	return false;
}

//------------------------------------------------------------------------------
//
template<class T, int SIZE>
bool Point<T, SIZE>::operator <= (const Point<T, SIZE> &rhs) const
{
	for (int i = 0; i < SIZE; ++i)
	{
		if ((*this)[i] < rhs[i])		return true;
		else if ((*this)[i] > rhs[i])	return false;
	}
	return true;
}

//------------------------------------------------------------------------------
//
template<class T, int SIZE>
bool Point<T, SIZE>::operator > (const Point<T, SIZE> &rhs) const
{
	for (int i = 0; i < SIZE; ++i)
	{
		if ((*this)[i] > rhs[i])		return true;
		else if ((*this)[i] < rhs[i])	return false;
	}
	return false;
}

//------------------------------------------------------------------------------
//
template<class T, int SIZE>
bool Point<T, SIZE>::operator >= (const Point<T, SIZE> &rhs) const
{
	for (int i = 0; i < SIZE; ++i)
	{
		if ((*this)[i] > rhs[i])		return true;
		else if ((*this)[i] < rhs[i])	return false;
	}
	return true;
}

//------------------------------------------------------------------------------
//
template<class T, int SIZE>
const T& Point<T, SIZE>::operator [] (int index) const
{
	return coords[index];
}

//------------------------------------------------------------------------------
//
template<class T, int SIZE>
T& Point<T, SIZE>::operator [] (int index)
{
	return coords[index];
}

//------------------------------------------------------------------------------
//
template<class T, int SIZE>
T* Point<T, SIZE>::ptr()
{
	return reinterpret_cast<T*>(this);
}

//------------------------------------------------------------------------------
//
template<class T, int SIZE>
const T* Point<T, SIZE>::ptr() const
{
	return reinterpret_cast<const T*>(this);
}

//------------------------------------------------------------------------------
//
template<class T, int SIZE>
Point<T, SIZE> Point<T, SIZE>::operator + (const Point<T, SIZE> &rhs) const
{
	Point<T, SIZE> pt;
	for (int i = 0; i < SIZE; ++i) pt[i] = coords[i] + rhs.coords[i];
	return pt;
}

//------------------------------------------------------------------------------
//
template<class T, int SIZE>
Point<T, SIZE> Point<T, SIZE>::operator - (const Point<T, SIZE> &rhs) const
{
	Point<T, SIZE> pt;
	for (int i = 0; i < SIZE; ++i) pt[i] = coords[i] - rhs.coords[i];
	return pt;
}

//------------------------------------------------------------------------------
//
template<class T, int SIZE>
Matrix<T, 1, SIZE>& Point<T, SIZE>::asMatrix()
{
	return *reinterpret_cast<Matrix<T, 1, SIZE>*>(this);
}

//------------------------------------------------------------------------------
//
template<class T, int SIZE>
const Matrix<T, 1, SIZE>& Point<T, SIZE>::asMatrix() const
{
	return *reinterpret_cast<const Matrix<T, 1, SIZE>*>(this);
}

//------------------------------------------------------------------------------
//
template<class T, int SIZE>
const Matrix<T, SIZE, 1>& Point<T, SIZE>::asTMatrix() const
{
	return *reinterpret_cast<const Matrix<T, SIZE, 1>*>(this);
}

//------------------------------------------------------------------------------
//
template<class T, int SIZE>
Matrix<T, SIZE, 1>& Point<T, SIZE>::asTMatrix()
{
	return *reinterpret_cast<Matrix<T, SIZE, 1>*>(this);
}

//------------------------------------------------------------------------------
//
template<class T, int SIZE>
T Point<T, SIZE>::operator * (const Point &rhs) const
{
	T v = 0;
	for (int i = 0; i < SIZE; ++i)
		v += (*this)[i]*rhs[i];

	return v;
}

//------------------------------------------------------------------------------
//
template<class T, int SIZE>
Point<T, SIZE> Point<T, SIZE>::operator * (const T &rhs) const
{
	Point<T, SIZE> pt;
	for (int i = 0; i < SIZE; ++i) pt[i] = coords[i] * rhs;
	return pt;
}

//------------------------------------------------------------------------------
//
template<class T, int SIZE>
Point<T, SIZE> Point<T, SIZE>::operator / (const T &rhs) const
{
	Point<T, SIZE> pt;
	for (int i = 0; i < SIZE; ++i) pt[i] = coords[i] / rhs;
	return pt;
}

//------------------------------------------------------------------------------
//
template<class T, int SIZE>
Point<T, SIZE>& Point<T, SIZE>::operator += (const Point &rhs)
{
	for (int i = 0; i < SIZE; ++i) coords[i] += rhs.coords[i];
	return *this;
}

//------------------------------------------------------------------------------
//
template<class T, int SIZE>
Point<T, SIZE>& Point<T, SIZE>::operator -= (const Point &rhs)
{
	for (int i = 0; i < SIZE; ++i) coords[i] -= rhs.coords[i];
	return *this;
}

//------------------------------------------------------------------------------
//
template<class T, int SIZE>
Point<T, SIZE>& Point<T, SIZE>::operator *= (const T &rhs)
{
	for (int i = 0; i < SIZE; ++i) coords[i] -= rhs;
	return *this;
}

//------------------------------------------------------------------------------
//
template<class T, int SIZE>
Point<T, SIZE>& Point<T, SIZE>::operator /= (const T &rhs)
{
	for (int i = 0; i < SIZE; ++i) coords[i] /= rhs;
	return *this;
}

//------------------------------------------------------------------------------
//
template<class T, int SIZE>
Point<T, SIZE>& Point<T, SIZE>::operator - ()
{
	for (int i = 0; i < SIZE; ++i) coords[i] = -coords[i];
	return *this;
}

//------------------------------------------------------------------------------
//
template<class T, int SIZE>
T Point<T, SIZE>::length() const
{
	return sqrt(length2());
}

//------------------------------------------------------------------------------
//
template<class T, int SIZE>
T Point<T, SIZE>::length2() const
{
	T len2 = 0;
	for (int i = 0; i < SIZE; ++i)
		len2 += (*this)[i]*(*this)[i];
	
	return len2;
}

#endif // __MK_GEOMETRY_POINT_INLINE__

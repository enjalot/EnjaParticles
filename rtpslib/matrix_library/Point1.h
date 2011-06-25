/*******************************************************************************
 File:				Point1.h

 Author: 			Gaspard Petit (gaspardpetit@gmail.com)
 Last Revision:		March 14, 2007
 
 This code may be reused without my permission as long as credits are given to
 the original author.  If you find bugs, please send me a note...
*******************************************************************************/
#ifndef __MK_GEOMETRY_POINT1__
#define __MK_GEOMETRY_POINT1__

//==============================================================================
//	EXTERNAL DECLARATIONS
//==============================================================================
#include "Point.h"

//==============================================================================
//	CLASS Point1
//==============================================================================
template<class T>
class Point<T, 1>
{
public:
	Point();
	Point(T inX);
	Point(const Matrix<T, 1, 1> &rhs);
	Point(const Point &rhs);

	Point& operator = (const Point &rhs);
	Point& operator = (const Matrix<T, 1, 1> &rhs);

	bool operator == (const Point &rhs) const;
	bool operator != (const Point &rhs) const;
	bool operator < (const Point &rhs) const;
	bool operator <= (const Point &rhs) const;
	bool operator > (const Point &rhs) const;
	bool operator >= (const Point &rhs) const;

	const T& operator [] (int index) const;
	T& operator [] (int index);
	T* ptr();
	const T* ptr() const;

	Point operator + (const Point &rhs) const;
	Point operator - (const Point &rhs) const;
	T operator * (const Point &rhs) const;
	Point operator * (const T &rhs) const;
	Point operator / (const T &rhs) const;
	Point& operator += (const Point &rhs);
	Point& operator -= (const Point &rhs);
	Point& operator *= (const T &rhs);
	Point& operator /= (const T &rhs);
	Point& operator - ();

	T length() const;
	T length2() const;
	Matrix<T, 1, 1>& asMatrix();
	const Matrix<T, 1, 1>& asMatrix() const;
	bool hasNan() const;
	bool hasInf() const;
	
public:
	T x;

public:
	IMPLEMENT_POINT_ACCESSOR2(x,x)
	IMPLEMENT_POINT_ACCESSOR3(x, x, x)
	IMPLEMENT_POINT_ACCESSOR4(x, x, x, x)

};

template<class U, class T>
static inline Point<T, 1> operator * (const U &lhs, const Point<T, 1> &rhs);

//==============================================================================
//	TYPE DECLARATIONS
//==============================================================================
typedef Point<float, 1> 	Point1f;
typedef Point<int, 1> 		Point1i;
typedef Point<double, 1>	Point1d;

//==============================================================================
//	INLINED CODE
//==============================================================================
#include "Point1.inline.h"

#endif // __MK_GEOMETRY_POINT1__

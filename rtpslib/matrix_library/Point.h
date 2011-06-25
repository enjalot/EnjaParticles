/*******************************************************************************
 File:				Point.h

 Author: 			Gaspard Petit (gaspardpetit@gmail.com)
 Last Revision:		March 14, 2007
 
 This code may be reused without my permission as long as credits are given to
 the original author.  If you find bugs, please send me a note...
*******************************************************************************/
#ifndef __MK_GEOMETRY_POINT__
#define __MK_GEOMETRY_POINT__

//==============================================================================
//	EXTERNAL DECLARATIONS
//==============================================================================
template<class T, int W, int H> class Matrix;

//==============================================================================
//	CLASS Point
//==============================================================================
template<class T, int SIZE>
class Point
{
public:
	Point();
	Point(const T *ptData);

	Point(const Matrix<T, 1, SIZE> &rhs);
	Point(const Point &rhs);
	
	Point& operator = (const Matrix<T, 1, SIZE> &rhs);
	Point& operator = (const Point &rhs);
	
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

	Matrix<T, 1, SIZE>& asMatrix();
	const Matrix<T, 1, SIZE>& asMatrix() const;
	const Matrix<T, SIZE, 1>& asTMatrix() const;
	Matrix<T, SIZE, 1>& asTMatrix();

public:
	static int size();
	T coords[SIZE];
};

#define IMPLEMENT_POINT_ACCESSOR2(a,b)\
Point<T, 2> a##b() const\
{\
	return Point<T, 2>(a, b);\
}

#define IMPLEMENT_POINT_ACCESSOR3(a, b, c)\
Point<T, 3> a##b##c() const\
{\
	return Point<T, 3>(a, b, c);\
}


#define IMPLEMENT_POINT_ACCESSOR4(a, b, c, d)\
Point<T, 4> a##b##c##d() const\
{\
	return Point<T, 4>(a, b, c, d);\
}

//==============================================================================
//	INLINED CODE
//==============================================================================
#include "Point1.h"
#include "Point2.h"
#include "Point3.h"
#include "Point4.h"
#include "Point.inline.h"

#endif // __MK_GEOMETRY_POINT__

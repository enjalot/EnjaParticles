/*******************************************************************************
 File:				Point2.h

 Author: 			Gaspard Petit (gaspardpetit@gmail.com)
 Last Revision:		March 14, 2007
 
 This code may be reused without my permission as long as credits are given to
 the original author.  If you find bugs, please send me a note...
*******************************************************************************/
#ifndef __MK_GEOMETRY_POINT2__
#define __MK_GEOMETRY_POINT2__

//==============================================================================
//	EXTERNAL DECLARATIONS
//==============================================================================
#include "Point.h"

//==============================================================================
//	CLASS Point2
//==============================================================================

//	+ : addition of points
//	- : different of points
//	* : dot product or scalar multiplication
//	/ : scalar division
//	^ : cross product (determinant)
//	~ : perpendicular

template<class T>
class Point<T, 2>
{
public:
	Point();
	Point(T inX, T inY);
	Point(const Point<T, 1> &inX, const Point<T, 1> &inY);
	Point(const T &inX, const Point<T, 1> &inY);
	Point(const Point<T, 1> &inX, const T &inY);
	explicit Point(T inVal);
	Point(const Matrix<T, 1, 2> &rhs);
	Point(const Point &rhs);
	Point& operator = (const Point &rhs);
	Point& operator = (const Matrix<T, 1, 2> &rhs);
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
	T operator ^ (const Point &rhs) const;
	Point operator * (const T &rhs) const;
	Point operator / (const T &rhs) const;
	Point& operator += (const Point &rhs);
	
	Point& operator -= (const Point &rhs);
	Point& operator *= (const T &rhs);
	Point& operator /= (const T &rhs);
	Point& operator - ();
	Point& operator ~ ();
	float length() const;
	float length2() const;

	void scale(float newLength);
	Point scaled(float newLength) const;

	Point normalized() const;
	void normalize();

	Point perpendicular() const;

	Matrix<T, 1, 2>& asTMatrix();
	const Matrix<T, 1, 2>& asTMatrix() const;
	Matrix<T, 2, 1>& asMatrix();
	const Matrix<T, 2, 1>& asMatrix() const;
	bool hasNan() const;
	bool hasInf() const;

	IMPLEMENT_POINT_ACCESSOR2(x,x); IMPLEMENT_POINT_ACCESSOR2(x,y);
	IMPLEMENT_POINT_ACCESSOR2(y,x); IMPLEMENT_POINT_ACCESSOR2(y,y);

	IMPLEMENT_POINT_ACCESSOR3(x,x,x); IMPLEMENT_POINT_ACCESSOR3(x,x,y);
	IMPLEMENT_POINT_ACCESSOR3(x,y,x); IMPLEMENT_POINT_ACCESSOR3(x,y,y);
	IMPLEMENT_POINT_ACCESSOR3(y,x,x); IMPLEMENT_POINT_ACCESSOR3(y,x,y);
	IMPLEMENT_POINT_ACCESSOR3(y,y,x); IMPLEMENT_POINT_ACCESSOR3(y,y,y);

	IMPLEMENT_POINT_ACCESSOR4(x,x,x,x); IMPLEMENT_POINT_ACCESSOR4(x,x,x,y);
	IMPLEMENT_POINT_ACCESSOR4(x,x,y,x); IMPLEMENT_POINT_ACCESSOR4(x,x,y,y);
	IMPLEMENT_POINT_ACCESSOR4(x,y,x,x); IMPLEMENT_POINT_ACCESSOR4(x,y,x,y);
	IMPLEMENT_POINT_ACCESSOR4(x,y,y,x); IMPLEMENT_POINT_ACCESSOR4(x,y,y,y);

	IMPLEMENT_POINT_ACCESSOR4(y,x,x,x); IMPLEMENT_POINT_ACCESSOR4(y,x,x,y);
	IMPLEMENT_POINT_ACCESSOR4(y,x,y,x); IMPLEMENT_POINT_ACCESSOR4(y,x,y,y);
	IMPLEMENT_POINT_ACCESSOR4(y,y,x,x); IMPLEMENT_POINT_ACCESSOR4(y,y,x,y);
	IMPLEMENT_POINT_ACCESSOR4(y,y,y,x); IMPLEMENT_POINT_ACCESSOR4(y,y,y,y);

public:
	T x;
	T y;
};

template<class U, class T>
static inline Point<T, 2> operator * (const U &lhs, const Point<T, 2> &rhs);

//==============================================================================
//	TYPE DECLARATION
//==============================================================================
typedef Point<float, 2> 	Point2f;
typedef Point<int, 2> 		Point2i;
typedef Point<double, 2>	Point2d;

//==============================================================================
//	INLINED CODE
//==============================================================================
#include "Point2.inline.h"


#endif // __MK_GEOMETRY_POINT2__

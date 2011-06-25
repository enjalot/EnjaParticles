/*******************************************************************************
 File:				Quat.h

 Author: 			Gaspard Petit (gaspardpetit@gmail.com)
 Last Revision:		March 14, 2007
 
 This code may be reused without my permission as long as credits are given to
 the original author.  If you find bugs, please send me a note...
*******************************************************************************/
#ifndef __MK_GEOMETRY_QUAT__
#define __MK_GEOMETRY_QUAT__

//==============================================================================
//	FORWARD DECLARATIONS
//==============================================================================
template<class T, int S> class Point;
template<class T, int W, int H> class Matrix;

//==============================================================================
//	CLASS Quat
//==============================================================================
template<class T>
class Quat
{
public:
	Quat();
	Quat(T inW, T inX, T inY, T inZ);
	Quat(const Point<T,3> &axis, T angle);
	Quat(const Matrix<T, 1, 4> &rhs);
	Quat& operator = (const Matrix<T, 1, 4> &rhs);
	Quat(const Quat &rhs);
	Quat& operator = (const Quat &rhs);
	bool operator == (const Quat &rhs) const;
	bool operator != (const Quat &rhs) const;
	bool operator < (const Quat &rhs) const;
	bool operator <= (const Quat &rhs) const;
	bool operator > (const Quat &rhs) const;
	bool operator >= (const Quat &rhs) const;
	const T& operator [] (int index) const;
	operator T*();
	operator const T*() const;
	Quat operator + (const Quat &rhs) const;
	Quat operator - (const Quat &rhs) const;
	Quat operator * (const Quat &rhs) const;

public:
	
	Quat operator * (const T &rhs) const;
	Quat operator / (const T &rhs) const;
	Quat& operator += (const Quat &rhs);
	Quat& operator -= (const Quat &rhs);
	Quat& operator *= (const T &rhs);
	Quat& operator /= (const T &rhs);
	Quat& operator - ();
	float length();
	float length2();
	Quat& scale(float newLength);
	Quat& normalize();
	Quat conjugate();
	Quat unitInverse();	//	assumes we have a unit quaternion
	Quat inverse();
	void to4x4Matrix(Matrix<T, 4, 4> *outMatrix);
	void to3x3Matrix(Matrix<T, 3, 3> *outMatrix);

public:
	T w;
	T x;
	T y;
	T z;
};


template<class T>
static inline Quat<T> operator * (const T &lhs, const Quat<T> &rhs);

typedef Quat<float> 	Quatf;
typedef Quat<double>	Quatd;

//==============================================================================
//	INLINED CODE
//==============================================================================
#include "Quat.inline.h"

#endif // __MK_GEOMETRY_QUAT__

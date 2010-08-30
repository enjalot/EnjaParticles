#ifndef ISPH_VEC_H
#define ISPH_VEC_H

#include <cmath>

namespace isph {


/*!
 *	\struct	Vec
 *	\brief	Template for 2D or 3D vector with components of any number type.
 */
template<int dim, typename typ>
struct Vec
{
};


/*!
 *	\brief	2D vector with components of any number type.
 */
template<typename typ>
struct Vec<2,typ>
{
public:
	typ x, y;

	Vec()
	{ 
		x = y = 0;
	}

	Vec(typ v)
	{ 
		x = y = v;
	}

	Vec(typ X, typ Y)
	{
		x = X; y = Y;
	}

	Vec(typ X, typ Y, typ Z)
	{
		x = X; y = Y;
	}

	Vec(typ* v)
	{
		x = v[0]; y = v[1];
	}

	Vec<2,typ>& operator = (const Vec<3,typ>& other)
	{
		x = other.x;
		y = other.y;
		return *this;
	}

	inline typ& operator [] (unsigned int id)
	{
		return id ? y : x; //return (&x)[id];
	}

	inline typ dot(const Vec &v)
	{
		return x*v.x + y*v.y;
	}


	inline typ lengthSq()
	{
		return x*x + y*y;
	}

	inline typ length()
	{
		return sqrt(lengthSq());
	}

	inline Vec normalDir()
	{
		return Vec(-y, x) / length();
	}


	inline const Vec operator - (void) const
	{
		return Vec(-x, -y);
	}

	inline Vec operator + ( const Vec &v ) const
	{
		return Vec(x + v.x, y + v.y);
	}

	inline Vec operator - ( const Vec &v ) const
	{
		return Vec(x - v.x, y - v.y);
	}

	inline Vec operator * ( typ d ) const
	{
		return Vec(x*d, y*d);
	}
	
	inline Vec operator / ( typ d ) const
	{
		return Vec(x/d, y/d);
	}

	inline void operator += ( const Vec &v )
	{
		x += v.x; y += v.y;
	}
	
	inline void operator -= ( const Vec &v )
	{
		x -= v.x; y -= v.y;
	}
	
	inline void operator *= ( typ d )
	{
		x *= d; y *= d;
	}
	
	inline void operator /= ( typ d )
	{
		x /= d; y /= d;
	}

};


/*!
 *	\brief	3D vector with components of any number type.
 */
template<typename typ>
struct Vec<3,typ>
{
public:
	typ x, y, z;

	Vec()
	{ 
		x = y = z = 0;
	}

	Vec(typ v)
	{ 
		x = y = z = v;
	}

	Vec(typ X, typ Y)
	{
		x = X; y = Y; z = 0;
	}

	Vec(typ X, typ Y, typ Z)
	{
		x = X; y = Y; z = Z;
	}

	Vec(typ* v)
	{
		x = v[0]; y = v[1]; z = v[2];
	}

	Vec<3,typ>& operator = (const Vec<2,typ>& other)
	{
		x = other.x;
		y = other.y;
		z = 0;
		return *this;
	}

	inline typ& operator [] (unsigned int id)
	{
		return (&x)[id];
	}

	inline typ dot(const Vec &v)
	{
		return x*v.x + y*v.y + z*v.z;
	}

    inline Vec cross(const Vec &v)
	{
		return Vec(y - v.z + z - v.y, z - v.x + x - v.z , x - v.y + y - v.x);
	}

	inline typ lengthSq()
	{
		return x*x + y*y + z*z;
	}

	inline typ length()
	{
		return sqrt(lengthSq());
	}

	inline const Vec operator - (void) const
	{
		return Vec(-x, -y, -z);
	}

	inline Vec operator + ( const Vec &v ) const
	{
		return Vec(x + v.x, y + v.y, z + v.z);
	}

	inline Vec operator - ( const Vec &v ) const
	{
		return Vec(x - v.x, y - v.y, z - v.z);
	}

	inline Vec operator * ( typ d ) const
	{
		return Vec(x*d, y*d, z*d);
	}
	
	inline Vec operator / ( typ d ) const
	{
		return Vec(x/d, y/d, z/d);
	}

	inline void operator += ( const Vec &v )
	{
		x += v.x; y += v.y; z += v.z;
	}
	
	inline void operator -= ( const Vec &v )
	{
		x -= v.x; y -= v.y; z -= v.z;
	}
	
	inline void operator *= ( typ d )
	{
		x *= d; y *= d; z *= d;
	}
	
	inline void operator /= ( typ d )
	{
		x /= d; y /= d; z /= d;
	}

};


/*!
 *	\brief	4D vector with components of any number type.
 */
template<typename typ>
struct Vec<4,typ>
{
public:
	typ x, y, z, w;

	Vec()
	{ 
		x = y = z = w = 0;
	}

	Vec(typ v)
	{ 
		x = y = z = w = v;
	}

	Vec(typ X, typ Y, typ Z, typ W)
	{
		x = X; y = Y; z = Z; w = W;
	}

	Vec(typ* v)
	{
		x = v[0]; y = v[1]; z = v[2]; w = v[3];
	}

	inline typ& operator [] (unsigned int id)
	{
		return (&x)[id];
	}

	inline typ dot(const Vec &v)
	{
		return x*v.x + y*v.y + z*v.z - w*v.w;
	}

    
	inline typ lengthSq()
	{
		return x*x + y*y + z*z + w*w;
	}

	inline typ length()
	{
		return sqrt(lengthSq());
	}

	inline const Vec operator - (void) const
	{
		return Vec(-x, -y, -z, -w);
	}

	inline Vec operator + ( const Vec &v ) const
	{
		return Vec(x + v.x, y + v.y, z + v.z, w + v.w);
	}

	inline Vec operator - ( const Vec &v ) const
	{
		return Vec(x - v.x, y - v.y, z - v.z, w - v.w);
	}

	inline Vec operator * ( typ d ) const
	{
		return Vec(x*d, y*d, z*d, w*d);
	}
	
	inline Vec operator / ( typ d ) const
	{
		return Vec(x/d, y/d, z/d, w/d);
	}

	inline void operator += ( const Vec &v )
	{
		x += v.x; y += v.y; z += v.z; w += v.w;
	}
	
	inline void operator -= ( const Vec &v )
	{
		x -= v.x; y -= v.y; z -= v.z; w -= v.w;
	}
	
	inline void operator *= ( typ d )
	{
		x *= d; y *= d; z *= d; w *= d;
	}
	
	inline void operator /= ( typ d )
	{
		x /= d; y /= d; z /= d; w /= d;
	}

};


/*!
 *	\brief	Multiplication operator: number*vector (number is left operand).
 */
template<int dim, typename typ, typename typ2>
inline Vec<dim,typ> operator * (typ2 d, const Vec<dim,typ>& v)
{
    return v * static_cast<typ>(d);
}


// some easy vector creating

template<typename typ1, typename typ2>
inline Vec<2,float> Vec2f(typ1 x, typ2 y)
{
	return Vec<2,float>((float)x, (float)y);
}

template<typename typ1, typename typ2>
inline Vec<2,double> Vec2d(typ1 x, typ2 y)
{
	return Vec<2,double>(x, y);
}

template<typename typ1, typename typ2, typename typ3>
inline Vec<3,float> Vec3f(typ1 x, typ2 y, typ3 z)
{
	return Vec<3,float>((float)x, (float)y, (float)z);
}

template<typename typ1, typename typ2, typename typ3>
inline Vec<3,double> Vec3d(typ1 x, typ2 y, typ3 z)
{
	return Vec<3,double>(x, y, z);
}


} // namespace isph

#endif

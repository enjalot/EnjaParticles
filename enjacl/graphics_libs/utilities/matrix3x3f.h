#ifndef _MATRIX3X3F_H_
#define _MATRIX3X3F_H_

#include "Vec3.h"

class Matrix3x3f {
private:
	Vec3 rows[3];

public:
	Matrix3x3f(Vec3& v1, Vec3& v2, Vec3& v3);
	Matrix3x3f();
	//----------------------------------------------------------------------
	void setTo(float val);
	//----------------------------------------------------------------------
	float& operator()(int i, int j) { return rows[i][j]; } 
	//----------------------------------------------------------------------
 	//Vec3& operator*(Vec3& v)
	//{
		//Vec3& vv = *(new Vec3(rows[0]*v, rows[1]*v, rows[2]*v)); // calls constructor of Vec3 3 times
		//return vv;
	//}
	//----------------------------------------------------------------------
 	Vec3& operator*(Vec3& v) 
	{

		float r0 = rows[0]*v;
		float r1 = rows[1]*v;
		float r2 = rows[2]*v;
		Vec3* vv = new Vec3(r0,r1,r2);  // calls constructor of Vec3 3 times 
		//Vec3& vv = *(new Vec3(2.,3.,4.)); // ok
		return *vv;
	}
	//----------------------------------------------------------------------

	friend Vec3& operator*(Vec3& v, Matrix3x3f& mat)
	{
		float r0 = v[0]*mat(0,0) + v[1]*mat(1,0) + v[2]*mat(2,0);
		float r1 = v[0]*mat(0,1) + v[1]*mat(1,1) + v[2]*mat(2,1);
		float r2 = v[0]*mat(0,2) + v[1]*mat(1,2) + v[2]*mat(2,2);
		Vec3& vv = *(new Vec3(r0, r1, r2));
		return vv;
	}
	//----------------------------------------------------------------------
};

#endif

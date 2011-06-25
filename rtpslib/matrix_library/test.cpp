#include "Quat.h"
#include "Point.h"
#include "Matrix.h"

#include <cstdio>
#include <cmath>

int main(int argc, const char *argv[])
{
	// rotate a point around an axis
	Point3d axis = Point3d(1.0, 1.0, 0.0).normalized();
	Quatd q(axis, 2*M_PI/36);
	
	Quatd qpt(0, 1, 0, 0); // x = 1, y =0, z = 0
	
	// do a 360 around the axis
	printf("doing a 360 around axis (%f %f %f) using a quaternion\n", axis.x, axis.y, axis.z);
	for (int i = 0; i < 36; ++i)
	{
		qpt = q*qpt*q.unitInverse();
		printf("(%f %f %f)\n", qpt.x, qpt.y, qpt.z);
	}
	
	// same thing, but with a rotation matrix
	
	Matrix4d m;
	q.to4x4Matrix(&m);
	Point4d pt(qpt.x, qpt.y, qpt.z, 1.0);
	
	// do a 360 around the axis
	printf("\n\ndoing a 360 around axis (%f %f %f) using a matrix\n", axis.x, axis.y, axis.z);
	for (int i = 0; i < 36; ++i)
	{
		pt = m*pt;
		printf("(%f %f %f)\n", pt.x, pt.y, pt.z);
	}
	
	return 0;
}

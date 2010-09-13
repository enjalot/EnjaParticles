#include <stdio.h>
#include <stdlib.h>
#include "ArrayT.h"
#include "Vec3.h"
#include "matrix3x3f.h"

void testMatrix3x3f()
{
	Vec3 row1(1.,2.,3.);
	Vec3 row2(3.,4.,5.);
	Vec3 row3(7.,8.,9.);

	Matrix3x3f mat(row1, row2, row3);
	Vec3 v = mat*row1;
	Vec3 w = row1*mat;
	v.print("mat*row1");
	w.print("row1*mat");
	printf("row1*mat*row1= %f\n", row1*mat*row1);
}
//----------------------------------------------------------------------
void testArrayT_float()
{
	ArrayT<float> a(10,20);
	ArrayT<float> b(10,20);
	ArrayT<float> c(10,20);
	a.setTo(45.);
	b.setTo(40.);
	const ArrayT<float>& d = a + b;
	printf("d = %f\n", d[3]);

	c = a + b;
	printf("c = %f\n", c[3]);
	printf("testArrayT\n");
}
//----------------------------------------------------------------------
void testArrayT_Vec3()
{
	ArrayT<Vec3> a(10,20);
	ArrayT<Vec3> b(10,20);
	a.setTo(Vec3(1.,2.,3.));
	Vec3 vz(2.,4.,-3.);
	b.setTo(vz);

	const ArrayT<Vec3>& d = a + b;
	a[3].print("a[3]");
	b[5].print("b[5]");
	d[5].print("c = a + b");

	//ArrayT<Vec3> c(10,20);
	ArrayT<Vec3> c(d);
	c[5].print("c(d)");
	c = a; // operator= not working properly
	c[5].print("c = a");

	c = a + vz;
	//c = vz + a;
	vz.print("vz");
	c[5].print("c = a + vz");

	c = vz + a;
	c[5].print("c = a + vz");
}
//----------------------------------------------------------------------
int main()
{
	//testArrayT_float();
	testMatrix3x3f();
	testArrayT_Vec3();
	exit(0);
}
//----------------------------------------------------------------------

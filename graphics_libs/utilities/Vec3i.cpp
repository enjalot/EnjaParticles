//================================================================================

// $Author: erlebach $
// $Date: 2002/01/05 15:18:28 $
// $Header: /home/erlebach/D2/src/CVS/computeCP/Vec3i.cpp,v 2.4 2002/01/05 15:18:28 erlebach Exp $
// Sticky tag, $Name:  $
// $RCSfile: Vec3i.cpp,v $
// $Revision: 2.4 $
// $State: Exp $

//================================================================================

#include <math.h>
#include <stdio.h>
#include "Vec3i.h"

Vec3i::Vec3i()
{
    Vec3i((int) 0, (int) 0, (int) 0);
}
//--------------------------------
Vec3i::Vec3i(int x, int y, int z)
{
    vec[0] = x;
    vec[1] = y;
    vec[2] = z;
}
//----------------------------------------------------------------------
//Vec3i::Vec3i(int pt[3])
//{
    //vec[0] = pt[0];
    //vec[1] = pt[1];
    //vec[2] = pt[2];
//}
//----------------------------------------------------------------------
Vec3i::Vec3i(const int* pt)
{
    vec[0] = *pt++;
    vec[1] = *pt++;
    vec[2] = *pt;
}
//----------------------------------------------------------------------
Vec3i::Vec3i(int* pt)
{
    vec[0] = *pt++;
    vec[1] = *pt++;
    vec[2] = *pt;
}
//---------------------------------
Vec3i::Vec3i(Vec3i& vec)
{
    this->vec[0] = vec.x();
    this->vec[1] = vec.y();
    this->vec[2] = vec.z();
}
//---------------------------------------
void Vec3i::print(const char *msg) const
{
	if (msg) {
    	printf("%s: %d, %d, %d\n", msg, vec[0], vec[1], vec[2]);
	} else {
    	printf("%d, %d, %d\n", vec[0], vec[1], vec[2]);
	}
}
//---------------------------------
void Vec3i::getVec(int* x, int* y, int* z)
{
    *x = vec[0];
    *y = vec[1];
    *z = vec[2];
}
//------------------------------------------
int* Vec3i::getVec()
{
    return vec;
}
//------------------------------------------
void Vec3i::setValue(int x, int y, int z)
{
    vec[0] = x;
    vec[1] = y;
    vec[2] = z;
}
//--------------------------------------
void Vec3i::clampMinTo(const Vec3i& v)
{
	if (vec[0] < v[0]) vec[0] = v[0];
	if (vec[1] < v[1]) vec[1] = v[1];
	if (vec[2] < v[2]) vec[2] = v[2];
}
//--------------------------------------
void Vec3i::clampMaxTo(const Vec3i& v)
{
	if (vec[0] > v[0]) vec[0] = v[0];
	if (vec[1] > v[1]) vec[1] = v[1];
	if (vec[2] > v[2]) vec[2] = v[2];
}
//--------------------------------------
#ifdef STANDALONE
void main()
{
	Vec3i a(1,2,3);
	Vec3i b(4,1,-3);
	Vec3i c = a + b;
	Vec3i d = a + b + Vec3i(1,1,1);
	c.print("c = a + b = ");
	d.print("c = a + b + 1 = ");
	a.print("a= ");
	b.print("b= ");
	Vec3i e;
	Vec3i f;
	e = b.min(a);
	f = a.min(b);
	e.print("a.min(b) = "); // change a
	f.print("b.min(a) = "); // change b
	e = a.max(b);
	f = b.max(a);
	e.print("a.max(b) = "); // change a
	f.print("b.max(a) = "); // change b
	e = a + b;
	e.print("a + b= ");
}
#endif
//----------------------------------------------------------------------


/*
 * Written by Dr. Gordon Erlebacher.
 */

#include <math.h>
#include <stdio.h>
#include <iostream>
//#include <Amira/HxMessage.h>
#include "CVec3.h"

using namespace std;

//=======================================================================

#if 0
CVec3::CVec3()
{
    CVec3((float)0., (float)0., (float)0.);
}

//=======================================================================

CVec3::CVec3(int x, int y, int z)
{
    vec[0] = (float) x;
    vec[1] = (float) y;
    vec[2] = (float) z;
}

//=======================================================================

CVec3::CVec3(float x, float y, float z)
{
    vec[0] = CMPLX(x,0.);
    vec[1] = CMPLX(y,0.);
    vec[2] = CMPLX(z,0.);
}

//=======================================================================

CVec3::CVec3(CMPLX x, CMPLX y, CMPLX z)
{
    vec[0] = x;
    vec[1] = y;
    vec[2] = z;
}

//=======================================================================

CVec3::CVec3(float* pt)
{
    vec[0] = CMPLX(pt[0],0.);
    vec[1] = CMPLX(pt[1],0.);
    vec[2] = CMPLX(pt[2],0.);
}

//=======================================================================

CVec3::CVec3(CMPLX* pt)
{
    vec[0] = *pt++;
    vec[1] = *pt++;
    vec[2] = *pt;
}
#endif

//=======================================================================

CVec3::CVec3(CVec3& vec)
{
    this->vec[0] = vec.x();
    this->vec[1] = vec.y();
    this->vec[2] = vec.z();
}

//----------------------------------------------------------------------
CVec3::CVec3(Vec3& vec)
{
    this->vec[0] = CMPLX(vec.x(), 0.);
    this->vec[1] = CMPLX(vec.y(), 0.);
    this->vec[2] = CMPLX(vec.z(), 0.);
}

//=======================================================================

CMPLX* CVec3::getVec()
{
    return &vec[0];
}

//=======================================================================

void CVec3::getVec(CMPLX* x, CMPLX* y, CMPLX* z)
{
    *x = vec[0];
    *y = vec[1];
    *z = vec[2];
}

//=======================================================================
void CVec3::setValue(float x)
{
    vec[0] = x;
    vec[1] = x;
    vec[2] = x;
}
//=======================================================================

void CVec3::setValue(float x, float y, float z)
{
    vec[0] = x;
    vec[1] = y;
    vec[2] = z;
	printf("vec3, setValue\n");
}

//=======================================================================

void CVec3::setValue(CVec3& v)
{
	vec[0] = v[0];
	vec[1] = v[1];
	vec[2] = v[2];
}

//=======================================================================

void CVec3::setValue(float* val)
{
	vec[0] = val[0];
	vec[1] = val[1];
	vec[2] = val[2];
}
//=======================================================================

void CVec3::normalize(float scale)
{
    float norm = sqrt(scalprod(*this,*this));
    if (norm != 0.0)
        norm = 1.0/norm;
    else
        norm = 1.0;
    vec[0] *= norm*scale;
    vec[1] *= norm*scale;
    vec[2] *= norm*scale;
}

//=======================================================================

double CVec3::magnitude()
{
    return sqrt(scalprod(*this,*this));
}

double CVec3::magnitude() const
{
    return sqrt(scalprod(*this, *this));
}

//=======================================================================

void CVec3::print(const char *msg) const
{
	if (msg) {
        cout << msg;
        }
        cout << vec[0] << ", " << vec[1] << ", " << vec[2] << endl;
}
//----------------------------------------------------------------------
std::ostream&
operator<< (std::ostream&  os,
            const CVec3& p)
{
    os << '(' << p.x()  << ',' << p.y() << ',' << p.z() << ')';
    if (os.fail())
        cout << "operator<<(ostream&,IntVect&) failed" << endl;
    return os;
}

//----------------------------------------------------------------------
#ifdef STANDALONE

//void testvec(CVec3& a)
//{
	//a.print("inside testvec CVec3&, a= ");
//}
void testvec(CVec3 a)
{
	a.print("inside testvec CVec3, a= ");
}

// Problems occur when I use testvec(CVec3&) and 
// I allocate the vector on the stack. That is probably 
// because I cannot take a reference of such a vector.
// Therefore, one should either work with testvec(CVec3) 
// or testvec(CVec3&) but not both (for safety). It is also 
// safer not to allocate memory for the the arguments in
// place when calling the function IF the function argument 
// is a reference.

int main()
{
	CVec3 a(.2,.5,.7);
	CVec3 b(-.2,-.2,.8);

	CVec3 c;

	c = a + b;
	a.print("a= ");
	b.print("b= ");
	c.print("a+b");
	c = a - b;
	c.print("a-b");
	c = b - a;
	c.print("b-a");
	(a-b).print("inline a-b: ");

	CVec3 d = c  + b - 3*c;
	(a^b).print("a^b" );

    CVec3 dd = CVec3(.2,.6,.9) + a;
	testvec(CVec3(.2,.2,.2)+CVec3(.1,.1,.1));
	testvec(dd^CVec3(.2,.3,.5));
	testvec(dd += CVec3(.3, .7. .2));

	return 0;
}
#endif

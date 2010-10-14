#include "matrix3x3f.h"

Matrix3x3f::Matrix3x3f()
{
}
//----------------------------------------------------------------------
Matrix3x3f::Matrix3x3f(Vec3& v1, Vec3& v2, Vec3& v3)
{
	rows[0] = v1;
	rows[1] = v2;
	rows[2] = v3;
}
//----------------------------------------------------------------------
void Matrix3x3f::setTo(float val)
{
	rows[0].setValue(val, val, val);
	rows[1].setValue(val, val, val);
	rows[2].setValue(val, val, val);
}
//----------------------------------------------------------------------

/****************************************************************************************
* Real-Time Particle System - An OpenCL based Particle system developed to run on modern GPUs. Includes SPH fluid simulations.
* version 1.0, September 14th 2011
* 
* Copyright (C) 2011 Ian Johnson, Andrew Young, Gordon Erlebacher, Myrna Merced, Evan Bollig
* 
* This software is provided 'as-is', without any express or implied
* warranty.  In no event will the authors be held liable for any damages
* arising from the use of this software.
* 
* Permission is granted to anyone to use this software for any purpose,
* including commercial applications, and to alter it and redistribute it
* freely, subject to the following restrictions:
* 
* 1. The origin of this software must not be misrepresented; you must not
* claim that you wrote the original software. If you use this software
* in a product, an acknowledgment in the product documentation would be
* appreciated but is not required.
* 2. Altered source versions must be plainly marked as such, and must not be
* misrepresented as being the original software.
* 3. This notice may not be removed or altered from any source distribution.
****************************************************************************************/


// Matrix 4x4.cpp
//


#include "Matrix 4x4.h"
#include "Math\FastSqrt.h"
#include "Math\FastTrigonometry.h"
#include <math.h>


const Matrix4x4 IdentityMatrix = { 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f };


#define MATRIX_ORTHONORMAL_TOLERANCE    0.001
#define MATRIX_ORTHOGONAL_TOLERANCE     0.001


Bool MatrixIsValid(RMatrix4x4 matrix)
{
    if ((matrix.f11 * matrix.f11) < 0.0f)
		return FALSE;
	if ((matrix.f21 * matrix.f21) < 0.0f)
		return FALSE;
	if ((matrix.f31 * matrix.f31) < 0.0f)
		return FALSE;
	if ((matrix.f41 * matrix.f41) < 0.0f)
		return FALSE;

	if ((matrix.f12 * matrix.f12) < 0.0f)
		return FALSE;
	if ((matrix.f22 * matrix.f22) < 0.0f)
		return FALSE;
	if ((matrix.f32 * matrix.f32) < 0.0f)
		return FALSE;
	if ((matrix.f42 * matrix.f42) < 0.0f)
		return FALSE;

	if ((matrix.f13 * matrix.f13) < 0.0f)
		return FALSE;
	if ((matrix.f23 * matrix.f23) < 0.0f)
		return FALSE;
	if ((matrix.f33 * matrix.f33) < 0.0f)
		return FALSE;
	if ((matrix.f43 * matrix.f43) < 0.0f)
		return FALSE;

	if ((matrix.f14 * matrix.f14) < 0.0f)
		return FALSE;
	if ((matrix.f24 * matrix.f24) < 0.0f)
		return FALSE;
	if ((matrix.f34 * matrix.f34) < 0.0f)
		return FALSE;
	if ((matrix.f44 * matrix.f44) < 0.0f)
		return FALSE;

	return TRUE;
}

Bool MatrixIsOrthonormal(RMatrix4x4 matrix)
{
	Vector3 normal12;
	Vector3 col3;
	
	normal12.m_X = (matrix.f21 * matrix.f32) - (matrix.f31 * matrix.f22);
	normal12.m_Y = (matrix.f31 * matrix.f12) - (matrix.f11 * matrix.f32);
	normal12.m_Z = (matrix.f11 * matrix.f22) - (matrix.f21 * matrix.f12);

	col3.m_X = matrix.f13;
	col3.m_Y = matrix.f23;
	col3.m_Z = matrix.f33;
	
	if (!Vec3Compare(normal12, col3, (Float) MATRIX_ORTHONORMAL_TOLERANCE))
	{
        Vec3Invert(col3);
		if (!Vec3Compare(normal12, col3, (Float) MATRIX_ORTHONORMAL_TOLERANCE))
		{
			return FALSE;
		}
	}

	return TRUE;
}

Bool MatrixIsOrthogonal(RMatrix4x4 matrix)
{
	Vector3 normal12;
	Vector3 col3;
	
	normal12.m_X = (matrix.f21 * matrix.f32) - (matrix.f31 * matrix.f22);
	normal12.m_Y = (matrix.f31 * matrix.f12) - (matrix.f11 * matrix.f32);
	normal12.m_Z = (matrix.f11 * matrix.f22) - (matrix.f21 * matrix.f12);

	col3.m_X = matrix.f13;
	col3.m_Y = matrix.f23;
	col3.m_Z = matrix.f33;

	Vec3Normalize(normal12);
	Vec3Normalize(col3);

	if (!Vec3Compare(normal12, col3, (Float) MATRIX_ORTHONORMAL_TOLERANCE))
	{
        Vec3Invert(col3);
		if (!Vec3Compare(normal12, col3, (Float) MATRIX_ORTHONORMAL_TOLERANCE))
		{
			return FALSE;
		}
	}

	return TRUE;
}

Void MatrixOrthonormalize(RMatrix4x4 matrix)
{
	Float fLength = FastSqrt((matrix.f11 * matrix.f11) + (matrix.f21 * matrix.f21) + (matrix.f31 * matrix.f31));
	if (fLength > 0.0f)
	{
		fLength = 1.0f / fLength;
		matrix.f11 *= fLength;
		matrix.f21 *= fLength;
		matrix.f31 *= fLength;
	}

	fLength = FastSqrt((matrix.f12 * matrix.f12) + (matrix.f22 * matrix.f22) + (matrix.f32 * matrix.f32));
	if (fLength > 0.0f)
	{
		fLength = 1.0f / fLength;
		matrix.f12 *= fLength;
		matrix.f22 *= fLength;
		matrix.f32 *= fLength;
	}

	matrix.f13 = (matrix.f21 * matrix.f32) - (matrix.f31 * matrix.f22);
	matrix.f23 = (matrix.f31 * matrix.f12) - (matrix.f11 * matrix.f32);
	matrix.f33 = (matrix.f11 * matrix.f22) - (matrix.f21 * matrix.f12);
}

Void MatrixSetXRotation(RMatrix4x4 matrix, Float m_XAngle)
{
	Float sinx, cosx;

	sinx = SinCos(m_XAngle, &cosx);

	matrix.f11 = matrix.f44 = 1.0f;
	matrix.f22 =  cosx;
	matrix.f23 = -sinx;
	matrix.f32 =  sinx;
	matrix.f33 =  cosx;

	matrix.f12 = matrix.f13 = matrix.f14 = matrix.f21 = matrix.f24 = matrix.f31 = matrix.f34 = matrix.f41 = matrix.f42 = matrix.f43 = 0.0f;
}

Void MatrixSetYRotation(RMatrix4x4 matrix, Float m_YAngle)
{
	Float sy, cy;

	sy = SinCos(m_YAngle, &cy);

	matrix.f22 = matrix.f44 = 1.0f;
	matrix.f11 =  cy;
	matrix.f13 =  sy;
    matrix.f31 = -sy;
	matrix.f33 =  cy;

	matrix.f12 = matrix.f14 = matrix.f21 = matrix.f23 = matrix.f24 = matrix.f32 = matrix.f34 = matrix.f41 = matrix.f42 = matrix.f43 = 0.0f;
}

Void MatrixSetZRotation(RMatrix4x4 matrix, Float m_ZAngle)
{
	Float sz, cz;

	sz = SinCos(m_ZAngle, &cz);

	matrix.f33 = matrix.f44 = 1.0f;
	matrix.f11 =  cz;
	matrix.f12 = -sz;
	matrix.f21 =  sz;
	matrix.f22 =  cz;

	matrix.f13 = matrix.f14 = matrix.f23 = matrix.f24 = matrix.f31 = matrix.f32 = matrix.f34 = matrix.f41 = matrix.f42 = matrix.f43 = 0.0f;
}

Void MatrixSetXYZRotation(RMatrix4x4 matrix, Float m_XAngle, Float m_YAngle, Float m_ZAngle)
{
    Float sx, cxx;
	Float sy, cy;
	Float sz, cz;

	sx = SinCos(m_XAngle, &cxx);
	sy = SinCos(m_YAngle, &cy);
	sz = SinCos(m_ZAngle, &cz);

	matrix.f11 = cy * cz;
	matrix.f12 = -cy * sz;
	matrix.f13 = sy;

	matrix.f21 = cz * sx * sy + cxx * sz;
	matrix.f22 = cxx * cz - sx * sy * sz;
	matrix.f23 = -cy * sx;
	
	matrix.f31 = -cxx * cz * sy + sx * sz;
	matrix.f32 = cz * sx + cxx * sy + sz;
	matrix.f33 = cxx * cy;

	matrix.f44 = 1.0f;
	matrix.f14 = matrix.f24 = matrix.f34 = matrix.f41 = matrix.f42 = matrix.f43 = 0.0f;
}

Void MatrixSetRotationScalingTranslation(RMatrix4x4 matrix, RVector3 vRotation, RVector3 vScaling, RVector3 vTranslation)
{
	Vector3 s, c;
	SinCosVec(&vRotation, &s, &c);

	matrix.f11 = vScaling.m_X * ( c.m_Y * c.m_Z);
	matrix.f12 = vScaling.m_Y * (-c.m_Y * s.m_Z);
	matrix.f13 = vScaling.m_Z * ( s.m_Y);
	matrix.f14 = vTranslation.m_X;

	matrix.f21 = vScaling.m_X * ( c.m_Z * s.m_X * s.m_Y + c.m_X * s.m_Z);
	matrix.f22 = vScaling.m_Y * ( c.m_X * c.m_Z - s.m_X * s.m_Y * s.m_Z);
	matrix.f23 = vScaling.m_Z * (-c.m_Y * s.m_X);
	matrix.f24 = vTranslation.m_Y;
	
	matrix.f31 = vScaling.m_X * (-c.m_X * c.m_Z * s.m_Y + s.m_X * s.m_Z);
	matrix.f32 = vScaling.m_Y * ( c.m_Z * s.m_X + c.m_X * s.m_Y + s.m_Z);
	matrix.f33 = vScaling.m_Z * ( c.m_X * c.m_Y);
	matrix.f34 = vTranslation.m_Z;

	matrix.f41 = matrix.f42 = matrix.f43 = 0.0f;
	matrix.f44 = 1.0f;
}

Bool MatrixSetViewMatrix(RMatrix4x4 matrix, RVector3 vFrom, RVector3 vView, RVector3 vWorldUp)
{
	Vector3 vUp, vRight;
	Float fDotProduct = (vWorldUp.m_X * vView.m_X) + (vWorldUp.m_Y * vView.m_Y) + (vWorldUp.m_Z * vView.m_Z);
	
	vUp.m_X = vWorldUp.m_X - (fDotProduct * vView.m_X);
	vUp.m_Y = vWorldUp.m_Y - (fDotProduct * vView.m_Y);
	vUp.m_Z = vWorldUp.m_Z - (fDotProduct * vView.m_Z);

	Float fLength = FastSqrt((vUp.m_X * vUp.m_X) + (vUp.m_Y * vUp.m_Y) + (vUp.m_Z * vUp.m_Z));
    if (1E-6f > fLength)
    {
		vUp.m_X = 0.0f - (vView.m_Y * vView.m_X);
		vUp.m_Y = 1.0f - (vView.m_Y * vView.m_Y);
		vUp.m_Z = 0.0f - (vView.m_Y * vView.m_Z);

        fLength = FastSqrt((vUp.m_X * vUp.m_X) + (vUp.m_Y * vUp.m_Y) + (vUp.m_Z * vUp.m_Z));
        if (1E-6f > fLength)
        {
			vUp.m_X = 0.0f - (vView.m_Z * vView.m_X);
			vUp.m_Y = 0.0f - (vView.m_Z * vView.m_Y);
			vUp.m_Z = 1.0f - (vView.m_Z * vView.m_Z);

			fLength = FastSqrt((vUp.m_X * vUp.m_X) + (vUp.m_Y * vUp.m_Y) + (vUp.m_Z * vUp.m_Z));
            if (1E-6f > fLength)
			{
                return FALSE;
			}
        }
    }

	fLength = 1.0f / fLength;
	vUp.m_X *= fLength;
	vUp.m_Y *= fLength;
	vUp.m_Z *= fLength;

	vRight.m_X = vUp.m_Y * vView.m_Z - vUp.m_Z * vView.m_Y;
	vRight.m_Y = vUp.m_Z * vView.m_X - vUp.m_X * vView.m_Z;
	vRight.m_Z = vUp.m_X * vView.m_Y - vUp.m_Y * vView.m_X;

    matrix.f11 = vRight.m_X;
	matrix.f12 = vRight.m_Y;
	matrix.f13 = vRight.m_Z;
	matrix.f14 = - ((vFrom.m_X * vRight.m_X) + (vFrom.m_Y * vRight.m_Y) + (vFrom.m_Z * vRight.m_Z));
	
	matrix.f21 = vUp.m_X;
	matrix.f22 = vUp.m_Y;
	matrix.f23 = vUp.m_Z;
	matrix.f24 = - ((vFrom.m_X * vUp.m_X)    + (vFrom.m_Y * vUp.m_Y)    + (vFrom.m_Z * vUp.m_Z));

	matrix.f31 = vView.m_X;
	matrix.f32 = vView.m_Y;
	matrix.f33 = vView.m_Z;
	matrix.f34 = - ((vFrom.m_X * vView.m_X) + (vFrom.m_Y * vView.m_Y) + (vFrom.m_Z * vView.m_Z));

	matrix.f41 = matrix.f42 = matrix.f43 = 0.0f;
	matrix.f44 = 1.0f;

	return TRUE;
}

Void MatrixSetOrthogonalMatrix(RMatrix4x4 matrix, Float fLeft, Float fRight, Float fBottom, Float fTop, Float fNear, Float fFar)
{
	matrix.f11 = 2.0f / (fRight - fLeft);
	matrix.f12 = 0.0f;
	matrix.f13 = 0.0f;
	matrix.f14 = (fLeft + fRight) / (fLeft - fRight);

	matrix.f21 = 0.0f;
	matrix.f22 = 2.0f / (fTop - fBottom);
	matrix.f23 = 0.0f;
	matrix.f24 = (fBottom + fTop) / (fBottom - fTop);

	matrix.f31 = 0.0f;
	matrix.f32 = 0.0f;
	matrix.f33 = 2.0f / (fNear - fFar);
	matrix.f34 = (fNear + fFar) / (fNear - fFar);

	matrix.f14 = matrix.f24 = matrix.f34 = 0.0f;
	matrix.f44 = 1.0f;
}

Void MatrixSetOrthogonal2DMatrix(RMatrix4x4 matrix, Float fLeft, Float fRight, Float fBottom, Float fTop)
{
	matrix.f11 = 2.0f / (fRight - fLeft);
	matrix.f12 = 0.0f;
	matrix.f13 = 0.0f;
	matrix.f14 = (fLeft + fRight) / (fLeft - fRight);

	matrix.f21 = 0.0f;
	matrix.f22 = 2.0f / (fTop - fBottom);
	matrix.f23 = 0.0f;
	matrix.f24 = (fBottom + fTop) / (fBottom - fTop);

	matrix.f31 = 0.0f;
	matrix.f32 = 0.0f;
	matrix.f33 = -1.0f;
	matrix.f34 = 0.0f;
	
	matrix.f41 = matrix.f42 = matrix.f43 = 0.0f;
	matrix.f44 = 1.0f;
}

Void MatrixSetFrustumMatrix(RMatrix4x4 matrix, Float fLeft, Float fRight, Float fBottom, Float fTop, Float fNear, Float fFar)
{
	matrix.f11 = (2.0f * fNear) / (fRight - fLeft);
	matrix.f12 = 0.0f;
	matrix.f13 = (fRight + fLeft) / (fRight - fLeft);
	matrix.f14 = 0.0f;

	matrix.f21 = 0.0f;
	matrix.f22 = (2.0f * fNear) / (fTop - fBottom);
	matrix.f23 = (fTop + fBottom) / (fTop - fBottom);
	matrix.f24 = 0.0f;

	matrix.f31 = 0.0f;
	matrix.f32 = 0.0f;
	matrix.f33 = (fFar + fNear) / (fNear - fFar);
	matrix.f34 = (2.0f * fFar * fNear) / (fNear - fFar);

	matrix.f41 = 0.0f;
	matrix.f42 = 0.0f;
	matrix.f43 = -1.0f;
	matrix.f44 = 0.0f;
}

Void MatrixSetPerspectiveMatrix(RMatrix4x4 matrix, Float fFov, Float fAspect, Float fNear, Float fFar)
{
	fFov = CoTan(fFov * 0.5f);

	matrix.f11 = fNear / (fFov * fAspect);
	matrix.f12 = 0.0f;
	matrix.f13 = 0.0f;
	matrix.f14 = 0.0f;

	matrix.f21 = 0.0f;
	matrix.f22 = fNear / fFov;
	matrix.f23 = 0.0f;
	matrix.f24 = 0.0f;

	matrix.f31 = 0.0f;
	matrix.f32 = 0.0f;
	matrix.f33 = (fFar + fNear) / (fNear - fFar);
	matrix.f34 = (2.0f * fFar * fNear) / (fNear - fFar);

	matrix.f41 = 0.0f;
	matrix.f42 = 0.0f;
	matrix.f43 = -1.0f;
	matrix.f44 = 0.0f;
}

Void MatrixAdd(RMatrix4x4 m1, RMatrix4x4 m2, RMatrix4x4 mdest)
{
    mdest.f11 = m1.f11 + m2.f11;
	mdest.f12 = m1.f12 + m2.f12;
	mdest.f13 = m1.f13 + m2.f13;
	mdest.f14 = m1.f14 + m2.f14;

	mdest.f21 = m1.f21 + m2.f21;
	mdest.f22 = m1.f22 + m2.f22;
	mdest.f23 = m1.f23 + m2.f23;
	mdest.f24 = m1.f24 + m2.f24;

	mdest.f31 = m1.f31 + m2.f31;
	mdest.f32 = m1.f32 + m2.f32;
	mdest.f33 = m1.f33 + m2.f33;
	mdest.f34 = m1.f34 + m2.f34;

	mdest.f41 = m1.f41 + m2.f41;
	mdest.f42 = m1.f42 + m2.f42;
	mdest.f43 = m1.f43 + m2.f43;
	mdest.f44 = m1.f44 + m2.f44;
}

Void MatrixMultiply(RMatrix4x4 m1, RMatrix4x4 m2, RMatrix4x4 mdest)
{
	Matrix4x4 tmp;

    tmp.f11 = (m1.f11 * m2.f11) + (m1.f12 * m2.f21) + (m1.f13 * m2.f31) + (m1.f14 * m2.f41);
	tmp.f12 = (m1.f11 * m2.f12) + (m1.f12 * m2.f22) + (m1.f13 * m2.f32) + (m1.f14 * m2.f42);
	tmp.f13 = (m1.f11 * m2.f13) + (m1.f12 * m2.f23) + (m1.f13 * m2.f33) + (m1.f14 * m2.f43);
	tmp.f14 = (m1.f11 * m2.f14) + (m1.f12 * m2.f24) + (m1.f13 * m2.f34) + (m1.f14 * m2.f44);

	tmp.f21 = (m1.f21 * m2.f11) + (m1.f22 * m2.f21) + (m1.f23 * m2.f31) + (m1.f24 * m2.f41);
	tmp.f22 = (m1.f21 * m2.f12) + (m1.f22 * m2.f22) + (m1.f23 * m2.f32) + (m1.f24 * m2.f42);
	tmp.f23 = (m1.f21 * m2.f13) + (m1.f22 * m2.f23) + (m1.f23 * m2.f33) + (m1.f24 * m2.f43);
	tmp.f24 = (m1.f21 * m2.f14) + (m1.f22 * m2.f24) + (m1.f23 * m2.f34) + (m1.f24 * m2.f44);

	tmp.f31 = (m1.f31 * m2.f11) + (m1.f32 * m2.f21) + (m1.f33 * m2.f31) + (m1.f34 * m2.f41);
	tmp.f32 = (m1.f31 * m2.f12) + (m1.f32 * m2.f22) + (m1.f33 * m2.f32) + (m1.f34 * m2.f42);
	tmp.f33 = (m1.f31 * m2.f13) + (m1.f32 * m2.f23) + (m1.f33 * m2.f33) + (m1.f34 * m2.f43);
	tmp.f34 = (m1.f31 * m2.f14) + (m1.f32 * m2.f24) + (m1.f33 * m2.f34) + (m1.f34 * m2.f44);

	tmp.f11 = (m1.f41 * m2.f11) + (m1.f42 * m2.f21) + (m1.f43 * m2.f31) + (m1.f44 * m2.f41);
	tmp.f12 = (m1.f41 * m2.f12) + (m1.f42 * m2.f22) + (m1.f43 * m2.f32) + (m1.f44 * m2.f42);
	tmp.f13 = (m1.f41 * m2.f13) + (m1.f42 * m2.f23) + (m1.f43 * m2.f33) + (m1.f44 * m2.f43);
	tmp.f14 = (m1.f41 * m2.f14) + (m1.f42 * m2.f24) + (m1.f43 * m2.f34) + (m1.f44 * m2.f44);

	mdest = tmp;
}

Void MatrixRotateX(RMatrix4x4 matrix, Float m_XAngle)
{
    Float sx, cx;

	sx = SinCos(m_XAngle, &cx);

	matrix.f12 = (matrix.f12 * cx) + (matrix.f13 * sx);
	matrix.f22 = (matrix.f22 * cx) + (matrix.f23 * sx);
	matrix.f32 = (matrix.f32 * cx) + (matrix.f33 * sx);
	matrix.f42 = (matrix.f42 * cx) + (matrix.f43 * sx);

	matrix.f13 = (matrix.f13 * cx) - (matrix.f12 * sx);
	matrix.f23 = (matrix.f23 * cx) - (matrix.f22 * sx);
	matrix.f33 = (matrix.f33 * cx) - (matrix.f32 * sx);
	matrix.f43 = (matrix.f43 * cx) - (matrix.f42 * sx);
}

Void MatrixRotateY(RMatrix4x4 matrix, Float m_YAngle)
{
    Float sy, cy;

	sy = SinCos(m_YAngle, &cy);

	matrix.f11 = (matrix.f11 * cy) - (matrix.f13 * sy);
	matrix.f21 = (matrix.f21 * cy) - (matrix.f23 * sy);
	matrix.f31 = (matrix.f31 * cy) - (matrix.f33 * sy);
	matrix.f41 = (matrix.f41 * cy) - (matrix.f43 * sy);

	matrix.f13 = (matrix.f13 * cy) + (matrix.f11 * sy);
	matrix.f23 = (matrix.f23 * cy) + (matrix.f21 * sy);
	matrix.f33 = (matrix.f33 * cy) + (matrix.f31 * sy);
	matrix.f43 = (matrix.f43 * cy) + (matrix.f41 * sy);
}

Void MatrixRotateZ(RMatrix4x4 matrix, Float m_ZAngle)
{
    Float sz, cz;

	sz = SinCos(m_ZAngle, &cz);

	matrix.f11 = (matrix.f11 * cz) + (matrix.f12 * sz);
	matrix.f21 = (matrix.f21 * cz) + (matrix.f22 * sz);
	matrix.f31 = (matrix.f31 * cz) + (matrix.f32 * sz);
	matrix.f41 = (matrix.f41 * cz) + (matrix.f42 * sz);

	matrix.f12 = (matrix.f12 * cz) - (matrix.f11 * sz);
	matrix.f22 = (matrix.f22 * cz) - (matrix.f21 * sz);
	matrix.f32 = (matrix.f32 * cz) - (matrix.f31 * sz);
	matrix.f42 = (matrix.f42 * cz) - (matrix.f41 * sz);
}

Void MatrixRotateXYZ(RMatrix4x4 matrix, Float m_XAngle, Float m_YAngle, Float m_ZAngle)
{
	Matrix4x4 mrot;

	MatrixSetXYZRotation(mrot, m_XAngle, m_YAngle, m_ZAngle);
	MatrixMultiply(matrix, mrot, matrix);
}

Void MatrixGetXYZRotationAngles(RMatrix4x4 matrix, RVector3 vAngles)
{
    vAngles.m_Y = asinf(matrix.f13);
	
	if (vAngles.m_Y < ((Float) CONST_PI_OVER_2))
	{
        if (vAngles.m_Y > ((Float) -CONST_PI_OVER_2))
		{
			vAngles.m_X = atan2f(-matrix.f23, matrix.f33);
			vAngles.m_Z = atan2f(-matrix.f12, matrix.f11);
		}
		else
		{
            vAngles.m_X = -atan2f(matrix.f21, matrix.f22);
			vAngles.m_Z = 0.0f;
		}
	}
	else
	{
		vAngles.m_X = atan2f(matrix.f21, matrix.f22);
		vAngles.m_Z = 0.0f;
	}
}

Void MatrixRotateVectorArray(RMatrix4x4 matrix, PVector3 vIn, PVector3 vOut, UInt uiNumVecs)
{
    while (uiNumVecs--)
	{
		MatrixRotateVector(matrix, *vIn, *vOut);
		vIn++;
		vOut++;
	}
}

Void MatrixTransformVectorArray(RMatrix4x4 matrix, PVector3 vIn, PVector3 vOut, UInt uiNumVecs)
{
    while (uiNumVecs--)
	{
		MatrixTransformVector(matrix, *vIn, *vOut);
		vIn++;
		vOut++;
	}
}


Void MatrixProjectVectorArray(RMatrix4x4 matrix, PVector3 vIn, PVector3 vOut, UInt uiNumVecs)
{
    while (uiNumVecs--)
	{
		MatrixProjectVector(matrix, *vIn, *vOut);
		vIn++;
		vOut++;
	}
}
#ifndef RTPS_STRUCTS_H_INCLUDED
#define RTPS_STRUCTS_H_INCLUDED

#include <stdio.h>
#include <math.h>


// GE: Sept. 8, 2010
// Coded as float4 since OpenCL does not have float3
typedef struct float3 {
	float x, y, z;
	float w;
	float3() {}
	float3(float x, float y, float z) {
		this->x = x;
		this->y = y;
		this->z = z;
		this->w = 1.;
	}
} float3;

// GE: Sept. 8, 2010
// Coded as int4 since OpenCL does not have int3
typedef struct int4 {
	int x, y, z;
	int w;
	int4() {}
	int4(float x, float y, float z, float w=1.) {
		this->x = x;
		this->y = y;
		this->z = z;
		this->w = w;
	}
	int4(int x, int y, int z, int w=1) {
		this->x = x;
		this->y = y;
		this->z = z;
		this->w = w;
	}
} int4;

// IJ
typedef struct float4
{
    float x;
    float y;
    float z;
    float w;

    float4(){};
    float4(float xx, float yy, float zz, float ww):
        x(xx),
        y(yy),
        z(zz),
        w(ww)
    {}
	void set(float xx, float yy, float zz, float ww=1.) {
		x = xx;
		y = yy;
		z = zz;
		w = ww;
	}
	void print(const char* msg=0) {
		printf("%s: %f, %f, %f, %f\n", x, y, z, w);
	}

	friend float4 operator-(float4& a, float4& b) {
		float4 c = float4(b.x-a.x, b.y-a.y, b.z-a.z, b.w-a.w);
		return c;
	}

	friend float4& operator+(float4& a, float4& b) {
		float4 c = float4(b.x+a.x, b.y+a.y, b.z+a.z, b.w+a.w);
		return c;
	}

	float4 operator+=(float4 a) {
		(*this).x += a.x;
		(*this).y += a.w;
		(*this).z += a.z;
		(*this).w += a.w;
	}

	//friend float4& operator+(const float4& a, const float4& b) const {
		//float4 c = float4(b.x+a.x, b.y+a.y, b.z+a.z, b.w+a.w);
		//return c;
	//}

	//friend float4 operator*(const float r, const float4& b) const {
		//return float4(r*b.x, r*b.y, r*b.z, r*b.w);
	//}
	//friend float4& operator*(const float4& b, const float r) const {
		//return float4(r*b.x, r*b.y, r*b.z, r*b.w);
	//}

	friend float4 operator*(float r, float4& b) {
		float4 m = float4(r*b.x, r*b.y, r*b.z, r*b.w);
		return m;
	}
	friend float4 operator*(float4& b, float r) {
		float4 m = float4(r*b.x, r*b.y, r*b.z, r*b.w);
		return m;
	}

	float length() {
		float4& f = *this;
		return sqrt(f.x*f.x + f.y*f.y + f.z*f.z);
	}
} float4;




#endif

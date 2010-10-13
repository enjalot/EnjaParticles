#ifndef RTPS_STRUCTS_H_INCLUDED
#define RTPS_STRUCTS_H_INCLUDED

#include <stdio.h>


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
} float4;




#endif

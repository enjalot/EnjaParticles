#ifndef RTPS_STRUCTS_H_INCLUDED
#define RTPS_STRUCTS_H_INCLUDED


typedef struct float3 {
    //we have to add 4th component to match how OpenCL does float3 on GPU
	float x, y, z, w;
	float3() {}
	float3(float x, float y, float z) {
		this->x = x;
		this->y = y;
		this->z = z;
	}
} float3;

// GE: Sept. 8, 2010
typedef struct int3 {
	int x, y, z;
	int3() {}
	int3(float x, float y, float z) {
		this->x = x;
		this->y = y;
		this->z = z;
	}
} int3;

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
} float4;




#endif

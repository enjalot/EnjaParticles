#ifndef RTPS_STRUCTS_H_INCLUDED
#define RTPS_STRUCTS_H_INCLUDED

#include <math.h>

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
	int x, y, z, w;
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

    void print(const char* msg=0) {
        //printf("%s: %e, %e, %e, %f\n", msg, x, y, z, w);
    }

    friend float4 operator-(float4& a, float4& b) {
        float4 c = float4(a.x-b.x, a.y-b.y, a.z-b.z, a.w-b.w);
        return c;
    }

    // to do: float4 aa = min - float4(5.,5.,5.,5.); // min is float4
    friend const float4 operator-(const float4& a, const float4& b) {
        float4 c = float4(a.x-b.x, a.y-b.y, a.z-b.z, a.w-b.w);
        return c;
    }

    friend float4 operator+(float4& a, float4& b) {
        float4 c = float4(b.x+a.x, b.y+a.y, b.z+a.z, b.w+a.w);
        return c;
    }

    friend const float4 operator+(const float4& a, const float4& b) {
        float4 c = float4(b.x+a.x, b.y+a.y, b.z+a.z, b.w+a.w);
        return c;
    }

    void operator+=(float4 a) {
        (*this).x += a.x;
        (*this).y += a.y;
        (*this).z += a.z;
        (*this).w += a.w;
    }

    friend float4 operator*(float r, float4& b) {
        float4 m = float4(r*b.x, r*b.y, r*b.z, r*b.w);
        return m;
    }
    friend float4 operator*(float4& b, float r) {
        float4 m = float4(r*b.x, r*b.y, r*b.z, r*b.w);
        return m;
    }

    friend float4 operator/(float4& b, float r) {
        float d = 1./r;
        float4 m = float4(d*b.x, d*b.y, d*b.z, d*b.w);
        return m;
    }

    float length() {
        float4& f = *this;
        return sqrt(f.x*f.x + f.y*f.y + f.z*f.z);
    }


} float4;


//maybe these helper functions should go elsewhere? 
//or be functions of the structs
float magnitude(float4 vec);
float dist_squared(float4 vec);



#endif
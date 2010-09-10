//from http://www.songho.ca/opengl/gl_vbo.html

#ifndef ENJA_UTIL_H_INCLUDED
#define ENJA_UTIL_H_INCLUDED


// GE: Sept. 8, 2010
typedef struct float3 {
	float x, y, z;
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




char *file_contents(const char *filename, int *length);

GLuint createVBO(const void* data, int dataSize, GLenum target, GLenum usage);
int deleteVBO(GLuint id);

cl_int oclGetPlatformID(cl_platform_id* clSelectedPlatformID);
const char* oclErrorString(cl_int error);


#endif

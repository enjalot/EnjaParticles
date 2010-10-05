//from http://www.songho.ca/opengl/gl_vbo.html

#ifndef RTPS_UTIL_H_INCLUDED
#define RTPS_UTIL_H_INCLUDED

//OpenCL API
//#include "opencl/CLL.h"

#include "structs.h"

char *file_contents(const char *filename, int *length);


GLuint createVBO(const void* data, int dataSize, GLenum target, GLenum usage);
GLuint registerVBO();
int deleteVBO(GLuint id);

cl_int oclGetPlatformID(cl_platform_id* clSelectedPlatformID);

float rand_float(float mn, float mx);


#endif

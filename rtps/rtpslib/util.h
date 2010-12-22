//from http://www.songho.ca/opengl/gl_vbo.html

#ifndef RTPS_UTIL_H_INCLUDED
#define RTPS_UTIL_H_INCLUDED

#include <GL/glew.h>
#include "structs.h"

namespace rtps
{

char *file_contents(const char *filename, int *length);

GLuint createVBO(const void* data, int dataSize, GLenum target, GLenum usage);
int deleteVBO(GLuint id);

float distance(float4 p1, float4 p2);
float length(float4 v);

}

#endif

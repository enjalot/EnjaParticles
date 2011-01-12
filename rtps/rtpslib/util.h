//from http://www.songho.ca/opengl/gl_vbo.html

#ifndef RTPS_UTIL_H_INCLUDED
#define RTPS_UTIL_H_INCLUDED

#include "structs.h"

char *file_contents(const char *filename, int *length);

GLuint createVBO(const void* data, int dataSize, GLenum target, GLenum usage);
int deleteVBO(GLuint id);


#endif

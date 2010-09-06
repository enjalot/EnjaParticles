//from http://www.songho.ca/opengl/gl_vbo.html

#ifndef ENJA_UTIL_H_INCLUDED
#define ENJA_UTIL_H_INCLUDED

char *file_contents(const char *filename, int *length);

GLuint createVBO(const void* data, int dataSize, GLenum target, GLenum usage);
int deleteVBO(GLuint id);

cl_int oclGetPlatformID(cl_platform_id* clSelectedPlatformID);
const char* oclErrorString(cl_int error);


#endif

#ifndef __GLINCLUDES__
#define __GLINCLUDES__

#if defined (__APPLE_CC__)
	//#include <GL/glew.h>
	//#include <Cg/cg.h>
	//#include <Cg/cgGL.h>
	//#include <OpenGL/glext.h> // included by glew
	#define GL_GLEXT_PROTOTYPES 1
	#include <GLUT/glut.h>
	//#define FLOAT_BUFFER GL_RGBA32F_ARB
	//#define FLOAT_BUFFER GL_RGBA16F_ARB
	//#define FLOAT_BUFFER GL_RGBA
#else	
	#include <GL/glew.h>
	#include <GL/gl.h>
	#include <GL/glu.h>
	#include <GL/glext.h>
	#include <GL/glut.h>
#endif


#define FLOAT_BUFFER  GL_RGBA32F_ARB
#define FLOAT32_4  GL_RGBA32F_ARB
#define FLOAT32_3  GL_RGB32F_ARB
#define FLOAT32_2  GL_LUMINANCE_ALPHA32F_ARB
#define FLOAT32_1  GL_ALPHA32F_ARB

#define FLOAT16_4  GL_RGBA16F_ARB
#define FLOAT16_3  GL_RGB16F_ARB
#define FLOAT16_2  GL_LUMINANCE_ALPHA16F_ARB
#define FLOAT16_1  GL_ALPHA16F_ARB

// WHAT do these do?
//#define FLOAT32_1  GL_INTENSITY32F_ARB
//#define FLOAT32_1  GL_LUMINANCE32F_ARB

#endif

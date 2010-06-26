#ifndef ENJA_INCOPENCL_H_INCLUDED
#define ENJA_INCOPENCL_H_INCLUDED


#include <GL/glew.h>
#if defined __APPLE__ || defined(MACOSX)
    //OpenGL stuff
    #include <OpenGL/gl.h>
    #include <OpenGL/glext.h>
//    #include <GLUT/glut.h>
    #include <OpenGL/CGLCurrent.h> //is this really necessary?
    //OpenCL stuff
    #include <OpenCL/opencl.h>
    #include <OpenCL/cl_gl.h>
    #include <OpenCL/cl_gl_ext.h>
    #define GL_SHARING_EXTENSION "cl_APPLE_gl_sharing"
#else
    //OpenGL stuff
    #include <GL/glx.h>
//    #include <GL/glut.h>
    //OpenCL stuff
    #include <CL/opencl.h>
    #include <CL/cl_gl.h>
    #include <CL/cl_gl_ext.h>
    #define GL_SHARING_EXTENSION "cl_khr_gl_sharing"
#endif

#endif

/****************************************************************************************
* Real-Time Particle System - An OpenCL based Particle system developed to run on modern GPUs. Includes SPH fluid simulations.
* version 1.0, September 14th 2011
* 
* Copyright (C) 2011 Ian Johnson, Andrew Young, Gordon Erlebacher, Myrna Merced, Evan Bollig
* 
* This software is provided 'as-is', without any express or implied
* warranty.  In no event will the authors be held liable for any damages
* arising from the use of this software.
* 
* Permission is granted to anyone to use this software for any purpose,
* including commercial applications, and to alter it and redistribute it
* freely, subject to the following restrictions:
* 
* 1. The origin of this software must not be misrepresented; you must not
* claim that you wrote the original software. If you use this software
* in a product, an acknowledgment in the product documentation would be
* appreciated but is not required.
* 2. Altered source versions must be plainly marked as such, and must not be
* misrepresented as being the original software.
* 3. This notice may not be removed or altered from any source distribution.
****************************************************************************************/


//from http://www.songho.ca/opengl/gl_vbo.html

#ifndef RTPS_UTIL_H_INCLUDED
#define RTPS_UTIL_H_INCLUDED

#include "structs.h"
#include <vector>

#ifdef WIN32
    #if defined(rtps_EXPORTS)
        #define RTPS_EXPORT __declspec(dllexport)
    #else
        #define RTPS_EXPORT __declspec(dllimport)
	#endif 
#else
    #define RTPS_EXPORT
#endif

#include <opencl/Buffer.h>

namespace rtps
{

    char RTPS_EXPORT *file_contents(const char *filename, int *length);

    GLuint RTPS_EXPORT createVBO(const void* data, int dataSize, GLenum target, GLenum usage);
    int RTPS_EXPORT deleteVBO(GLuint id);


    void RTPS_EXPORT make_cube(std::vector<Triangle> &triangles, float4 center, float half_edge);


class Utils
{
public:
	void printDevArray(Buffer<unsigned int>& cl_array, char* msg, int nb_el, int nb_print);
	void printDevArray(Buffer<int>& cl_array, char* msg, int nb_el, int nb_print);
	void printDevArray(Buffer<int4>& cl_array, char* msg, int nb_el, int nb_print);
	void printDevArray(Buffer<float>& cl_array, char* msg, int nb_el, int nb_print);
	void printDevArray(Buffer<float4>& cl_array, char* msg, int nb_el, int nb_print);
};

}

#endif

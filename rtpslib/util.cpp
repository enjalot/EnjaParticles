#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>

#include <GL/glew.h>
#include <CL/cl.hpp>

#include "util.h"

namespace rtps
{

//----------------------------------------------------------------------
void Utils::printDevArray(Buffer<int4>& cl_array, char* msg, int nb_el, int nb_print)
{
	std::vector<int4> pos(nb_el);
	cl_array.copyToHost(pos);
	printf("*** %s ***\n", msg);
	for (int i=0; i < nb_print; i++) {
		printf("i= %d: ", i);
		pos[i].print(msg);
	}
}
//----------------------------------------------------------------------
void Utils::printDevArray(Buffer<int>& cl_array, char* msg, int nb_el, int nb_print)
{
	std::vector<int> pos(nb_el);
	cl_array.copyToHost(pos);
	printf("*** %s ***\n", msg);
	for (int i=0; i < nb_print; i++) {
		printf("%s[%d]: %d ", msg, i, pos[i]);
	}
}
//----------------------------------------------------------------------
void Utils::printDevArray(Buffer<float>& cl_array, char* msg, int nb_el, int nb_print)
{
	std::vector<float> pos(nb_el);
	cl_array.copyToHost(pos);
	printf("*** %s ***\n", msg);
	for (int i=0; i < nb_print; i++) {
		printf("%s[%d]: %f ", msg, i, pos[i]);
	}
}
//----------------------------------------------------------------------
void Utils::printDevArray(Buffer<float4>& cl_array, char* msg, int nb_el, int nb_print)
{
	std::vector<float4> pos(nb_el);
	cl_array.copyToHost(pos);
	printf("*** %s ***\n", msg);
	for (int i=0; i < nb_print; i++) {
		printf("i= %d: ", i);
		pos[i].print(msg);
	}
}
//----------------------------------------------------------------------
//----------------------------------------------------------------------
//----------------------------------------------------------------------
//----------------------------------------------------------------------
//----------------------------------------------------------------------



	//----------------------------------------------------------------------
    char *file_contents(const char *filename, int *length)
    {
        FILE *f = fopen(filename, "r");
        void *buffer;

        if (!f)
        {
            fprintf(stderr, "Unable to open %s for reading\n", filename);
            return NULL;
        }

        fseek(f, 0, SEEK_END);
        *length = ftell(f);
        fseek(f, 0, SEEK_SET);

        buffer = malloc(*length+1);
        *length = fread(buffer, 1, *length, f);
        fclose(f);
        ((char*)buffer)[*length] = '\0';

        return(char*)buffer;
    }

	//----------------------------------------------------------------------
    int deleteVBO(GLuint id)
    {
        glBindBuffer(1, id);
        glDeleteBuffers(1, (GLuint*)&id);
        return 1; //success
    }

	//----------------------------------------------------------------------
    GLuint createVBO(const void* data, int dataSize, GLenum target, GLenum usage)
    {
        GLuint id = 0;  // 0 is reserved, glGenBuffersARB() will return non-zero id if success

        glGenBuffers(1, &id);                        // create a vbo
        glBindBuffer(target, id);                    // activate vbo id to use
        glBufferData(target, dataSize, data, usage); // upload data to video card

        // check data size in VBO is same as input array, if not return 0 and delete VBO
        int bufferSize = 0;
        glGetBufferParameteriv(target, GL_BUFFER_SIZE, &bufferSize);
        if (dataSize != bufferSize)
        {
            glDeleteBuffers(1, &id);
            id = 0;
            //cout << "[createVBO()] Data size is mismatch with input array\n";
            printf("[createVB()] Data size is mismatch with input array\n");
        }
        //this was important for working inside blender!
        glBindBuffer(target, 0);

        return id;      // return VBO id
    }

	//----------------------------------------------------------------------
    void make_cube(std::vector<Triangle> &triangles, float4 cen, float half_edge)
    {
        // Written by G. Erlebacher Aug. 5, 2010
        /*
        
                7-----------6 
               /           /|
              /           / |           Z
             4-----------5  |           |
             |           |  2           |  Y
             |           | /            | /
             |           |/             |/
             0-----------1              x------- X
                          
        */
        printf("inside make_cube\n");
        // vertices
        std::vector<float4> v;
        float h = half_edge;
        float4 vv;

        vv.set(cen.x-h, cen.y-h, cen.z-h);
        v.push_back(vv);
        vv.set(cen.x+h, cen.y-h, cen.z-h);
        v.push_back(vv);
        vv.set(cen.x+h, cen.y+h, cen.z-h);
        v.push_back(vv);
        vv.set(cen.x-h, cen.y+h, cen.z-h);
        v.push_back(vv);
        vv.set(cen.x-h, cen.y-h, cen.z+h);
        v.push_back(vv);
        vv.set(cen.x+h, cen.y-h, cen.z+h);
        v.push_back(vv);
        vv.set(cen.x+h, cen.y+h, cen.z+h);
        v.push_back(vv);
        vv.set(cen.x-h, cen.y+h, cen.z+h);
        v.push_back(vv);

        // Triangles
        Triangle tri;

        tri.verts[0] = v[2];
        tri.verts[1] = v[1];
        tri.verts[2] = v[0];
        tri.normal.set(0.,0.,-1.,0.);
        triangles.push_back(tri);

        tri.verts[0] = v[3];
        tri.verts[1] = v[2];
        tri.verts[2] = v[0];
        tri.normal.set(0.,0.,-1.,0.);
        triangles.push_back(tri);
        //printf("triangles: size: %zd\n", triangles.size());

        tri.verts[0] = v[4];
        tri.verts[1] = v[5];
        tri.verts[2] = v[6];
        tri.normal.set(0.,0.,+1.,0.);
        triangles.push_back(tri);

        tri.verts[0] = v[4];
        tri.verts[1] = v[6];
        tri.verts[2] = v[7];
        tri.normal.set(0.,0.,+1.,0.);
        triangles.push_back(tri);

        //---
        tri.verts[0] = v[0];
        tri.verts[1] = v[1];
        tri.verts[2] = v[5];
        tri.normal.set(0.,-1.,0.,0.);
        triangles.push_back(tri);

        tri.verts[0] = v[0];
        tri.verts[1] = v[5];
        tri.verts[2] = v[4];
        tri.normal.set(0.,-1.,0.,0.);
        triangles.push_back(tri);

        tri.verts[0] = v[7];
        tri.verts[1] = v[6];
        tri.verts[2] = v[2];
        tri.normal.set(0.,+1.,0.,0.);
        triangles.push_back(tri);

        tri.verts[0] = v[7];
        tri.verts[1] = v[2];
        tri.verts[2] = v[3];
        tri.normal.set(0.,+1.,0.,0.);
        triangles.push_back(tri);

        //----
        tri.verts[0] = v[1];
        tri.verts[1] = v[2];
        tri.verts[2] = v[6];
        tri.normal.set(+1.,0.,0.,0.);
        triangles.push_back(tri);

        tri.verts[0] = v[1];
        tri.verts[1] = v[6];
        tri.verts[2] = v[5];
        tri.normal.set(+1.,0.,0.,0.);
        triangles.push_back(tri);

        tri.verts[0] = v[0];
        tri.verts[1] = v[4];
        tri.verts[2] = v[7];
        tri.normal.set(-1.,0.,0.,0.);
        triangles.push_back(tri);

        tri.verts[0] = v[0];
        tri.verts[1] = v[7];
        tri.verts[2] = v[3];
        tri.normal.set(-1.,0.,0.,0.);
        triangles.push_back(tri);
    }
	//----------------------------------------------------------------------
}

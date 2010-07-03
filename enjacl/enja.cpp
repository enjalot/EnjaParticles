#include <stdio.h>


#include <GL/glew.h>
#if defined __APPLE__ || defined(MACOSX)
    //OpenGL stuff
    #include <OpenGL/gl.h>
    #include <OpenGL/glext.h>
    #include <GLUT/glut.h>
    #include <OpenGL/CGLCurrent.h> //is this really necessary?
#else
    //OpenGL stuff
    #include <GL/glx.h>
#endif


#include "enja.h"
//#include "incopencl.h"
#include "util.h"


#include<math.h>
#include<stdlib.h>
#include<time.h>

int EnjaParticles::init(Vec4* g, Vec4* c, int n)
{
    num = n;
    generators = g;
    colors = c;

    //initialize our vbos
    vbo_size = sizeof(Vec4) * n;
    v_vbo = createVBO(generators, vbo_size, GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW);
    c_vbo = createVBO(colors, vbo_size, GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW);

    srand48(time(NULL));

    velocities = new Vec4[num];
    for(int i=0; i < n; i++)
    {
        velocities[i].x = .01 * (1. - 2.*drand48()); // between -.02 and .02
        velocities[i].y = .05 * drand48();
        velocities[i].z = .01 * (1. - 2.*drand48());
        velocities[i].w = 0.f;
    }

 
    life = new float[num];
    for(int i=0; i < n; i++)
    {
        life[i] = drand48();
    }


    return 1;
}


EnjaParticles::EnjaParticles()
{
    printf("default constructor\n");
    made_default = true; //need to remember to delete our allocated arrays
    //init system
    int n = 500000;
    Vec4* g = new Vec4[n];

    float f = 0;
    for(int i=0; i < n; i++)
    {
        f = (float)i;
        g[i].x = 0.0 + 500*cos(2.*M_PI*(f/n));  //with lorentz this looks more interesting
        //g[i].x = 1.0f;
        //g[i].y = 0.0 + .05*sin(2.*M_PI*(f/n));
        //g[i].y = -1.0f;
        g[i].z = 0.f;
        g[i].y = 0.0 + 500*sin(2.*M_PI*(f/n));
        //g[i].z = 0.0f;
        //g[i].z = 0.f;// + f/nums;
        g[i].w = 1.f;
    }

    Vec4* c = new Vec4[n];
    for(int i=0; i < n; i++)
    {
        c[i].x = 1.0;   //Red
        c[i].y = 0.0;   //Green
        c[i].z = 0.0;   //Blue
        c[i].w = 1.0;   //Alpha
    }

    printf("before init call\n");

    //init particle system
    init(g, c, n);

    printf("before opencl call\n");

    //init opencl
    int success = init_cl();
    
}

EnjaParticles::EnjaParticles(Vec4* g, int n)
{
    made_default = false;

    Vec4* c = new Vec4[n];
    for(int i=0; i < n; i++)
    {
        c[i].x = 1.0;   //Red
        c[i].y = 0.0;   //Green
        c[i].z = 0.0;   //Blue
        c[i].w = 1.0;   //Alpha
    }

    //init particle system
    init(g, c, n);

    //init opencl
    int success = init_cl();
}

EnjaParticles::EnjaParticles(Vec4* g, Vec4* c, int n)
{
    made_default = false;
    //init particle system
    init(g, c, n);

    //init opencl
    int success = init_cl();
}

EnjaParticles::~EnjaParticles()
{

    printf("Release!\n");
    ts_cl[0]->print();
    ts_cl[1]->print();
    ts_cl[2]->print();
    delete ts_cl[0];
    delete ts_cl[1];
    delete ts_cl[2];
    //should probably just make a copy of passed in generator
    //and always clean things up
    if (made_default)
    {
        delete generators;
        delete colors;
    }

    delete velocities;
    delete life;

    if(ckKernel)clReleaseKernel(ckKernel); 
    if(cpProgram)clReleaseProgram(cpProgram);
    if(cqCommandQueue)clReleaseCommandQueue(cqCommandQueue);
    if(v_vbo)
    {
        glBindBuffer(1, v_vbo);
        glDeleteBuffers(1, (GLuint*)&v_vbo);
        v_vbo = 0;
    }
    if(c_vbo)
    {
        glBindBuffer(1, c_vbo);
        glDeleteBuffers(1, (GLuint*)&c_vbo);
        c_vbo = 0;
    }
    //if(vbo_cl)clReleaseMemObject(vbo_cl);
    if(cl_vbos[0])clReleaseMemObject(cl_vbos[0]);
    if(cl_vbos[1])clReleaseMemObject(cl_vbos[1]);
    if(cl_generators)clReleaseMemObject(cl_generators);
    if(cl_velocities)clReleaseMemObject(cl_velocities);
    if(cl_life)clReleaseMemObject(cl_life);
    if(cxGPUContext)clReleaseContext(cxGPUContext);
    
    if(cdDevices)delete(cdDevices);

}


int EnjaParticles::getVertexVBO()
{
    return v_vbo;
}

int EnjaParticles::getColorVBO()
{
    return c_vbo;
}


int EnjaParticles::getNum()
{
    return num;
}

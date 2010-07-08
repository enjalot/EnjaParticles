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
#include <sstream>
#include <iomanip>


//the paths to these programs are relative to the source dir
//this is used in init_cl
const std::string EnjaParticles::programs[] = {
    "/physics/lorentz.cl",
    "/physics/gravity.cl"
};



int EnjaParticles::init(Vec4* g, Vec4* c, int n)
{
    // This is the main initialization function for our particle systems
    // Vec4* g is the array of generator points (this initializes our system)
    // Vec4* c is the array of color values 
    num = n;
    generators = g;
    colors = c;

    //initialize our vbos
    vbo_size = sizeof(Vec4) * n;
    v_vbo = createVBO(generators, vbo_size, GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW);
    c_vbo = createVBO(colors, vbo_size, GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW);

    srand48(time(NULL));

    //initialize the velocities array, by default we just set to 0
    velocities = new Vec4[num];
    for(int i=0; i < n; i++)
    {
        velocities[i].x = 0.f; //.01 * (1. - 2.*drand48()); // between -.02 and .02
        velocities[i].y = 0.f; //.05 * drand48();
        velocities[i].z = 0.f; //.01 * (1. - 2.*drand48());
        velocities[i].w = 0.f;
    }

 
    //initialize the particle life array with random values between 0 and 1
    life = new float[num];
    for(int i=0; i < n; i++)
    {
        life[i] = drand48();
    }

    //we initialize our timers, they only time every 5th call
    ts[0] = new GE::Time("update", 5);
    ts[1] = new GE::Time("render", 5);
    ts[2] = new GE::Time("total render", 5);


    return 1;
}


EnjaParticles::EnjaParticles(int s, int n)
{
    printf("default constructor\n");
    system = s;
    //init system
    Vec4* g = new Vec4[n];

    float f = 0;
    for(int i=0; i < n; i++)
    {
        f = (float)i;
        g[i].x = 0.0 + 10*cos(2.*M_PI*(f/n));  //with lorentz this looks more interesting
        //g[i].x = 1.0f;
        //g[i].y = 0.0 + .05*sin(2.*M_PI*(f/n));
        //g[i].y = -1.0f;
        g[i].z = 0.f;
        g[i].y = 0.0 + 10*sin(2.*M_PI*(f/n));
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

EnjaParticles::EnjaParticles(int s, Vec4* g, int n)
{
    system = s;

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

EnjaParticles::EnjaParticles(int s, Vec4* g, Vec4* c, int n)
{
    system = s;
    //init particle system
    init(g, c, n);

    //init opencl
    int success = init_cl();
}

EnjaParticles::~EnjaParticles()
{

    printf("Release! num=%d\n", num);
    ts[0]->print();
    ts[1]->print();
    ts[2]->print();
    delete ts[0];
    delete ts[1];
    delete ts[2];

    ts_cl[0]->print();
    ts_cl[1]->print();
    ts_cl[2]->print();
    ts_cl[3]->print();
    delete ts_cl[0];
    delete ts_cl[1];
    delete ts_cl[2];
    delete ts_cl[3];
    
    delete generators;
    delete colors;

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

float EnjaParticles::getFPS()
{
    return 1000.f / ts[2]->getAverage();    //1 second divided by total render time 
}

std::string* EnjaParticles::getReport()
{
    std::stringstream ss1;
    std::stringstream ss2;
    std::string* s = new std::string[2];
    ss1 << std::fixed << std::setprecision(6);
    ss1 << "Average Render Time (per frame): " << ts[2]->getAverage() << std::ends;
    s[0] = ss1.str();
    ss2 << std::fixed << std::setprecision(6);
    ss2 << "Average OpenCL Time (per frame): " << ts_cl[0]->getAverage() + ts_cl[1]->getAverage() + ts_cl[2]->getAverage() << std::ends;
    s[1] = ss2.str();
    return s;

}


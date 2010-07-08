#ifndef ENJA_PARTICLES_H_INCLUDED
#define ENJA_PARTICLES_H_INCLUDED

#include <string>
#include "incopencl.h"
#include "timege.h"


typedef struct
{
    float x;
    float y;
    float z;
    float w;
} Vec4;


class EnjaParticles
{

public:

    int update(float dt);   //update the particle system
    int render(float dt, int type); //render calls update then renders the particles

    int getVertexVBO(); //get the vertices vbo id
    int getColorVBO(); //get the color vbo id
    int getNum(); //get the number of particles

    float getFPS(); //get the calculated frames per second for entire rendering process
    std::string* getReport(); //get string report of timings

    //constructors: will probably have more as more options are added
    EnjaParticles(int system, int num);
    EnjaParticles(int system, Vec4* generators, int num);
    EnjaParticles(int system, Vec4* generators, Vec4* colors, int num);

    ~EnjaParticles();

    enum {LORENTZ, GRAVITY};
    static const std::string programs[];

private:
    //particles
    int num;            //number of particles
    int system;         //what kind of system?
    Vec4* generators;
    Vec4* colors;
    Vec4* velocities;
    float* life;

    int init(Vec4* generators, Vec4* colors, int num);

    
    //opencl
    cl_platform_id cpPlatform;
    cl_context cxGPUContext;
    cl_device_id* cdDevices;
    cl_uint uiDevCount;
    unsigned int uiDeviceUsed;
    cl_command_queue cqCommandQueue;
    cl_kernel ckKernel;
    cl_program cpProgram;
    cl_int ciErrNum;
    size_t szGlobalWorkSize[1];

    //cl_mem vbo_cl;
    cl_mem cl_vbos[2];  //0: vertex vbo, 1: color vbo
    cl_mem cl_generators;  //want to have the start points for reseting particles
    cl_mem cl_velocities;  //particle velocities
    cl_mem cl_life;        //keep track where in their life the particles are
    int v_vbo;   //vertices vbo
    int c_vbo;   //colors vbo
    unsigned int vbo_size; //size in bytes of the vbo

    //timers
    GE::Time *ts[3];    //library timers (update, render, total)
    GE::Time *ts_cl[4]; //opencl timers (acquire, kernel exec, release)


    int init_cl();
    int setup_cl(); //helper function that initializes the devices and the context
    void popCorn(); // sets up the kernel and pushes data
    
    bool made_default;

};




#endif

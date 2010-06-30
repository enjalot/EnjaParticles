#ifndef ENJA_PARTICLES_H_INCLUDED
#define ENJA_PARTICLES_H_INCLUDED

#include "incopencl.h"

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

    int update(float dt);
    int getVertexVBO(); //get the vertices vbo id
    int getColorVBO(); //get the color vbo id
    int getNum(); //get the number of particles

    //constructors: will probably have more as more options are added
    EnjaParticles();
    EnjaParticles(Vec4* generators, int num);
    EnjaParticles(Vec4* generators, Vec4* colors, int num);

    ~EnjaParticles();

private:
    //particles
    int num;
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


    int init_cl();
    void popCorn(); //purely convenient function to make init_cl shorter
    
    bool made_default;

};

#endif

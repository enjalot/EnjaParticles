#ifndef RTPS_CL_H_INCLUDED
#define RTPS_CL_H_INCLUDED

#define __CL_ENABLE_EXCEPTIONS

#include <CL/cl.hpp>

namespace rtps{

    
const char* oclErrorString(cl_int error);
    
class CL
{
public:
    CL();
/*
    std::vector<Buffer> buffers;
    std::vector<Program> programs;
    std::vector<Kernel> kernels;

    int addBuffer(Buffer buff);
    int addProgram(Program prog);
    int addKernel(Kernel kern);
*/

    cl::Context context;
    cl::CommandQueue queue;

    std::vector<cl::Device> devices;
    int deviceUsed;

    //error checking stuff
    int err;
    cl::Event event;

    //setup an OpenCL context that shares with OpenGL
    void setup_gl_cl();

    cl::Program loadProgram(std::string path);
    cl::Kernel loadKernel(std::string path, std::string name);

    //TODO add oclErrorString to the class
    //move from util.h/cpp
};







}

#endif


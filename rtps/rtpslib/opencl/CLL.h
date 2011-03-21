#ifndef RTPS_CL_H_INCLUDED
#define RTPS_CL_H_INCLUDED

#define __CL_ENABLE_EXCEPTIONS

#include <CL/cl.hpp>

namespace rtps
{


    //NVIDIA helper functions    
    const char* oclErrorString(cl_int error);
    cl_int oclGetPlatformID(cl_platform_id* clSelectedPlatformID);

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

        cl::Program loadProgram(std::string path, std::string options="");
        cl::Kernel loadKernel(std::string path, std::string name);
        cl::Kernel loadKernel(cl::Program program, std::string kernel_name);

        //TODO add oclErrorString to the class
        //move from util.h/cpp
    };







}

#endif


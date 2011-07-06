#ifndef RTPS_CL_H_INCLUDED
#define RTPS_CL_H_INCLUDED

#define __CL_ENABLE_EXCEPTIONS

#include <CL/cl.hpp>

#ifdef WIN32
    #if defined(rtps_EXPORTS)
        #define RTPS_EXPORT __declspec(dllexport)
    #else
        #define RTPS_EXPORT __declspec(dllimport)
	#endif 
#else
    #define RTPS_EXPORT
#endif

namespace rtps
{


    //NVIDIA helper functions    
    RTPS_EXPORT const char* oclErrorString(cl_int error);
    RTPS_EXPORT cl_int oclGetPlatformID(cl_platform_id* clSelectedPlatformID);

    class RTPS_EXPORT CL
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

        int err;
        cl::Event event;

        cl::Context context;
        std::vector<cl::Context> context_vec;
        cl::Context cpu_context;
        std::vector<cl::CommandQueue> queue;
        std::vector<cl::CommandQueue> cpu_queue;
        std::vector<cl::Device> devices;
        std::vector<cl::Device> cpu_devices;

        //setup an OpenCL context that shares with OpenGL
        void setup_gl_cl();

        cl::Program loadProgram(std::string path, std::string options="");
        cl::Kernel loadKernel(std::string path, std::string name);
        cl::Kernel loadKernel(cl::Program program, std::string kernel_name);

        //TODO make more general
        void addIncludeDir(std::string);

        //TODO add oclErrorString to the class
        //move from util.h/cpp
        //

    private:
        std::string inc_dir;
        //error checking stuff
        int deviceUsed;
    };
};
#endif

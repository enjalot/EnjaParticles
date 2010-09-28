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

	#if 1
    cl::Context context;
    cl::CommandQueue queue;
	//bool is_initialized;

    std::vector<cl::Device> devices;
    int deviceUsed;
	//cl::Device device; // = devices[deviceUsed]

	//bool profiling;
	#endif


	inline cl_context getRawContext() {
		return context();
	}

	inline cl_command_queue getRawQueue() {
		return queue();
	}

    //error checking stuff
    int err;
    cl::Event event;

    //setup an OpenCL context that shares with OpenGL
    void setup_gl_cl();

    cl::Program loadProgram(std::string source);
    cl::Kernel loadKernel(std::string name, std::string source);

    //TODO add oclErrorString to the class
    //move from util.h/cpp
};







}

#endif


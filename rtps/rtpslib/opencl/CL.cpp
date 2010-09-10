#include <stdio.h>
#include <iostream>

//#include <GL/glew.h>
#if defined __APPLE__ || defined(MACOSX)
    //OpenGL stuff
#else
    //OpenGL stuff
    #include <GL/glx.h>
#endif

//

#include "CL.h"
#include "../util.h"

namespace rtps {

CL::CL()
{
    setup_gl_cl();
}

//----------------------------------------------------------------------
cl::Program CL::loadProgram(std::string kernel_source)
{
     // Program Setup
    int pl;
    //printf("load the program\n");
    
    pl = kernel_source.size();
    //printf("kernel size: %d\n", pl);
    //printf("kernel: \n %s\n", kernel_source.c_str());
    cl::Program program;
    try
    {
        cl::Program::Sources source(1,
            std::make_pair(kernel_source.c_str(), pl));
    
        program = cl::Program(context, source);
    
    }
    catch (cl::Error er) {
		printf("loadProgram\n");
        printf("ERROR: %s(%s)\n", er.what(), oclErrorString(er.err()));
    }

    try
    {
        //err = program.build(devices, "-cl-nv-verbose");
        err = program.build(devices);
    }
    catch (cl::Error er) {
		printf("loadProgram::program.build\n");
		printf("source= %s\n", kernel_source.c_str());
        printf("ERROR: %s(%s)\n", er.what(), oclErrorString(er.err()));
        std::cout << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(devices.front()) << std::endl;
        std::cout << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(devices.front()) << std::endl;
        std::cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices.front()) << std::endl;
    } 
    return program;
}

//----------------------------------------------------------------------
cl::Kernel CL::loadKernel(std::string kernel_source, std::string kernel_name)
{
    cl::Program program;
    cl::Kernel kernel;
    try{
        program = loadProgram(kernel_source);
        kernel = cl::Kernel(program, kernel_name.c_str(), &err);
    }
    catch (cl::Error er) {
        printf("ERROR: %s(%s)\n", er.what(), oclErrorString(er.err()));
    }
    return kernel;
}

void CL::setup_gl_cl()
{
    std::vector<cl::Platform> platforms;
    err = cl::Platform::get(&platforms);
    printf("cl::Platform::get(): %s\n", oclErrorString(err));
    printf("platforms.size(): %d\n", platforms.size());

    deviceUsed = 0;
    err = platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);
    printf("getDevices: %s\n", oclErrorString(err));
    printf("devices.size(): %d\n", devices.size());
    //const char* s = devices[0].getInfo<CL_DEVICE_EXTENSIONS>().c_str();
    //printf("extensions: \n %s \n", s);
    int t = devices.front().getInfo<CL_DEVICE_TYPE>();
    printf("type: \n %d %d \n", t, CL_DEVICE_TYPE_GPU);
    
    //assume sharing for now, at some point we should implement a check
    //to make sure the devices can do context sharing
    

    // Define OS-specific context properties and create the OpenCL context
    //#if defined (__APPLE_CC__)
    #if defined (__APPLE__) || defined(MACOSX)
        CGLContextObj kCGLContext = CGLGetCurrentContext();
        CGLShareGroupObj kCGLShareGroup = CGLGetShareGroup(kCGLContext);
        cl_context_properties props[] =
        {
            CL_CONTEXT_PROPERTY_USE_CGL_SHAREGROUP_APPLE, (cl_context_properties)kCGLShareGroup,
            0
        };
        //this works
        //cl_context cxGPUContext = clCreateContext(props, 0, 0, NULL, NULL, &err);
        //these dont
        //cl_context cxGPUContext = clCreateContext(props, 1,(cl_device_id*)&devices.front(), NULL, NULL, &err);
        //cl_context cxGPUContext = clCreateContextFromType(props, CL_DEVICE_TYPE_GPU, NULL, NULL, &err);
        //printf("IS IT ERR???? %s\n", oclErrorString(err));
        try{
            context = cl::Context(props);   //had to edit line 1448 of cl.hpp to add this constructor
        }
        catch (cl::Error er) {
            printf("ERROR: %s(%s)\n", er.what(), oclErrorString(er.err()));
        }  
    #else
        #if defined WIN32 // Win32
            cl_context_properties props[] = 
            {
                CL_GL_CONTEXT_KHR, (cl_context_properties)wglGetCurrentContext(), 
                CL_WGL_HDC_KHR, (cl_context_properties)wglGetCurrentDC(), 
                CL_CONTEXT_PLATFORM, (cl_context_properties)(platforms[0])(),
                0
            };
            //cl_context cxGPUContext = clCreateContext(props, 1, &cdDevices[uiDeviceUsed], NULL, NULL, &err);
            try{
                context = cl::Context(CL_DEVICE_TYPE_GPU, props);
            }
            catch (cl::Error er) {
                printf("ERROR: %s(%s)\n", er.what(), oclErrorString(er.err()));
            }       
        #else
            cl_context_properties props[] = 
            {
                CL_GL_CONTEXT_KHR, (cl_context_properties)glXGetCurrentContext(), 
                CL_GLX_DISPLAY_KHR, (cl_context_properties)glXGetCurrentDisplay(), 
                CL_CONTEXT_PLATFORM, (cl_context_properties)(platforms[0])(),
                0
            };
            //cl_context cxGPUContext = clCreateContext(props, 1, &cdDevices[uiDeviceUsed], NULL, NULL, &err);
            try{
                context = cl::Context(CL_DEVICE_TYPE_GPU, props);
            }
            catch (cl::Error er) {
                printf("ERROR: %s(%s)\n", er.what(), oclErrorString(er.err()));
            } 
        #endif
    #endif
 
    //for some reason this properties works but props doesn't with c++ bindings
    //cl_context_properties properties[] =
    //    { CL_CONTEXT_PLATFORM, (cl_context_properties)(platforms[0])(), 0};

    /*
    try{
        context = cl::Context(CL_DEVICE_TYPE_GPU, props);
        //context = cl::Context(devices, props);
        //context = cl::Context(devices, props, NULL, NULL, &err);
        //printf("IS IT ERR222 ???? %s\n", oclErrorString(err));
        //context = cl::Context(CL_DEVICE_TYPE_GPU, props);
        //context = cl::Context(cxGPUContext);
    }
    catch (cl::Error er) {
        printf("ERROR: %s(%s)\n", er.what(), oclErrorString(er.err()));
    }
    */
    //devices = context.getInfo<CL_CONTEXT_DEVICES>();

    //create the command queue we will use to execute OpenCL commands
    ///command_queue = clCreateCommandQueue(context, devices[deviceUsed], 0, &err);
    try{
        queue = cl::CommandQueue(context, devices[deviceUsed], 0, &err);
    }
    catch (cl::Error er) {
        printf("ERROR: %s(%s)\n", er.what(), oclErrorString(er.err()));
    }
}


}

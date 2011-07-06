#include <stdio.h>
#include <iostream>

//#include <GL/glew.h>
#if defined __APPLE__ || defined(MACOSX)
//OpenGL stuff
#elif defined(WIN32)
#else
//OpenGL stuff
    #include <GL/glx.h>
#endif

//

#include "CLL.h"
#include "../util.h"

namespace rtps
{

    CL::CL()
    {
        inc_dir = "";
        setup_gl_cl();
    }

    void CL::addIncludeDir(std::string path)
    {
        this->inc_dir += " -I" + path;// + " -I./" + std::string(COMMON_CL_SOURCE_DIR);
    }

    //----------------------------------------------------------------------
    cl::Program CL::loadProgram(std::string path, std::string options)
    {
        // Program Setup

        int length;
        char* src = file_contents(path.c_str(), &length);
        std::string kernel_source(src);
        free(src);


        //printf("kernel size: %d\n", pl);
        //printf("kernel: \n %s\n", kernel_source.c_str());
        cl::Program program;
        try
        {
            cl::Program::Sources source(1,
                                        std::make_pair(kernel_source.c_str(), length));

            program = cl::Program(context, source);

        }
        catch (cl::Error er)
        {
            printf("loadProgram\n");
            printf("ERROR: %s(%s)\n", er.what(), oclErrorString(er.err()));
        }

        try
        {
            printf("build program\n");
            //#ifdef DEBUG
#if 0
            srand(time(NULL));
            int rnd = rand() % 200 + 100;
            char dbgoptions[100];
            //should really check for NVIDIA platform before doing this
            sprintf(dbgoptions, "%s -cl-nv-verbose -cl-nv-maxrregcount=%d", options.c_str(), rnd);
            //sprintf(options, "-D rand=%d -D DEBUG", rnd);
            err = program.build(devices, dbgoptions);
#else

            options += this->inc_dir;
            printf("OPTIONS: %s\n", options.c_str());
            

            err = program.build(devices, options.c_str());
#endif
        }
        catch (cl::Error er)
        {
            printf("loadProgram::program.build\n");
            printf("source= %s\n", kernel_source.c_str());
            printf("ERROR: %s(%s)\n", er.what(), oclErrorString(er.err()));
        }
        std::cout << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(devices.front()) << std::endl;
        std::cout << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(devices.front()) << std::endl;
        std::cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices.front()) << std::endl;

        return program;
    }

    //----------------------------------------------------------------------
    cl::Kernel CL::loadKernel(std::string path, std::string kernel_name)
    {
        cl::Program program;
        cl::Kernel kernel;
        try
        {
            program = loadProgram(path);
            kernel = cl::Kernel(program, kernel_name.c_str(), &err);
        }
        catch (cl::Error er)
        {
            printf("ERROR: %s(%s)\n", er.what(), oclErrorString(er.err()));
        }
        return kernel;
    }

    //----------------------------------------------------------------------
    cl::Kernel CL::loadKernel(cl::Program program, std::string kernel_name)
    {
        cl::Kernel kernel;
        try
        {
            kernel = cl::Kernel(program, kernel_name.c_str(), &err);
        }
        catch (cl::Error er)
        {
            printf("ERROR: %s(%s)\n", er.what(), oclErrorString(er.err()));
        }
        return kernel;
    }


    void CL::setup_gl_cl()
    {
        std::vector<cl::Platform> platforms;
        err = cl::Platform::get(&platforms);
        printf("cl::Platform::get(): %s\n", oclErrorString(err));
        printf("platforms.size(): %zd\n", platforms.size());

        for(int i = 0; i < platforms.size(); i++)
        {
            printf("platform name: %s\n",platforms[i].getInfo<CL_PLATFORM_NAME>().c_str());
            deviceUsed = 0;
            err = platforms[i].getDevices(CL_DEVICE_TYPE_GPU, &devices);
            printf("getDevices GPU: %s\n", oclErrorString(err));
            printf("devices.size(): %zd\n", devices.size());
            err = platforms[i].getDevices(CL_DEVICE_TYPE_CPU, &cpu_devices);
            printf("getDevices CPU: %s\n", oclErrorString(err));
            printf("cpu_devices.size(): %zd\n", cpu_devices.size());
            //const char* s = devices[0].getInfo<CL_DEVICE_EXTENSIONS>().c_str();
            //printf("extensions: \n %s \n", s);

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
                try
                {
                    context = cl::Context(props);   //had to edit line 1448 of cl.hpp to add this constructor
                }
                catch (cl::Error er)
                {
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
                try
                {
                    context = cl::Context(CL_DEVICE_TYPE_GPU, props);
                }
                catch (cl::Error er)
                {
                    printf("ERROR: %s(%s)\n", er.what(), oclErrorString(er.err()));
                }
        #else
                /*Display* dis = glXGetCurrentDisplay();
                GLXContext con = glXGetCurrentContext();
                printf("gl context 0x%08x\n",&con);
                printf("gl Current Display 0x%08x\n",dis);*/
                cl_context_properties props[] = 
                {
                    CL_GL_CONTEXT_KHR, (cl_context_properties)glXGetCurrentContext(), 
                    CL_GLX_DISPLAY_KHR, (cl_context_properties)glXGetCurrentDisplay(), 
                    CL_CONTEXT_PLATFORM, NULL,//(cl_context_properties)(platforms[i])(),
                    0
                };
                //cl_context cxGPUContext = clCreateContext(props, 1, &cdDevices[uiDeviceUsed], NULL, NULL, &err);
                //std::vector<cl::Device> devtmp(1);
                //devtmp[0] = devices.back();
                //devices.pop_back();
                #if 0
                try
                {
                    //context = cl::Context(CL_DEVICE_TYPE_GPU, props);
                    context = cl::Context(devices, props);
                    //context = cl::Context(devices);
                }
                catch (cl::Error er)
                {
                    printf("ERROR: %s(%s)%d Line number %d\n", er.what(), oclErrorString(er.err()),er.err(),__LINE__);
                }
                #else
                for(int i = 0; i<devices.size(); i++)
                {
                    std::vector<cl::Device> dev;
                    dev.push_back(devices[i]);
                    try
                    {
                        //context = cl::Context(CL_DEVICE_TYPE_GPU, props);
                        //context = cl::Context(devices, props);
                        context_vec.push_back(cl::Context(dev));
                    }
                    catch (cl::Error er)
                    {
                        printf("ERROR: %s(%s)%d Line number %d\n", er.what(), oclErrorString(er.err()),er.err(),__LINE__);
                    }
                    
                }
                //Just to make things work for now. Definitely need to rework the structure of this class.
                context = context_vec[0];
                #endif
                //For TESTING if I can create 1 context per device
                /*try
                {
                    //context = cl::Context(CL_DEVICE_TYPE_GPU, props);
                    //context2 = cl::Context(devtmp, props);
                    context2 = cl::Context(devices);
                }
                catch (cl::Error er)
                {
                    printf("ERROR: %s(%s)%d Line number %d\n", er.what(), oclErrorString(er.err()),er.err(),__LINE__);
                }
                devices.push_back(devtmp.back());*/
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
                printf("devices.size %d\n",devices.size());
            for(int j = 0; j < devices.size(); j++)
            {
                //int t = devices.front().getInfo<CL_DEVICE_TYPE>();
                int t = devices[j].getInfo<CL_DEVICE_TYPE>();
                printf("type: \n %d %d \n", t, CL_DEVICE_TYPE_GPU);
                printf("device name: %s\n", devices[j].getInfo<CL_DEVICE_NAME>().c_str());
                printf("device extensions: %s\n", devices[j].getInfo<CL_DEVICE_EXTENSIONS>().c_str());
        

                //devices = context.getInfo<CL_CONTEXT_DEVICES>();
        
                //create the command queue we will use to execute OpenCL commands
                ///command_queue = clCreateCommandQueue(context, devices[deviceUsed], 0, &err);
                cl_command_queue_properties cq_props = CL_QUEUE_PROFILING_ENABLE;
                try
                {
                    queue.push_back(cl::CommandQueue(context_vec[j], devices[j], cq_props, &err));
                }
                catch (cl::Error er)
                {
                    printf("ERROR: %s(%s)\n", er.what(), oclErrorString(er.err()));
                }
            }

        #if defined (__APPLE__) || defined(MACOSX)
                CGLContextObj kCGLContext = CGLGetCurrentContext();
                /*CGLShareGroupObj kCGLShareGroup = CGLGetShareGroup(kCGLContext);*/
                cl_context_properties props_cpu[] =
                {
                    0
                };
                try
                {
                    cpu_context = cl::Context(props_cpu);   //had to edit line 1448 of cl.hpp to add this constructor
                }
                catch (cl::Error er)
                {
                    printf("ERROR: %s(%s)\n", er.what(), oclErrorString(er.err()));
                }
        #else
        #if defined WIN32 // Win32
                cl_context_properties props_cpu[] = 
                {
                    /*CL_GL_CONTEXT_KHR, (cl_context_properties)wglGetCurrentContext(), 
                    CL_WGL_HDC_KHR, (cl_context_properties)wglGetCurrentDC(), */
                    CL_CONTEXT_PLATFORM, (cl_context_properties)(platforms[i])(),
                    0
                };
                try
                {
                    cpu_context = cl::Context(CL_DEVICE_TYPE_CPU, props_cpu);
                }
                catch (cl::Error er)
                {
                    printf("ERROR: %s(%s)\n", er.what(), oclErrorString(er.err()));
                }
        #else
                cl_context_properties props_cpu[] = 
                {
                    /*CL_GL_CONTEXT_KHR, (cl_context_properties)glXGetCurrentContext(), 
                    CL_GLX_DISPLAY_KHR, (cl_context_properties)glXGetCurrentDisplay(), */
                    CL_CONTEXT_PLATFORM, (cl_context_properties)(platforms[i])(),
                    0
                };
                //cl_context cxGPUContext = clCreateContext(props, 1, &cdDevices[uiDeviceUsed], NULL, NULL, &err);
                try
                {
                    cpu_context = cl::Context(cpu_devices);
                }
                catch (cl::Error er)
                {
                    printf("ERROR: %s(%s)\n", er.what(), oclErrorString(er.err()));
                }
        #endif
        #endif
        
            for(int j = 0; j < cpu_devices.size(); j++)
            {
                //int t = devices.front().getInfo<CL_DEVICE_TYPE>();
                int t = cpu_devices[j].getInfo<CL_DEVICE_TYPE>();
                printf("type: \n %d %d \n", t, CL_DEVICE_TYPE_CPU);
                printf("device name: %s\n", cpu_devices[j].getInfo<CL_DEVICE_NAME>().c_str());
                printf("device extensions: %s\n", cpu_devices[j].getInfo<CL_DEVICE_EXTENSIONS>().c_str());
        

                //devices = context.getInfo<CL_CONTEXT_DEVICES>();
        
                //create the command queue we will use to execute OpenCL commands
                ///command_queue = clCreateCommandQueue(context, devices[deviceUsed], 0, &err);
                cl_command_queue_properties cq_props = CL_QUEUE_PROFILING_ENABLE;
                try
                {
                    cpu_queue.push_back(cl::CommandQueue(cpu_context, cpu_devices[j], cq_props, &err));
                }
                catch (cl::Error er)
                {
                    printf("ERROR: %s(%s)\n", er.what(), oclErrorString(er.err()));
                }
            }
        }
    }






    // Helper function to get error string
    // From NVIDIA
    // *********************************************************************
    const char* oclErrorString(cl_int error)
    {
        static const char* errorString[] = {
            "CL_SUCCESS",
            "CL_DEVICE_NOT_FOUND",
            "CL_DEVICE_NOT_AVAILABLE",
            "CL_COMPILER_NOT_AVAILABLE",
            "CL_MEM_OBJECT_ALLOCATION_FAILURE",
            "CL_OUT_OF_RESOURCES",
            "CL_OUT_OF_HOST_MEMORY",
            "CL_PROFILING_INFO_NOT_AVAILABLE",
            "CL_MEM_COPY_OVERLAP",
            "CL_IMAGE_FORMAT_MISMATCH",
            "CL_IMAGE_FORMAT_NOT_SUPPORTED",
            "CL_BUILD_PROGRAM_FAILURE",
            "CL_MAP_FAILURE",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "CL_INVALID_VALUE",
            "CL_INVALID_DEVICE_TYPE",
            "CL_INVALID_PLATFORM",
            "CL_INVALID_DEVICE",
            "CL_INVALID_CONTEXT",
            "CL_INVALID_QUEUE_PROPERTIES",
            "CL_INVALID_COMMAND_QUEUE",
            "CL_INVALID_HOST_PTR",
            "CL_INVALID_MEM_OBJECT",
            "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR",
            "CL_INVALID_IMAGE_SIZE",
            "CL_INVALID_SAMPLER",
            "CL_INVALID_BINARY",
            "CL_INVALID_BUILD_OPTIONS",
            "CL_INVALID_PROGRAM",
            "CL_INVALID_PROGRAM_EXECUTABLE",
            "CL_INVALID_KERNEL_NAME",
            "CL_INVALID_KERNEL_DEFINITION",
            "CL_INVALID_KERNEL",
            "CL_INVALID_ARG_INDEX",
            "CL_INVALID_ARG_VALUE",
            "CL_INVALID_ARG_SIZE",
            "CL_INVALID_KERNEL_ARGS",
            "CL_INVALID_WORK_DIMENSION",
            "CL_INVALID_WORK_GROUP_SIZE",
            "CL_INVALID_WORK_ITEM_SIZE",
            "CL_INVALID_GLOBAL_OFFSET",
            "CL_INVALID_EVENT_WAIT_LIST",
            "CL_INVALID_EVENT",
            "CL_INVALID_OPERATION",
            "CL_INVALID_GL_OBJECT",
            "CL_INVALID_BUFFER_SIZE",
            "CL_INVALID_MIP_LEVEL",
            "CL_INVALID_GLOBAL_WORK_SIZE",
        };

        const int errorCount = sizeof(errorString) / sizeof(errorString[0]);

        const int index = -error;

        return(index >= 0 && index < errorCount) ? errorString[index] : "";

    }



    //NVIDIA's code
    //////////////////////////////////////////////////////////////////////////////
    //! Gets the platform ID for NVIDIA if available, otherwise default to platform 0
    //!
    //! @return the id 
    //! @param clSelectedPlatformID         OpenCL platform ID
    //////////////////////////////////////////////////////////////////////////////
    cl_int oclGetPlatformID(cl_platform_id* clSelectedPlatformID)
    {
        char chBuffer[1024];
        cl_uint num_platforms;
        cl_platform_id* clPlatformIDs;
        cl_int ciErrNum;
        *clSelectedPlatformID = NULL;
        cl_uint i = 0;

        // Get OpenCL platform count
        ciErrNum = clGetPlatformIDs (0, NULL, &num_platforms);
        if (ciErrNum != CL_SUCCESS)
        {
            //shrLog(" Error %i in clGetPlatformIDs Call !!!\n\n", ciErrNum);
            printf(" Error %i in clGetPlatformIDs Call !!!\n\n", ciErrNum);
            return -1000;
        }
        else
        {
            if (num_platforms == 0)
            {
                //shrLog("No OpenCL platform found!\n\n");
                printf("No OpenCL platform found!\n\n");
                return -2000;
            }
            else
            {
                // if there's a platform or more, make space for ID's
                if ((clPlatformIDs = (cl_platform_id*)malloc(num_platforms * sizeof(cl_platform_id))) == NULL)
                {
                    //shrLog("Failed to allocate memory for cl_platform ID's!\n\n");
                    printf("Failed to allocate memory for cl_platform ID's!\n\n");
                    return -3000;
                }

                // get platform info for each platform and trap the NVIDIA platform if found
                ciErrNum = clGetPlatformIDs (num_platforms, clPlatformIDs, NULL);
                for (i = 0; i < num_platforms; ++i)
                {
                    ciErrNum = clGetPlatformInfo (clPlatformIDs[i], CL_PLATFORM_NAME, 1024, &chBuffer, NULL);
                    if (ciErrNum == CL_SUCCESS)
                    {
                        if (strstr(chBuffer, "NVIDIA") != NULL)
                        {
                            *clSelectedPlatformID = clPlatformIDs[i];
                            break;
                        }
                    }
                }

                // default to zeroeth platform if NVIDIA not found
                if (*clSelectedPlatformID == NULL)
                {
                    //shrLog("WARNING: NVIDIA OpenCL platform not found - defaulting to first platform!\n\n");
                    printf("WARNING: NVIDIA OpenCL platform not found - defaulting to first platform!\n\n");
                    *clSelectedPlatformID = clPlatformIDs[0];
                }

                free(clPlatformIDs);
            }
        }

        return CL_SUCCESS;
    }
}

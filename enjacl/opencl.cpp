#include <vector>
#include <string.h>
#include <string>
#include <iostream>
#include <stdlib.h>
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
#include "util.h"
#include "timege.h"
//#include "incopencl.h"

//----------------------------------------------------------------------
int EnjaParticles::update()
{
	printf("inside update\n");
    m_system->update();
#if 0

	ts_cl[0]->start();
#ifdef GL_INTEROP   
    // map OpenGL buffer object for writing from OpenCL
    //clFinish(cqCommandQueue);
    glFinish();

    err = queue.enqueueAcquireGLObjects(&cl_vbos, NULL, &event);
    //printf("acquire: %s\n", oclErrorString(err));
    queue.finish();
#endif

    //clFinish(cqCommandQueue);
	ts_cl[1]->start();
    err = queue.enqueueWriteBuffer(cl_transform, CL_TRUE, 0, 4*sizeof(Vec4), &transform[0], NULL, &event);
    queue.finish();
    
    err = vel_update_kernel.setArg(4, cl_transform);
    err = vel_update_kernel.setArg(5, dt);
    err = queue.enqueueNDRangeKernel(vel_update_kernel, cl::NullRange, cl::NDRange(num), cl::NullRange, NULL, &event);
    queue.finish();

    if(collision)
    {
        err = collision_kernel.setArg(4, dt);
		size_t glob = num; // 10000
		size_t loc = 256;
		try {
		//exit(0);
        //err = queue.enqueueNDRangeKernel(collision_kernel, cl::NullRange, cl::NDRange(glob), cl::NDRange(loc), NULL, &event);
        //err = queue.enqueueNDRangeKernel(collision_kernel, cl::NullRange, cl::NDRange(glob), cl::NDRange(loc), NULL, &event);
        //err = queue.enqueueNDRangeKernel(collision_kernel, cl::NullRange, cl::NDRange(glob), cl::NDRange(loc), NULL, &event);
        //err = queue.enqueueNDRangeKernel(collision_kernel, cl::NullRange, cl::NDRange(glob), cl::NDRange(loc), NULL, &event);
        err = queue.enqueueNDRangeKernel(collision_kernel, cl::NullRange, cl::NDRange(num), cl::NullRange, NULL, &event);
	//printf("end\n");exit(0); // >>>>>>>
		}
      catch (cl::Error err) {
         std::cerr
            << "ERROR: "
            << err.what()
            << "("
            << err.err()
            << ")"
            << std::endl;
      }
      queue.finish();
    }

    err = pos_update_kernel.setArg(3, cl_transform);
    err = pos_update_kernel.setArg(4, dt);
    err = queue.enqueueNDRangeKernel(pos_update_kernel, cl::NullRange, cl::NDRange(num), cl::NullRange, NULL, &event);
    //printf("enqueue: %s\n", oclErrorString(err));
    queue.finish();
    ts_cl[1]->stop();

#ifdef GL_INTEROP
    // unmap buffer object
    //ciErrNum = clEnqueueReleaseGLObjects(cqCommandQueue, 1, &vbo_cl, 0,0,0);
    
    //clFinish(cqCommandQueue);
    err = queue.enqueueReleaseGLObjects(&cl_vbos, NULL, &event);
    //printf("release gl: %s\n", oclErrorString(err));
    queue.finish();
#else

    /* implement this with opencl c++ bindings later
    // Explicit Copy 
    // this doesn't get called when we use GL_INTEROP
    glBindBufferARB(GL_ARRAY_BUFFER, v_vbo);    
    // map the buffer object into client's memory
    void* ptr = glMapBufferARB(GL_ARRAY_BUFFER, GL_WRITE_ONLY_ARB);
    ciErrNum = clEnqueueReadBuffer(cqCommandQueue, cl_vbos[0], CL_TRUE, 0, vbo_size, ptr, 0, NULL, &evt);
    clReleaseEvent(evt);
    glUnmapBufferARB(GL_ARRAY_BUFFER); 
    
    glBindBufferARB(GL_ARRAY_BUFFER, c_vbo);    
    // map the buffer object into client's memory
    ptr = glMapBufferARB(GL_ARRAY_BUFFER, GL_WRITE_ONLY_ARB);
    ciErrNum = clEnqueueReadBuffer(cqCommandQueue, cl_vbos[1], CL_TRUE, 0, vbo_size, ptr, 0, NULL, &evt);
    clReleaseEvent(evt);
    glUnmapBufferARB(GL_ARRAY_BUFFER); 
    */
#endif

	ts_cl[0]->stop();

#endif

#if 1
// Sorting
	std::vector<int> sort_int;
	std::vector<int> unsort_int;
	int nb_el = 2 << 12;
	cl::Buffer cl_sort(context, CL_MEM_WRITE_ONLY, nb_el*sizeof(int), NULL, &err);
	cl::Buffer cl_unsort(context, CL_MEM_WRITE_ONLY, nb_el*sizeof(int), NULL, &err);


	for (int i=0; i < nb_el; i++) {
		sort_int.push_back(0);
		unsort_int.push_back(10000-i);
	}

    try {
        err = queue.enqueueWriteBuffer(cl_sort, CL_TRUE, 0, nb_el*sizeof(int), &unsort_int[0], NULL, &event);

		size_t glob = num; // 10000
		size_t loc = 256;
		err = sort_kernel.setArg(0, cl_unsort);
		//err = sort_kernel.setArg(1, cl_sort);
    	//err = queue.enqueueNDRangeKernel(sort_kernel, cl::NullRange, cl::NDRange(glob), cl::NDRange(loc), NULL, &event);

    } catch (cl::Error er) {
        printf("ERROR: %s(%s)\n", er.what(), oclErrorString(er.err()));
    }

    queue.finish();
    
#endif

}


//----------------------------------------------------------------------
void EnjaParticles::popCorn()
{

    try{
        //#include "physics/collision.cl"
		printf("before load program system\n");
        vel_update_program = loadProgram(sources[system]);
		printf("before load program vel_update\n");
        vel_update_kernel = cl::Kernel(vel_update_program, "vel_update", &err);
        //if(collision) //we setup collision kernel either way
        //{
			printf("before load program sources[collision]\n");
            collision_program = loadProgram(sources[COLLISION]);
#ifdef OPENCL_SHARED
			// version that works (80 fps with 220 tri and 16,000 particles)
			// file: collision_ge.cl
            //collision_kernel = cl::Kernel(collision_program, "collision_ge", &err);

			// experimental version, with collision_ge as starting point
			// file: collision_ge_a.cl
            collision_kernel = cl::Kernel(collision_program, "collision_ge", &err);
#else
            collision_kernel = cl::Kernel(collision_program, "collision", &err);
#endif
            long s = collision_kernel.getWorkGroupInfo<CL_KERNEL_LOCAL_MEM_SIZE>(devices.front());
            printf("kernel local mem: %d\n", s);
            size_t wgs = collision_kernel.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(devices.front());
            printf("kernel workgroup size: %d\n", wgs);
        //}
		printf("before load program sources[position]\n");
        pos_update_program = loadProgram(sources[POSITION]);
        pos_update_kernel = cl::Kernel(pos_update_program, "pos_update", &err);

		printf("before load program sources[sort]\n");
		sort_program = loadProgram(sources[SORT]);
		sort_kernel = cl::Kernel(sort_program, "sort", &err);
    }
    catch (cl::Error er) {
        printf("ERROR: %s(%s)\n", er.what(), oclErrorString(er.err()));
		exit(0);
    }


    //This is a purely internal helper function, all this code could easily be at the bottom of init_cl
    //init_cl shouldn't change much, and this may
    #ifdef GL_INTEROP
        //printf("gl interop!\n");
        // create OpenCL buffer from GL VBO
        cl_vbos.push_back(cl::BufferGL(context, CL_MEM_READ_WRITE, v_vbo, &err));
        //printf("v_vbo: %s\n", oclErrorString(err));
        cl_vbos.push_back(cl::BufferGL(context, CL_MEM_READ_WRITE, c_vbo, &err));
        //printf("c_vbo: %s\n", oclErrorString(err));
        //printf("SUCCES?: %s\n", oclErrorString(ciErrNum));
    #else
        //printf("no gl interop!\n");
        // create standard OpenCL mem buffer
        /* convert this to cpp headers as necessary
        cl_vbos[0] = clCreateBuffer(cxGPUContext, CL_MEM_WRITE_ONLY, vbo_size, NULL, &ciErrNum);
        cl_vbos[1] = clCreateBuffer(cxGPUContext, CL_MEM_WRITE_ONLY, vbo_size, NULL, &ciErrNum);
        cl_vbos[2] = clCreateBuffer(cxGPUContext, CL_MEM_WRITE_ONLY, sizeof(int) * num, NULL, &ciErrNum);
        //Since we don't get the data from OpenGL we have to manually push the CPU side data to the GPU
        ciErrNum = clEnqueueWriteBuffer(cqCommandQueue, cl_vbos[0], CL_TRUE, 0, vbo_size, &vert_gen[0], 0, NULL, &evt);
        clReleaseEvent(evt);
        ciErrNum = clEnqueueWriteBuffer(cqCommandQueue, cl_vbos[1], CL_TRUE, 0, vbo_size, &colors[0], 0, NULL, &evt);
        clReleaseEvent(evt);
        ciErrNum = clEnqueueWriteBuffer(cqCommandQueue, cl_vbos[2], CL_TRUE, 0, sizeof(int) * num, &colors[0], 0, NULL, &evt);
        clReleaseEvent(evt);
        */
        //make sure we are finished copying over before going on
    #endif
    
    //support arrays for the particle system
    cl_vert_gen = cl::Buffer(context, CL_MEM_WRITE_ONLY, vbo_size, NULL, &err);
    cl_velo_gen = cl::Buffer(context, CL_MEM_WRITE_ONLY, vbo_size, NULL, &err);
    cl_velocities = cl::Buffer(context, CL_MEM_WRITE_ONLY, vbo_size, NULL, &err);

    cl_transform = cl::Buffer(context, CL_MEM_WRITE_ONLY, 4*sizeof(Vec4), NULL, &err);
    
    err = queue.enqueueWriteBuffer(cl_vert_gen, CL_TRUE, 0, vbo_size, &vert_gen[0], NULL, &event);
    err = queue.enqueueWriteBuffer(cl_velo_gen, CL_TRUE, 0, vbo_size, &velo_gen[0], NULL, &event);
    err = queue.enqueueWriteBuffer(cl_velocities, CL_TRUE, 0, vbo_size, &velo_gen[0], NULL, &event);
    
    err = queue.enqueueWriteBuffer(cl_transform, CL_TRUE, 0, 4*sizeof(Vec4), &transform[0], NULL, &event);
    
    queue.finish();

    //printf("about to set kernel args\n");
    err = vel_update_kernel.setArg(0, cl_vbos[0]);      //position
    err = vel_update_kernel.setArg(1, cl_vbos[1]);      //color
    err = vel_update_kernel.setArg(2, cl_velo_gen);     //velocity generators
    err = vel_update_kernel.setArg(3, cl_velocities);   //velocities
    //if(collision) //we setup collision either way
    //{
        err = collision_kernel.setArg(0, cl_vbos[0]);      //position
        //printf("collision arg 0: %s\n", oclErrorString(err));
        err = collision_kernel.setArg(1, cl_velocities);   //velocities
        //printf("collision arg 1: %s\n", oclErrorString(err));
    //}

    err = pos_update_kernel.setArg(0, cl_vbos[0]);      //position
    err = pos_update_kernel.setArg(1, cl_vert_gen);     //position generators
    err = pos_update_kernel.setArg(2, cl_velocities);   //velocities
    
    printf("done with popCorn()\n");

}
//----------------------------------------------------------------------
int EnjaParticles::init_cl()
{
    setup_cl();

    ts_cl[0] = new GE::Time("cl update routine", 5);
    ts_cl[1] = new GE::Time("execute kernels", 5);

    popCorn();

    return 1;
}
//----------------------------------------------------------------------
int EnjaParticles::setup_cl()
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
    
    /*
    //assume sharing for now, need to do this check with the c++ bindings
    bool bSharingSupported = false;
    for(unsigned int i = uiDeviceUsed; (!bSharingSupported && (i <= uiEndDev)); ++i) 
    {
        size_t extensionSize;
        ciErrNum = clGetDeviceInfo(cdDevices[i], CL_DEVICE_EXTENSIONS, 0, NULL, &extensionSize );
        //oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
        if(extensionSize > 0) 
        {
            char* extensions = (char*)malloc(extensionSize);
            ciErrNum = clGetDeviceInfo(cdDevices[i], CL_DEVICE_EXTENSIONS, extensionSize, extensions, &extensionSize);
            //oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
            std::string stdDevString(extensions);
            free(extensions);

            size_t szOldPos = 0;
            size_t szSpacePos = stdDevString.find(' ', szOldPos); // extensions string is space delimited
            while (szSpacePos != stdDevString.npos)
            {
                if( strcmp(GL_SHARING_EXTENSION, stdDevString.substr(szOldPos, szSpacePos - szOldPos).c_str()) == 0 ) 
                {
                    // Device supports context sharing with OpenGL
                    uiDeviceUsed = i;
                    bSharingSupported = true;
                    break;
                }
                do 
                {
                    szOldPos = szSpacePos + 1;
                    szSpacePos = stdDevString.find(' ', szOldPos);
                } 
                while (szSpacePos == szOldPos);
            }
        }
    }
    */

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
//----------------------------------------------------------------------
cl::Program EnjaParticles::loadProgram(std::string kernel_source)
{
     // Program Setup
    int pl;
    //size_t program_length;
    printf("load the program\n");
    
    //CL_SOURCE_DIR is set in the CMakeLists.txt
    /*
    std::string path(CL_SOURCE_DIR);
    path += "/" + std::string(relative_path);
    printf("path: %s\n", path.c_str());
    */
    //file_contents is defined in util.cpp
    //it loads the contents of the file at the given path
    //char* cSourceCL = file_contents(path.c_str(), &pl);
    //#include "part1.cl"
    //printf("Program source:\n %s\n", kernel_source);

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
cl::Kernel EnjaParticles::loadKernel(std::string kernel_source, std::string kernel_name)
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
//----------------------------------------------------------------------

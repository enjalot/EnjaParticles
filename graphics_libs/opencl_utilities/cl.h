#ifndef _CL_H_
#define _CL_H_

#include <string>
#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <string>
#include "Vec3i.h"

// mac framework
#if defined (__APPLE_CC__)
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif

#include "cll_program.h"

//#include "array_opencl_1d.h"

class CL {
public:
    static cl_context context;                 // compute context
    static cl_command_queue commands;          // compute command queue
	static bool is_initialized;
	static bool program_added;
    static cl_device_id device_id;  // compute device id
	static bool profiling;
	std::vector<cll_Program> programs;  // multiple programs per kernel

private:
    int err;                        // error code returned from api calls
    
    size_t global;                  // global domain size for our calculation
    size_t local;                   // local domain size for our calculation

    cl_event event;					// time the kernel

	// PROFILING
	cl_ulong prof_queued;
	cl_ulong prof_submit;
	cl_ulong prof_start;
	cl_ulong prof_end;
	size_t return_size;
	std::string compiler_options;

private:
	char *image_filename;
	float angle;   // angle to rotate image by (in radians)

public:
	CL(bool is_profiling = false);
	cll_Program addProgram(const char* path_to_source_file);
	cll_Program& addProgramR(const char* path_to_source_file);
	void setCompilerOptions(const std::string options);
	void buildExecutable(cl_program program);
	cl_mem createReadBuffer(long nb_bytes);
	cl_mem createWriteBuffer(long nb_bytes);
	size_t getMaxWorkSize(cl_kernel kernel);
	void waitForKernelsToFinish();
	cl_platform_id getPlatformId();
	cl_int oclGetPlatformID(cl_platform_id* clSelectedPlatformID);

    // Retrieve data from the device
    cl_event connectReadBuffer(cl_mem buffer,
        cl_bool blocking_read,
        size_t cb,            // number of bytes to copy
        void *ptr,
        size_t offset=0,        // usually zero
        cl_uint num_events_in_wait_list=0,
        const cl_event *event_wait_list=NULL);


    cl_event connectWriteBuffer(cl_mem buffer,
        cl_bool blocking_write,
		// number of bytes to copy
        size_t cb,            
        const void *ptr,
		// usually zero (changed the order wrt original function)
        size_t offset=0,        
        cl_uint num_events_in_wait_list=0,
        const cl_event *event_wait_list=NULL);


	void setProfiling(bool is_profiling);
	//void profile();
	void profile(cl_event event= (cl_event)0);
	void printDeviceID() {
		printf("deviceID: %d\n", (int) device_id);
	}
	//void printKernel(const char* msg="") {
		//printf("%s, kernel: %d\n", msg, (int) kernel);
	//}
};

//----------------------------------------------------------------------

#endif

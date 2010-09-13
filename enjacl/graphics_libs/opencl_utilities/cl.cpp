#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <string>
#include <math.h>
//#include <typeinfo>
#include <glincludes.h>
#include <oclUtils.h>

#include "cl.h"
#include "cll_program.h"
//#include "array_opencl_t.h"
#include "array_opencl_1d.h"

void getDeviceInfo(cl_device_id device_id);
extern void oclPrintDevInfo(int iLogMode, cl_device_id device);

using namespace std;


//----------------------------------------------------------------------
cl_context CL::context = 0;
cl_command_queue CL::commands = 0;
bool CL::is_initialized = false;
bool CL::program_added = false;
cl_device_id CL::device_id = 0;
//cl_kernel CL::kernel = 0;
//cl_program CL::program = 0;
bool CL::profiling = true;
//----------------------------------------------------------------------
cl_event CL::connectWriteBuffer(
    cl_mem buffer,
    cl_bool blocking_write,
    size_t cb,            // number of bytes to copy
    const void *ptr,
    size_t offset,        // usually zero
    cl_uint num_events_in_wait_list,
    const cl_event *event_wait_list)
{
    cl_event event;
    int err = clEnqueueWriteBuffer(commands, buffer, blocking_write,
                    offset, cb, ptr,
                    num_events_in_wait_list,
                    event_wait_list, &event);

    if (err != CL_SUCCESS)
    {
        printf("Error(connectWriteBuffer()): Failed to write to source array!\n");
        exit(1);
    }

    return event;
}
//----------------------------------------------------------------------
// read from device
cl_event CL::connectReadBuffer(
    cl_mem buffer,
    cl_bool blocking_read,
    size_t cb,            // number of bytes to copy
    void *ptr,
    size_t offset,        // usually zero
    cl_uint num_events_in_wait_list,
    const cl_event *event_wait_list)
{
    cl_event event;
    int err = clEnqueueReadBuffer(commands, buffer, blocking_read,
                    offset, cb, ptr,
                    num_events_in_wait_list,
                    event_wait_list, &event);

    if (err != CL_SUCCESS)
    {
        printf("Error(connectReadBuffer()): Failed to write to source array!\n");
        exit(1);
    }
    return event;
}
//----------------------------------------------------------------------
void CL::waitForKernelsToFinish()
{
    //printf("sizeof(commands)= %d\n", (int) sizeof(commands));
    clFinish(commands);
}
//----------------------------------------------------------------------
#if 0
void CL::profile()
{
	cl_int err;

	err = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_QUEUED, sizeof(cl_ulong), &prof_queued, NULL);
	err |= clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_SUBMIT, sizeof(cl_ulong), &prof_submit, NULL);
	err |= clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &prof_start, NULL);
	err |= clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &prof_end, NULL);
		
	if (err != CL_SUCCESS) {
		printf("error with profiling of event. Ignore\n");
	}

	//printf("submit: %d, queued: %d, start: %d, end: %d\n", prof_submit, prof_queued, prof_start, prof_end);
	printf("submit->queued: %g (ms)\n", (prof_submit-prof_queued)/1.e6f);
	printf("start-submit: %g (ms)\n", (prof_start-prof_submit)/1.e6f);
	printf("end-start: %g (ms)\n", (prof_end-prof_start)*1.e-6f);
	printf("end-queued: %g (ms)\n", (prof_end-prof_queued)/1.e6f);
}
#endif
//----------------------------------------------------------------------
void CL::profile(cl_event event)
{
	cl_int err;

	if (profiling) {
		
		err = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_QUEUED, sizeof(cl_ulong), &prof_queued, NULL);
		err |= clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_SUBMIT, sizeof(cl_ulong), &prof_submit, NULL);
		err |= clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &prof_start, NULL);
		err |= clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &prof_end, NULL);
		
		if (err != CL_SUCCESS) {
			printf("error with profiling of event. Ignore\n");
		}

		//printf("submit: %d, queued: %d, start: %d, end: %d\n", prof_submit, prof_queued, prof_start, prof_end);
		printf("submit-queued: %g (ms)\n", (prof_submit-prof_queued)/1.e6f);
		printf("start-submit: %g (ms)\n", (prof_start-prof_submit)/1.e6f);
		printf("end-start: %g (ms)\n", (prof_end-prof_start)*1.e-6f);
		printf("end-queued: %g (ms)\n", (prof_end-prof_queued)/1.e6f);
	}
}
//----------------------------------------------------------------------
size_t CL::getMaxWorkSize(cl_kernel kernel)
{
    size_t local;
    int err = clGetKernelWorkGroupInfo(kernel, device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &local, NULL);

    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to retrieve kernel work group info! %d\n", err);
        exit(1);
    }
    return local;
}
//----------------------------------------------------------------------
cl_mem CL::createReadBuffer(long nb_bytes)
{
    cl_mem input =  clCreateBuffer(context,  CL_MEM_READ_ONLY,  nb_bytes, NULL, NULL);
    if (!input)
    {
        printf("createReadBuffer, Error: Failed to allocate device memory!\n");
        exit(1);
    }
    return input;
}
//----------------------------------------------------------------------
cl_mem CL::createWriteBuffer(long nb_bytes)
{
    cl_mem output = clCreateBuffer(context,  CL_MEM_WRITE_ONLY,  nb_bytes, NULL, NULL);
    if (!output)
    {
        printf("createWriteBuffer, Error: Failed to allocate device memory!\n");
        exit(1);
    }
    return output;
}

//----------------------------------------------------------------------
cll_Program& CL::addProgramR(const char* path_to_source_file)
{
	// Must find a way to only compile a single time. 
	// Define all programs before starting the code? 

	//printf("enter addProgramR\n");

    FILE* fd =  fopen(path_to_source_file, "r");
	if (fd == 0) {
		printf("cannot open file: %s\n", path_to_source_file);
	}
// do not limit string size
	int max_len = 100000;
    char* source = new char [max_len];
    int nb = fread(source, 1, max_len, fd);    

	if (nb > (max_len-2)) { 
        printf("cannot read program from %s\n", path_to_source_file);
        printf("   buffer size too small\n");
    }    
	source[nb] = '\0';

    int err;     
    const size_t sz = strlen(source);
    cl_program program = 
	    clCreateProgramWithSource(context, 1, (const char **) &source, &sz, &err);        
		if (err != 0) {
			printf("error when creating program from source\n");
			exit(1);
		}
    if (!program) {        
		printf("Error: Failed to create compute program!\n");
		printf("context= %d\n", context);
        exit(1); 
    }

	cll_Program* cp = new cll_Program(program);
	programs.push_back(*cp);
	cp->setName(path_to_source_file);


	//printf("name(*): %s\n", cp->getName().c_str());

	//printf("exit addProgram\n");

	buildExecutable(program);

    return *cp;
}
//----------------------------------------------------------------------
cll_Program CL::addProgram(const char* path_to_source_file)
{
	// Must find a way to only compile a single time. 
	// Define all programs before starting the code? 

    FILE* fd =  fopen(path_to_source_file, "r");
// do not limit string size
	int max_len = 10000;
    char* source = new char [max_len];
    int nb = fread(source, 1, max_len, fd);    

	if (nb > (max_len-2)) { 
        printf("cannot read program from %s\n", path_to_source_file);
        printf("   buffer size too small\n");
    }    
	source[nb] = '\0';

    int err;     
    const size_t sz = strlen(source);
    cl_program program = 
	    clCreateProgramWithSource(context, 1, (const char **) &source, &sz, &err);        
		if (err != 0) {
			printf("error when creating program from source\n");
			exit(1);
		}
    if (!program) {        
		printf("Error: Failed to create compute program!\n");
		printf("context= %d\n", context);
        exit(1); 
    }

	cll_Program cp(program);
	programs.push_back(cp);

	//printf("exit addProgram\n");

	buildExecutable(program);

    return cp;
}
//----------------------------------------------------------------------
void CL::setCompilerOptions(const string options)
{
	compiler_options = options;
}
//----------------------------------------------------------------------
void CL::buildExecutable(cl_program program)
{
    // Build the program executable
    //
    err = clBuildProgram(program, 0, NULL, compiler_options.c_str(), NULL, NULL);
    if (err != CL_SUCCESS)
    {
		printf("build: error != 0, err= %d\n", err);

        size_t len;
        char buffer[8*2048]; // maximum program length: 8*2048 characters

        printf("Error: Failed to build program executable!\n");
        int err = clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);

		if (err == CL_INVALID_VALUE) {
			printf("build info: invalid value\n");
		} else if (err == CL_INVALID_PROGRAM) {
			printf("build info: invalid program object\n");
		} else if (err == CL_INVALID_DEVICE) {
			printf("build info: invalid device\n");
		} 
        printf("buffer= %s\n", buffer);
		exit(1);
    }

	//printf("after compile program\n");
}
//----------------------------------------------------------------------
cl_platform_id CL::getPlatformId()
{
printf("**************** enter getPlatformId() ******************\n");
    bool bPassed = true;
    std::string sProfileString = "oclDeviceQuery, Platform Name = ";

    // Get OpenCL platform ID for NVIDIA if avaiable, otherwise default
    char cBuffer[1024];
    cl_platform_id clSelectedPlatformID = NULL; 
    cl_int ciErrNum = oclGetPlatformID (&clSelectedPlatformID);
	printf("selected platformID: %d\n", clSelectedPlatformID);
    //oclCheckError(ciErrNum, CL_SUCCESS);

    // Get OpenCL platform name and version
    ciErrNum = clGetPlatformInfo (clSelectedPlatformID, CL_PLATFORM_NAME, sizeof(cBuffer), cBuffer, NULL);
    if (ciErrNum == CL_SUCCESS) {
        sProfileString += cBuffer;
    } else {
        bPassed = false;
		printf("bPassed = false 1\n");
    }

	sProfileString += ", Platform Version = ";

    ciErrNum = clGetPlatformInfo (clSelectedPlatformID, CL_PLATFORM_VERSION, sizeof(cBuffer), cBuffer, NULL);

	//return clSelectedPlatformID;

    if (ciErrNum == CL_SUCCESS) {
        sProfileString += cBuffer;
    } else {
        bPassed = false;
		printf("bPassed = false 2\n");
    }

	//sProfileString += ", SDK Revision = ";
	//sProfileString += OCL_SDKREVISION;
	sProfileString += ", NumDevs = ";

	// Get and log OpenCL device info
	cl_uint ciDeviceCount;
	cl_device_id *devices;
    ciErrNum = clGetDeviceIDs (clSelectedPlatformID, CL_DEVICE_TYPE_ALL, 0, NULL, &ciDeviceCount);

	if (ciDeviceCount == 0) {
		printf("**** if statement ****, ciDeviceCount == 0\n");
		printf(" No devices found supporting OpenCL (return code %i)\n\n", ciErrNum);
		bPassed = false;
		sProfileString += "0";
	} else if (ciErrNum != CL_SUCCESS) {
		printf("**** else if statement, ciErrNUm != SUCCESS ****, ciDeviceCount == 0\n");
		printf(" Error %i in clGetDeviceIDs call !!!\n\n", ciErrNum);
		bPassed = false;
	} else {
		printf("**** else if statement, ciErrNUm == SUCCESS ****, ciDeviceCount == 0\n");
		char cTemp[2];	
		// For Linux only
		sprintf(cTemp, "%u", ciDeviceCount);
		sProfileString += cTemp;
			if ((devices = (cl_device_id*) malloc(sizeof(cl_device_id) * ciDeviceCount)) == NULL) {
				printf(" Failed to allocate memory for devices\n\n");
				bPassed = false;
			}
		ciErrNum = clGetDeviceIDs(clSelectedPlatformID, CL_DEVICE_TYPE_ALL, ciDeviceCount, devices, &ciDeviceCount);
		printf("*** 1 ****\n");
		if (ciErrNum == CL_SUCCESS) {
			printf("ciDeviceCount= %d\n", ciDeviceCount);
			for (int i=0; i < ciDeviceCount; ++i) {
				clGetDeviceInfo(devices[i], CL_DEVICE_NAME, sizeof(cBuffer), &cBuffer, NULL);
				printf("** devices[%d] = %d\n", i, devices[i]);
				printf(".... enter getDeviceInfo() ...\n");
				getDeviceInfo(devices[i]);
				printf(".... exit getDeviceInfo() ...\n");
				// GE: first arg hardcoded (BAD), not used (code simplification)
printf("** **************** print device info  ******************\n");
				//oclPrintDevInfo(2, devices[i]);
				oclPrintDevInfo(LOGBOTH, devices[i]);
				sProfileString += ", Device = ";
				sProfileString += cBuffer;
			}
		} else {
			bPassed = false;
		}
	}
	printf("sProfileString= %s\n", sProfileString.c_str());

printf("**************** exit getPlatformId() ******************\n");
	return clSelectedPlatformID;
}
//----------------------------------------------------------------------
CL::CL(bool is_profiling)
{
    // Connect to a compute device

	if (is_initialized) return;
	if (!is_initialized) is_initialized = true;

	printf("CL: constructor ++++++++++++++++++++++++++\n");

    int gpu = 1; // = 0 (CPU), = 1 (GPU)
	printf("...CL: enter getPlatformId...\n");
	cl_platform_id platform_id = getPlatformId();
	printf("...CL: exit getPlatformId...\n");
    err = clGetDeviceIDs(platform_id, gpu ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU, 1, &device_id, NULL);

printf("===============================================\n");


    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to create a device group!\n");
        //return EXIT_FAILURE;
    }

	printf("chosen device ID: %d\n", device_id);
	printf("...CL: enter getDeviceInfo...\n");
	getDeviceInfo(device_id);
	printf("...CL: exit getDeviceInfo...\n");

    // Create a compute context
    //
    // One would have to make sure that device_id is the same as earlier. What if it is not? (GE)
	printf("constructor: device_id= %d\n", device_id);
	if (context == 0) { // static variable
    	context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
		printf("created context: %d, device_id= %d\n", context, device_id);
		if (context == 0) exit(0);
	}
	//printf("sizeof(context) = %d\n", sizeof(context));
    if (!context)
    {
        printf("Error: Failed to create a compute context!\n");
        //return EXIT_FAILURE;
 	}

	profiling = is_profiling;

	cl_command_queue_properties properties;
	properties = 0;

	if (profiling) {
		properties |= CL_QUEUE_PROFILING_ENABLE;
	}
	//printf("sizeof(properties) = %d\n", (int) sizeof(properties));

    // Create a command commands
    //
	if (commands == 0) {
    	commands = clCreateCommandQueue(context, device_id, properties, &err);
	}
    if (!commands)
    {
        printf("Error: Failed to create a command commands!\n");
        //return EXIT_FAILURE;
    }
	//printf("exit constructor, device_id= %d\n", device_id);
}
//======================================================================

// CLEANUP
#if 0
// Shutdown and cleanup
//
    clReleaseMemObject(input);
    clReleaseMemObject(output);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(commands);
    clReleaseContext(context);
#endif


//======================================================================

//----------------------------------------------------------------------
void CL::setProfiling(bool is_profiling)
{
	profiling = is_profiling;
	cl_command_queue_properties properties;

	// Loses previously set properties (i.e., out of order execution)
	// this must be fixed. How to retrieve properties of command, easily? 

	if (profiling) {
		properties = CL_QUEUE_PROFILING_ENABLE;
    	clSetCommandQueueProperty(commands, properties, true, NULL);
	} else {
		properties = 0;
    	clSetCommandQueueProperty(commands, properties, false, NULL);
	}
}
//----------------------------------------------------------------------


#if 0
#include "oclUtils.h"
#include <fstream>
#include <vector>
#include <iostream>
#include <algorithm>
#include <stdarg.h>
#endif

//////////////////////////////////////////////////////////////////////////////
//! Gets the platform ID for NVIDIA if available, otherwise default
//!
//! @return the id 
//! @param clSelectedPlatformID         OpenCL platoform ID
//////////////////////////////////////////////////////////////////////////////
cl_int CL::oclGetPlatformID(cl_platform_id* clSelectedPlatformID)
{
    char chBuffer[1024];
    cl_uint num_platforms; 
    cl_platform_id* clPlatformIDs;
    cl_int ciErrNum;
    *clSelectedPlatformID = NULL;

    // Get OpenCL platform count
    ciErrNum = clGetPlatformIDs (0, NULL, &num_platforms);
	printf("number of platforms: %d\n", num_platforms);
    if (ciErrNum != CL_SUCCESS) {
        return -1000;
    } else {
        if(num_platforms == 0) {
            return -2000;
        } else {
            // if there's a platform or more, make space for ID's
            if ((clPlatformIDs = (cl_platform_id*)malloc(num_platforms * sizeof(cl_platform_id))) == NULL) {
                return -3000;
            }

            // get platform info for each platform and trap the NVIDIA platform if found
            ciErrNum = clGetPlatformIDs (num_platforms, clPlatformIDs, NULL);
            for(cl_uint i = 0; i < num_platforms; ++i) {
                ciErrNum = clGetPlatformInfo (clPlatformIDs[i], CL_PLATFORM_NAME, 1024, &chBuffer, NULL);
                if(ciErrNum == CL_SUCCESS) {
                    if(strstr(chBuffer, "NVIDIA") != NULL) {
                        *clSelectedPlatformID = clPlatformIDs[i];
                        break;
                    }
                }
            }

            // default to zeroeth platform if NVIDIA not found
            if(*clSelectedPlatformID == NULL) {
                *clSelectedPlatformID = clPlatformIDs[0];
            }

            free(clPlatformIDs);
        }
    }

    return CL_SUCCESS;
}


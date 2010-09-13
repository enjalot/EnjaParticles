
////////////////////////////////////////////////////////////////////////////////

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
// mac framework
#if defined (__APPLE_CC__)
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif

////////////////////////////////////////////////////////////////////////////////

// Use a static data size for simplicity
//
#define DATA_SIZE (1024)

////////////////////////////////////////////////////////////////////////////////

void getInfo_bool(cl_device_id device_id, cl_device_info info, const char* msg)
{
// there are errors in apple implementation of bool types
// just for int parameter
	int local;
	int err;

    err = clGetDeviceInfo(device_id, info, (size_t) sizeof(bool), &local, NULL);
	if (err != CL_SUCCESS) {
		printf("error getting (%s), err= %d\n", msg, err);
	} else {
		printf("%s: %d\n", msg, (int) local);
	}
}
//----------------------------------------------------------------------
void getInfo_sizet_array(cl_device_id device_id, cl_device_info info, const char* msg)
{
// just for int parameter
	size_t local[3];
	int err;

    err = clGetDeviceInfo(device_id, info, sizeof(size_t), &local, NULL);
	//if (err != CL_SUCCESS) {
		//printf("error getting (%s), err= %d\n", msg, err);
	//} else {
		for (int i=0; i < 3; i++) {
			printf("%s: %ld\n", msg, (long) local[i]);
		}
	//}
}
//----------------------------------------------------------------------
void getInfo_ulong(cl_device_id device_id, cl_device_info info, const char* msg)
{
// just for int parameter
	cl_ulong local;
	int err;

    err = clGetDeviceInfo(device_id, info, sizeof(cl_ulong), &local, NULL);
	if (err != CL_SUCCESS) {
		printf("error getting (%s), err= %d\n", msg, err);
	} else {
		printf("%s: %ld\n", msg, (long) local);
	}
}
//----------------------------------------------------------------------
void getInfo_uint(cl_device_id device_id, cl_device_info info, const char* msg)
{
// just for int parameter
	cl_uint local;
	int err;

    //err = clGetDeviceInfo(device_id, info, sizeof(size_t), &local, NULL);
    err = clGetDeviceInfo(device_id, info, sizeof(uint), &local, NULL);
	if (err != CL_SUCCESS) {
		printf("error getting (%s), err= %d\n", msg, err);
	} else {
		printf("%s: %d\n", msg, (int) local);
	}
}
//----------------------------------------------------------------------
void getInfo_char(cl_device_id device_id, cl_device_info info, const char* msg)
{
// just for int parameter
	char local[1024];
	int err;

    err = clGetDeviceInfo(device_id, info, 1024, &local, NULL);
	if (err != CL_SUCCESS) {
		printf("error getting (%s), err= %d\n", msg, err);
	} else {
		printf("%s: %s\n", msg, local);
	}
}
//----------------------------------------------------------------------
void getDeviceInfo(cl_device_id device_id)
{
	/* GE
    cl_int clGetDeviceInfo(  	cl_device_id device,
  		cl_device_info param_name,
  		size_t param_value_size,
  		void *param_value,
  		size_t *param_value_size_ret)
    */

	getInfo_uint(device_id, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, "max work item dims");
	getInfo_uint(device_id, CL_DEVICE_MAX_WORK_GROUP_SIZE, "group size");
	getInfo_uint(device_id, CL_DEVICE_ADDRESS_BITS, "address bits"); // 32 : (GPU)
	getInfo_bool(device_id, CL_DEVICE_ADDRESS_BITS, "is device available"); 
	getInfo_bool(device_id, CL_DEVICE_ENDIAN_LITTLE, "is endian little"); 
	getInfo_bool(device_id, CL_DEVICE_ERROR_CORRECTION_SUPPORT  , "has error support"); 
	getInfo_char(device_id, CL_DEVICE_EXTENSIONS  , "extension list"); 
	getInfo_ulong(device_id, CL_DEVICE_GLOBAL_MEM_CACHE_SIZE  , "global cache size"); 
	getInfo_uint(device_id, CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE  , "global cacheline size"); 
	getInfo_ulong(device_id, CL_DEVICE_GLOBAL_MEM_SIZE, "global memory size");
	getInfo_bool(device_id, CL_DEVICE_IMAGE_SUPPORT, "has image support");
	getInfo_ulong(device_id, CL_DEVICE_IMAGE2D_MAX_HEIGHT, "2D image height");
	getInfo_ulong(device_id, CL_DEVICE_IMAGE2D_MAX_WIDTH, "2D image width");
	getInfo_ulong(device_id, CL_DEVICE_IMAGE3D_MAX_DEPTH, "3D image depth");
	getInfo_ulong(device_id, CL_DEVICE_IMAGE3D_MAX_HEIGHT, "3D image height");
	getInfo_ulong(device_id, CL_DEVICE_IMAGE3D_MAX_WIDTH, "3D image width");
	getInfo_ulong(device_id, CL_DEVICE_LOCAL_MEM_SIZE  , "local memory size");
	getInfo_uint(device_id, CL_DEVICE_MAX_CLOCK_FREQUENCY  , "clock frequencey");
	getInfo_uint(device_id, CL_DEVICE_MAX_COMPUTE_UNITS    , "max nb compute units");
	getInfo_uint(device_id, CL_DEVICE_MAX_CONSTANT_ARGS      , "max nb constant args");
	getInfo_uint(device_id, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, "max constant buffer size");
	getInfo_ulong(device_id, CL_DEVICE_MAX_MEM_ALLOC_SIZE, "max mem alloc size");
	getInfo_ulong(device_id, CL_DEVICE_MAX_PARAMETER_SIZE, "max parameter size to kernel");
	getInfo_ulong(device_id, CL_DEVICE_MAX_READ_IMAGE_ARGS, "max read image args by kernel");
	getInfo_uint(device_id, CL_DEVICE_MAX_SAMPLERS, "max samplers");
	getInfo_uint(device_id, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS  , "max work item dimensions");

	// ERROR
	getInfo_sizet_array(device_id, CL_DEVICE_MAX_WORK_ITEM_SIZES, "max work item sizes");

	getInfo_char(device_id, CL_DEVICE_PROFILE, "device profile");
	getInfo_char(device_id, CL_DEVICE_PROFILING_TIMER_RESOLUTION  , "profiler timer resolution");
	getInfo_char(device_id, CL_DEVICE_VENDOR    , "device vendor");
	getInfo_uint(device_id, CL_DEVICE_VENDOR_ID      , "device vendor ID");
	getInfo_uint(device_id, CL_DEVICE_VERSION      , "device version");
	getInfo_char(device_id, CL_DRIVER_VERSION        , "driver version");
}
//----------------------------------------------------------------------
////////////////////////////////////////////////////////////////////////////////


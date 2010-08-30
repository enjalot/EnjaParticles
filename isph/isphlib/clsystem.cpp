#include "isph.h"
using namespace isph;


CLSystem* CLSystem::instance = NULL;

CLSystem* CLSystem::Instance()
{
	if(!instance)
		instance = new CLSystem();
	return instance;
}

CLSystem::CLSystem() :
	platformCount(0),
	platforms(NULL)
{
	Initialize();
}

CLSystem::~CLSystem()
{
	LogDebug("Unloading OpenCL system");

	for(cl_uint i=0; i<platformCount; i++)
		delete platforms[i];
}

void CLSystem::Initialize()
{
	LogDebug("Initializing OpenCL system");

	// get platform IDs
	cl_platform_id PIDs[4];
	clGetPlatformIDs(4, PIDs, &platformCount);
	platforms = new CLPlatform*[platformCount];
		
	// get info of all platforms
	for(cl_uint p=0; p<platformCount; p++)
	{
		CLPlatform* platform = new CLPlatform(PIDs[p]);
		
		char buf[4096];
		size_t size;

		clGetPlatformInfo(platform->id, CL_PLATFORM_NAME, sizeof(buf), buf, &size);
		platform->name = std::string(buf, size);
		Log::Send(Log::Info, "Found OpenCL platform: " + platform->name);

		clGetPlatformInfo(platform->id, CL_PLATFORM_VENDOR, sizeof(buf), buf, &size);
		platform->vendor = std::string(buf, size);

		clGetPlatformInfo(platform->id, CL_PLATFORM_VERSION, sizeof(buf), buf, &size);
		platform->clVersion = std::string(buf, size);

		// get device IDs
		cl_device_id DIDs[32];
		clGetDeviceIDs(platform->id, CL_DEVICE_TYPE_ALL, 32, DIDs, &platform->deviceCount);
		platform->devices = new CLDevice*[platform->deviceCount];

		// get info of all availible devices on platform
		for (cl_uint d=0; d<platform->deviceCount; d++)
		{
			CLDevice* device = new CLDevice(platform, DIDs[d]);
			
			clGetDeviceInfo(device->id, CL_DEVICE_TYPE, sizeof(cl_device_type), &device->type, NULL);
			clGetDeviceInfo(device->id, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &device->maxComputeUnits, NULL);
			clGetDeviceInfo(device->id, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(cl_uint), &device->maxClock, NULL);
			clGetDeviceInfo(device->id, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &device->maxWorkGroupSize, NULL);
			clGetDeviceInfo(device->id, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(device->maxWorkItemSize), &device->maxWorkItemSize, NULL);
			clGetDeviceInfo(device->id, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &device->globalMemSize, NULL);
			clGetDeviceInfo(device->id, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &device->localMemSize, NULL);
			clGetDeviceInfo(device->id, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(cl_ulong), &device->maxAllocSize, NULL);

			device->performanceIndex = device->Cores() * device->MaxFrequency();
			// if device isn't CPU, lower its performance index by 25% based on the memory transfer expense account
			if(device->type != CL_DEVICE_TYPE_CPU)
				device->performanceIndex = device->performanceIndex * 3 / 4; 

			clGetDeviceInfo(device->id, CL_DEVICE_NAME, sizeof(buf), buf, &size);
			device->name = std::string(buf, size);
			Log::Send(Log::Info, "Found OpenCL enabled device: " + device->name);

			clGetDeviceInfo(device->id, CL_DEVICE_VENDOR, sizeof(buf), buf, &size);
			device->vendor = std::string(buf, size);

			clGetDeviceInfo(device->id, CL_DEVICE_EXTENSIONS, sizeof(buf), buf, &size);
			std::string extensions(buf, size);

			// get floating precisions supported
			device->fp64 = (extensions.find("cl_khr_fp64") != std::string::npos);
			device->fp16 = (extensions.find("cl_khr_fp16") != std::string::npos);

			// are atomics supported
			device->globalAtomics = 
				(extensions.find("cl_khr_global_int32_base_atomics") != std::string::npos
				&& extensions.find("cl_khr_global_int32_extended_atomics") != std::string::npos);
			device->localAtomics = 
				(extensions.find("cl_khr_local_int32_base_atomics") != std::string::npos
				&& extensions.find("cl_khr_local_int32_extended_atomics") != std::string::npos);

			platform->devices[d] = device;
		}
 
		platforms[p] = platform;
	}
}

std::string CLSystem::ErrorDesc(cl_int status)
{
	switch (status) 
	{
		case CL_SUCCESS:						return "OpenCL - Success!";
		case CL_DEVICE_NOT_FOUND:				return "OpenCL - Device not found.";
		case CL_DEVICE_NOT_AVAILABLE:			return "OpenCL - Device not available";
		case CL_COMPILER_NOT_AVAILABLE:			return "OpenCL - Compiler not available";
		case CL_MEM_OBJECT_ALLOCATION_FAILURE:	return "OpenCL - Memory object allocation failure";
		case CL_OUT_OF_RESOURCES:				return "OpenCL - Out of resources";
		case CL_OUT_OF_HOST_MEMORY:				return "OpenCL - Out of host memory";
		case CL_PROFILING_INFO_NOT_AVAILABLE:	return "OpenCL - Profiling information not available";
		case CL_MEM_COPY_OVERLAP:				return "OpenCL - Memory copy overlap";
		case CL_IMAGE_FORMAT_MISMATCH:			return "OpenCL - Image format mismatch";
		case CL_IMAGE_FORMAT_NOT_SUPPORTED:		return "OpenCL - Image format not supported";
		case CL_BUILD_PROGRAM_FAILURE:			return "OpenCL - Program build failure";
		case CL_MAP_FAILURE:					return "OpenCL - Map failure";
		case CL_INVALID_VALUE:					return "OpenCL - Invalid value";
		case CL_INVALID_DEVICE_TYPE:			return "OpenCL - Invalid device type";
		case CL_INVALID_PLATFORM:				return "OpenCL - Invalid platform";
		case CL_INVALID_DEVICE:					return "OpenCL - Invalid device";
		case CL_INVALID_CONTEXT:				return "OpenCL - Invalid context";
		case CL_INVALID_QUEUE_PROPERTIES:		return "OpenCL - Invalid queue properties";
		case CL_INVALID_COMMAND_QUEUE:			return "OpenCL - Invalid command queue";
		case CL_INVALID_HOST_PTR:				return "OpenCL - Invalid host pointer";
		case CL_INVALID_MEM_OBJECT:				return "OpenCL - Invalid memory object";
		case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:return "OpenCL - Invalid image format descriptor";
		case CL_INVALID_IMAGE_SIZE:				return "OpenCL - Invalid image size";
		case CL_INVALID_SAMPLER:				return "OpenCL - Invalid sampler";
		case CL_INVALID_BINARY:					return "OpenCL - Invalid binary";
		case CL_INVALID_BUILD_OPTIONS:			return "OpenCL - Invalid build options";
		case CL_INVALID_PROGRAM:				return "OpenCL - Invalid program";
		case CL_INVALID_PROGRAM_EXECUTABLE:		return "OpenCL - Invalid program executable";
		case CL_INVALID_KERNEL_NAME:			return "OpenCL - Invalid kernel name";
		case CL_INVALID_KERNEL_DEFINITION:		return "OpenCL - Invalid kernel definition";
		case CL_INVALID_KERNEL:					return "OpenCL - Invalid kernel";
		case CL_INVALID_ARG_INDEX:				return "OpenCL - Invalid argument index";
		case CL_INVALID_ARG_VALUE:				return "OpenCL - Invalid argument value";
		case CL_INVALID_ARG_SIZE:				return "OpenCL - Invalid argument size";
		case CL_INVALID_KERNEL_ARGS:			return "OpenCL - Invalid kernel arguments";
		case CL_INVALID_WORK_DIMENSION:			return "OpenCL - Invalid work dimension";
		case CL_INVALID_WORK_GROUP_SIZE:		return "OpenCL - Invalid work group size";
		case CL_INVALID_WORK_ITEM_SIZE:			return "OpenCL - Invalid work item size";
		case CL_INVALID_GLOBAL_OFFSET:			return "OpenCL - Invalid global offset";
		case CL_INVALID_EVENT_WAIT_LIST:		return "OpenCL - Invalid event wait list";
		case CL_INVALID_EVENT:					return "OpenCL - Invalid event";
		case CL_INVALID_OPERATION:				return "OpenCL - Invalid operation";
		case CL_INVALID_GL_OBJECT:				return "OpenCL - Invalid OpenGL object";
		case CL_INVALID_BUFFER_SIZE:			return "OpenCL - Invalid buffer size";
		case CL_INVALID_MIP_LEVEL:				return "OpenCL - Invalid mip-map level";
		default:								return "OpenCL - Unknown error";
	}
}

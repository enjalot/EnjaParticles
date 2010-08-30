#include "isph.h"
using namespace isph;

CLLink::CLLink(std::vector<CLDevice*>& devices) :
	context(NULL),
	deviceCount(0),
	devices(NULL),
	deviceFactors(NULL),
	queues(NULL)
{
	Initialize(&devices.front(), (unsigned int)devices.size());
}

CLLink::CLLink(CLDevice* device) :
	context(NULL),
	deviceCount(0),
	devices(NULL),
	deviceFactors(NULL),
	queues(NULL)
{
	if(device)
		Initialize(&device, 1);
}

CLLink::CLLink( CLPlatform* platform ) :
	context(NULL),
	deviceCount(0),
	devices(NULL),
	deviceFactors(NULL),
	queues(NULL)
{
	if(platform)
	{
		Initialize(&platform->Device(0), platform->DeviceCount());
	}
	else if(CLSystem::Instance()->FirstPlatform())
	{
		Initialize(&CLSystem::Instance()->FirstPlatform()->Device(0), CLSystem::Instance()->FirstPlatform()->DeviceCount());
	}
}

bool CLLink::Initialize(CLDevice** devices, unsigned int size)
{
	LogDebug("Connecting to OpenCL enabled device(s)");

	if(!size || !devices)
	{
		Log::Send(Log::Error, "No devices to connect to.");
		return false;
	}

	deviceCount = size;
	this->devices = new CLDevice*[size];
	deviceFactors = new double[size];
	queues = new cl_command_queue[size];

	cl_device_id *devicesIDs = new cl_device_id[size];

	cl_uint totalPerformance = 0; 

	unsigned int i;
	for (i=0; i<size; i++)
	{
		this->devices[i] = devices[i];
		devicesIDs[i] = devices[i]->ID();
		totalPerformance += devices[i]->PerformanceIndex();
	}

	cl_int status;
	context = clCreateContext(NULL, (cl_uint)size, devicesIDs, NULL, NULL, &status);
	if(status)
	{
		Log::Send(Log::Error, CLSystem::Instance()->ErrorDesc(status));
		return false;
	}

	for (i=0; i<size; i++)
	{
		LogDebug("Linking to device: " + devices[i]->Name());
		queues[i] = clCreateCommandQueue(context, devices[i]->ID(), 0, &status);
		deviceFactors[i] = (double)devices[i]->PerformanceIndex() / totalPerformance;
	}

	delete [] devicesIDs;
	return true;
}

CLLink::~CLLink()
{
	cl_int status;

	if(deviceCount)
	{
		for (unsigned int i=0; i<deviceCount; i++)
		{
			status = clReleaseCommandQueue(queues[i]);
		}
		delete [] devices;
		delete [] deviceFactors;
		delete [] queues;
	}

	if(context)
	{
		status = clReleaseContext(context);
	}
}

bool CLLink::BuildProgram(cl_program& program, const std::string& buildOptions)
{
	// build program on each device
	for(unsigned int i=0; i<deviceCount; i++)
	{
		std::string finalBuildOptions = buildOptions;

		finalBuildOptions.append(" -cl-mad-enable");

		if(devices[i]->IsGPU())
			finalBuildOptions.append(" -D GPU");
		else if(devices[i]->IsCPU())
			finalBuildOptions.append(" -D CPU");

		cl_int status = clBuildProgram(program, 1, &devices[i]->ID(), finalBuildOptions.c_str(), NULL, NULL);
		
		if(status)
		{
			Log::Send(Log::Error, CLSystem::Instance()->ErrorDesc(status));

			if(status == CL_BUILD_PROGRAM_FAILURE)
			{
				size_t size = 2048;
				char *buf = new char[size];
				clGetProgramBuildInfo(program, devices[i]->ID(), CL_PROGRAM_BUILD_LOG, size, buf, &size);
				Log::Send(Log::Info, std::string(buf, size));
				delete [] buf;
			}

			return false;
		}
	}

	return true;
}

bool CLLink::Finish()
{
	cl_int status;
	bool success = true;

	for(unsigned int i=0; i<deviceCount; i++)
	{
		status = clFinish(queues[i]);
		if(status)
		{
			success = false;
			Log::Send(Log::Error, CLSystem::Instance()->ErrorDesc(status));
		}
	}
	return success;
}

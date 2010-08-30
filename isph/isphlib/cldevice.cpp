#include "isph.h"
using namespace isph;

CLDevice::CLDevice(CLPlatform* parentPlatform, cl_device_id device)
:	platform(parentPlatform),
	id(device),
	type(CL_DEVICE_TYPE_DEFAULT),
	maxComputeUnits(0),
	maxClock(0),
	performanceIndex(0),
	maxWorkGroupSize(0),
	globalMemSize(0),
	localMemSize(0),
	maxAllocSize(0),
	fp16(false),
	fp64(false),
	globalAtomics(false),
	localAtomics(false)
{
}

CLDevice::~CLDevice()
{

}

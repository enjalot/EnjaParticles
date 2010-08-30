#include "isph.h"
#include <string>
#include <cmath>
#include <cstring>

using namespace isph;


CLVariable::CLVariable(CLProgram* program, const std::string& semantic, VariableType type)
	: parentProgram(program)
	, clBuffers(NULL)
	, bufferCount(0)
	, partElementCount(NULL)
	, offsets(NULL)
	, simpleData(NULL)
	, varType(type)
	, varDataType(IntType)
	, elementCount(0)
	, memorySize(0)
	, needsUpdate(false)
{
	if(program)
		program->ConnectSemantic(semantic, this);
	else
		Log::Send(Log::Error, "Cannot create variable for NULL OpenCL program.");
}


CLVariable::~CLVariable()
{
	LogDebug("Destroying variable: " + semantics.front());

	// TODO properly delete semantic connections
	semantics.clear();

	Release();
}


bool CLVariable::ReadTo(void* memoryPos)
{
	LogDebug("Reading variable: " + semantics.front());

	parentProgram->Finish();

	if(!memoryPos)
	{
		Log::Send(Log::Error, "Cannot read OpenCL variable to NULL.");
		return false;
	}

	if(varType == KernelArgument)
	{
		if(simpleData)
		{
			std::memcpy(memoryPos, simpleData, memorySize);
			return true;
		}
		else
		{
			Log::Send(Log::Error, "Cannot read uninitalized variable.");
			return false;
		}
	}

	if(!memorySize || !parentProgram || !clBuffers)
	{
		Log::Send(Log::Error, "Cannot read uninitalized OpenCL buffer.");
		return false;
	}

	cl_int status = 0;
	//cl_event* events = new cl_event[bufferCount];

	// enqueue read data from all devices
	for (unsigned int i=0; i<bufferCount; i++)
	{
		/*size_t offsetBytes = Offset(i) * typeSize;
		char *offsetPos = (char*)memoryPos + offsetBytes;
		if(writeable)
			status = clEnqueueReadBuffer(parentProgram->Link()->Queue(i), clBuffers[i], CL_FALSE, offsetBytes, ElementCount(i)*typeSize, offsetPos, 0, NULL, &events[i]);
		else
			status = clEnqueueReadBuffer(parentProgram->Link()->Queue(i), clBuffers[i], CL_FALSE, 0, ElementCount(i)*typeSize, offsetPos, 0, NULL, &events[i]);*/

		status = clEnqueueReadBuffer(parentProgram->Link()->Queue(i), clBuffers[i], CL_TRUE, 0, memorySize, memoryPos, 0, NULL, NULL);

		if(status)
		{
			Log::Send(Log::Error, CLSystem::Instance()->ErrorDesc(status));
			return false;
		}
	}

	// wait for all devices
	/*status = clWaitForEvents(bufferCount, events);
	for (unsigned int i=0; i<bufferCount; i++)
		clReleaseEvent(events[i]);
	delete [] events;

	if(status)
	{
		Log::Send(Log::Error, CLSystem::Instance()->ErrorDesc(status));
		return false;
	}*/
	
	return true;
}


bool CLVariable::WriteFrom(const void* memoryPos)
{
	LogDebug("Writing variable: " + semantics.front());

	if(needsUpdate)
		if(!Allocate())
			return false;

	if(!memoryPos)
	{
		Log::Send(Log::Error, "Cannot write to OpenCL variable from NULL.");
		return false;
	}
	
	if(varType == KernelArgument)
	{
		if(simpleData)
		{
			std::memcpy(simpleData, memoryPos, memorySize);
			return true;
		}
		else
		{
			Log::Send(Log::Error, "Cannot read uninitalized variable.");
			return false;
		}
	}

	if(!memorySize || !parentProgram || !clBuffers)
	{
		Log::Send(Log::Error, "Cannot write to uninitalized OpenCL buffer.");
		return false;
	}

	cl_int status;

	// enqueue writing data on all devices
	for (unsigned int i=0; i<bufferCount; i++)
	{
		size_t offsetBytes = Offset(i) * DataTypeSize();
		char *offsetPos = (char*)memoryPos + offsetBytes;
		status = clEnqueueWriteBuffer(parentProgram->Link()->Queue(i), clBuffers[i], CL_FALSE, offsetBytes, ElementCount(i)*DataTypeSize(), offsetPos, 0, NULL, NULL);

		if(status)
		{
			Log::Send(Log::Error, CLSystem::Instance()->ErrorDesc(status));
			return false;
		}
	}

	return true;
}


bool CLVariable::WriteFrom(CLVariable* var)
{
	LogDebug("Copying variable: " + var->semantics.front() + ", to variable: " + semantics.front());

	if(!var)
	{
		Log::Send(Log::Error, "Cannot copy from NULL variable.");
		return false;
	}

	if(varType == KernelArgument)
	{
		Log::Send(Log::Error, "Copying simple variables not yet implemented.");
		return true;
	}

	if(!memorySize || !parentProgram || !clBuffers)
	{
		Log::Send(Log::Error, "Cannot write to uninitalized OpenCL buffer.");
		return false;
	}

	cl_int status;

	for (unsigned int i=0; i<bufferCount; i++)
	{
		status = clEnqueueCopyBuffer(parentProgram->Link()->Queue(i), var->clBuffers[i], clBuffers[i], 0, 0, var->MemorySize(), 0, NULL, NULL);
		
		if(status)
		{
			Log::Send(Log::Error, CLSystem::Instance()->ErrorDesc(status));
			return false;
		}
	}

	return true;
}


bool CLVariable::Allocate()
{
	Release();

	LogDebug("Allocating memory on device for variable: " + semantics.front());

	if((varType != LocalBuffer && !memorySize) || !parentProgram)
	{
		Log::Send(Log::Error, "Cannot allocate OpenCL variable without parameters set.");
		return false;
	}

	if(!parentProgram->link)
	{
		Log::Send(Log::Error, "Cannot allocate OpenCL variable without link.");
		return false;
	}
	
	if(varType == LocalBuffer)
	{
		needsUpdate = false;
		return true; // if we allocate local memory, no buffer is needed
	}

	if(varType == KernelArgument)
	{
		simpleData = new char[memorySize];
		needsUpdate = false;
		return true;
	}

	cl_mem_flags flag = CL_MEM_READ_WRITE;
	bufferCount = 1; // parentProgram->Link()->DeviceCount();

	cl_int status;

	if (IsSplit()) // split the buffer on devices
	{	
		clBuffers = new cl_mem[bufferCount];
		partElementCount = new size_t[bufferCount];
		offsets = new size_t[bufferCount];
		size_t off = 0;

		for (unsigned int i=0; i<bufferCount; i++)
		{
			offsets[i] = off;
			partElementCount[i] = (size_t)ceil(parentProgram->Link()->PerformanceFactor(i) * elementCount);

			off += partElementCount[i];
			if(off > elementCount) // total size will exceed elementCount cos of ceil
				partElementCount[i] -= off - elementCount; // fix last size

			clBuffers[i] = clCreateBuffer(parentProgram->Link()->Context(), flag, partElementCount[i] * DataTypeSize(), NULL, &status);
			
			if(status)
			{
				Log::Send(Log::Error, CLSystem::Instance()->ErrorDesc(status));
				return false;
			}
		}
	}
	else // no need for splitting buffer
	{
		clBuffers = new cl_mem;
		clBuffers[0] = clCreateBuffer(parentProgram->Link()->Context(), flag, memorySize, NULL, &status);
		
		if(status)
		{
			Log::Send(Log::Error, CLSystem::Instance()->ErrorDesc(status));
			return false;
		}
	}

	needsUpdate = false;
	return true;
}


bool CLVariable::SetAsArgument(CLSubProgram *kernel, unsigned int argID, unsigned int deviceID, size_t kernelLocalSize)
{
	if(!kernel)
		return false;

	if(!kernel->kernel)
		return false;
	
	cl_int status;

	if(varType == LocalBuffer)
	{
		size_t localMemSize;
		if(kernel->semantics[argID].find("LOCAL_SIZE_") != std::string::npos)
			localMemSize = DataTypeSize() * (kernelLocalSize + 1);
		else
			localMemSize = memorySize;
		status = clSetKernelArg(kernel->kernel, (cl_uint)argID, localMemSize, NULL);
	}
	else if(varType == KernelArgument)
		status = clSetKernelArg(kernel->kernel, (cl_uint)argID, DataTypeSize(), simpleData);
	else if(IsSplit())
		status = clSetKernelArg(kernel->kernel, (cl_uint)argID, sizeof(cl_mem), &clBuffers[deviceID]);
	else
		status = clSetKernelArg(kernel->kernel, (cl_uint)argID, sizeof(cl_mem), clBuffers);

	if(status)
	{
		Log::Send(Log::Error, CLSystem::Instance()->ErrorDesc(status));
		return false;
	}

	return true;
}


void CLVariable::Release()
{
	if(simpleData)
		delete [] simpleData;
	
	cl_int status;
	if(clBuffers)
	{
		if (IsSplit())
		{
			for (unsigned int i=0; i<bufferCount; i++)
			{
				status = clReleaseMemObject(clBuffers[i]);
				if(status)
					Log::Send(Log::Error, CLSystem::Instance()->ErrorDesc(status));
			}
		} 
		else
		{
			status = clReleaseMemObject(clBuffers[0]);
			if(status)
				Log::Send(Log::Error, CLSystem::Instance()->ErrorDesc(status));
		}

		delete [] clBuffers;
	}

	if(partElementCount)
		delete [] partElementCount;

	if(offsets)
		delete [] offsets;

	bufferCount = 0;
	clBuffers = NULL;
	simpleData = NULL;
	partElementCount = NULL;
	offsets = NULL;
}


void CLVariable::SetSpace( VariableDataType dataType, unsigned int elements )
{
	elementCount = elements;
	varDataType = dataType;
	memorySize = elements * DataTypeSize();

	needsUpdate = true;
}


size_t CLVariable::DataTypeSize()
{
	switch(varDataType)
	{
	case FloatType: 
		return sizeof(cl_float);
	case Float2Type: 
		return sizeof(cl_float2);
	case Float4Type:
		return sizeof(cl_float4);
	case DoubleType: 
		return sizeof(cl_double);
	case Double2Type: 
		return sizeof(cl_double2);
	case Double4Type:
		return sizeof(cl_double4);
	case UintType:
		return sizeof(cl_uint);
	case Uint2Type:
		return sizeof(cl_uint2);
	case Uint4Type:
		return sizeof(cl_uint4);
	case IntType:
		return sizeof(cl_int);
	case Int2Type:
		return sizeof(cl_int2);
	case Int4Type:
		return sizeof(cl_int4);
	default:
		return 0;
	}
}

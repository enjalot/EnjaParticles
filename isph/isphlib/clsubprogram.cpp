#include "isph.h"
#include <sstream>
using namespace isph;

CLSubProgram::CLSubProgram()
	: isKernel(false)
	, kernel(NULL)
	, workGroupSizes(NULL)
{
}

CLSubProgram::~CLSubProgram()
{
	ReleaseKernel();
	LogDebug("Subprogram destroyed");
}

bool CLSubProgram::Load(const std::string& filename)
{
	LogDebug("Loading subprogram '" + filename + "'");
	return SetSource(Utils::LoadCLSource(filename));
}

bool CLSubProgram::SetSource(const std::string& sourceCode)
{
	// delete old stuff
	isKernel = false;
	kernelName.clear();
	semantics.clear();
	parallelizeSemantic.clear();

	// set new source
	source = sourceCode;

	// check if it has a kernel function
	std::string::size_type findPos = source.find("__kernel");
	if(findPos != std::string::npos)
	{
		isKernel = true;
		ParseSemantics(findPos);
	}

	return true;
}

inline bool CLSubProgram::IsLiteral(char c)
{
	return (c>='A' && c<='Z') || (c>='a' && c<='z') || (c>='0' && c<='9') || c=='_';
}

void CLSubProgram::ParseSemantics(size_t startStringPos)
{
	// find start of parameter list
	size_t pos = startStringPos;
	while(source[pos]!='(') pos++;
	
	// get kernel name
	size_t nameEnd = pos;
	while(!IsLiteral(source[nameEnd-1])) nameEnd--;
	size_t nameStart = nameEnd;
	while(IsLiteral(source[nameStart-1])) nameStart--;
	kernelName = source.substr(nameStart, nameEnd - nameStart);

	// get semantics of function parameters
	while(source[pos] != ')')
	{
		if(source[pos] == ',')
		{
			semantics.push_back("");
			pos++;
		}
		else if(source[pos] == ':')
		{
			size_t colonPos = pos;

			// goto first char of semantic name
			while(!IsLiteral(source[pos])) pos++;
			size_t wordStart = pos;

			// goto last char of semantic name
			while(IsLiteral(source[pos])) pos++;
			size_t wordEnd = pos;

			// get semantic
			std::string semanticName = source.substr(wordStart, wordEnd - wordStart);
			if(!semantics.size())
				semantics.push_back(semanticName);
			else
				semantics[semantics.size()-1] = semanticName;

			// is this the semantic that function will be parallelized by
			if(source[wordEnd]=='#')
			{
				parallelizeSemantic = semanticName;
				wordEnd++;
			}

			// erase semantic
			source.erase(colonPos, wordEnd - colonPos);
			pos = colonPos;
		}
		else pos++;
	}

	// if user hasnt defined which is parallel semantic, use 1st parameter
	if(parallelizeSemantic.empty() && semantics.size())
		parallelizeSemantic = semantics.front();

}

void CLSubProgram::ReleaseKernel()
{
	if(kernel)
	{
		clReleaseKernel(kernel);
		kernel = NULL;
	}

	if(workGroupSizes)
	{
		delete [] workGroupSizes;
		workGroupSizes = NULL;
	}
}

bool CLSubProgram::Enqueue(size_t globalSize, size_t localSize)
{
	LogDebug("Enqueuing kernel: " + kernelName);

	if(!program)
	{
		Log::Send(Log::Error, "Subprogram doesnt belong to any program.");
		return false;
	}

	if(!isKernel)
	{
		Log::Send(Log::Error, "Cannot run subprogram with no kernel function in it.");
		return false;
	}

	if(!program->IsBuilt())
	{
		Log::Send(Log::Error, "Cannot run subprogram (kernel) of incorrectly built program");
		return false;
	}

	if(!program->Link())
	{
		Log::Send(Log::Error, "No link to devices set to run subprogram (kernel) on.");
		return false;
	}

	if(localSize > globalSize)
	{
		Log::Send(Log::Error, "Local size of subprogram can't be greater than the global size.");
		return false;
	}

	cl_int status = 0;

	for(unsigned int i=0; i<program->Link()->DeviceCount(); i++)
	{
		std::map<std::string,CLVariable*>::iterator arg;

		// enqueue for the device 
		if(!globalSize)
		{
			arg = program->variables.find(parallelizeSemantic);
			if(arg != program->variables.end())
				globalSize = arg->second->ElementCount(i);
			else
			{
				Log::Send(Log::Error, "Error choosing global size for kernel: " + kernelName);
				return false;
			}
		}
		if(!localSize)
		{
			localSize = workGroupSizes[i];
		}
		while(globalSize % localSize != 0) localSize--; // globalSize has to be dividable with localSize

		// pass arguments that opencl can't retain
		for(unsigned int j=0; j<semantics.size(); j++)
		{
			arg = program->variables.find(semantics[j]);
			if(arg != program->variables.end())
			{
				if(arg->second->Type() != GlobalBuffer)
					if(!arg->second->SetAsArgument(this, j, i, localSize))
						Log::Send(Log::Error, "Error setting OpenCL kernel argument: " + semantics[j]);
			}
			else
				Log::Send(Log::Error, "Program doesn't contain variable: " + semantics[j] + ", needed by kernel: " + kernelName);
		}

		status = clEnqueueNDRangeKernel(program->Link()->Queue(i), kernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);

		if(status)
		{
			Log::Send(Log::Error, CLSystem::Instance()->ErrorDesc(status));
			return false;
		}

	}

	return true;
}

bool CLSubProgram::CreateKernel()
{
	LogDebug("Creating kernel: " + kernelName);

	if(!program)
	{
		Log::Send(Log::Error, "Subprogram doesnt belong to any program");
		return false;
	}

	if(!program->Link())
	{
		Log::Send(Log::Error, "No link to devices set to create subprogram (kernel) for.");
		return false;
	}

	if(!isKernel)
	{
		Log::Send(Log::Error, "Cannot create kernel function, subrogram doesnt have any.");
		return false;
	}

	cl_int status;

	// delete previously allocated data, and create new kernel
	ReleaseKernel();
	kernel = clCreateKernel(program->program, kernelName.c_str(), &status);

	if(status)
	{
		Log::Send(Log::Error, CLSystem::Instance()->ErrorDesc(status));
		return false;
	}

	// immediately set arguments that opencl can retain
	SetPersistentArguments();

	workGroupSizes = new size_t[program->Link()->DeviceCount()];

	for(unsigned int i=0; i<program->Link()->DeviceCount(); i++)
	{
		status = clGetKernelWorkGroupInfo(kernel, program->Link()->Device(i)->ID(), CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &workGroupSizes[i], NULL);
		
		if(status)
		{
			Log::Send(Log::Error, CLSystem::Instance()->ErrorDesc(status));
			return false;
		}

		workGroupSizes[i] = std::min(workGroupSizes[i], (size_t)256); // TODO 256 is upper limit ??
	}

	return true;
}

void CLSubProgram::SetPersistentArguments()
{
	if(!isKernel)
		return;

	if(!kernel)
		return;

	for(unsigned int j=0; j<semantics.size(); j++)
	{
		std::map<std::string,CLVariable*>::iterator arg = program->variables.find(semantics[j]);
		if(arg != program->variables.end())
		{
			CLVariable* var = arg->second;
			if(var->Type() == GlobalBuffer)
				if(!var->SetAsArgument(this, j))
					Log::Send(Log::Error, "Error setting OpenCL kernel argument: " + semantics[j]);
		}
		else
			Log::Send(Log::Error, "Program doesn't contain variable: " + semantics[j] + ", needed by kernel: " + kernelName);
	}
}

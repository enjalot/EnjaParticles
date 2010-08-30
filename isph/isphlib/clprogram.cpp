#include "isph.h"
using namespace isph;

CLProgram::CLProgram()
	: link(NULL)
	, unsafeMath(true)
	, isBuilt(false)
	, program(NULL)
{
	CLVariable *var;
	var = new CLVariable(this, "LOCAL_SIZE_UINT", LocalBuffer);  var->SetSpace(UintType, 0);
	var = new CLVariable(this, "LOCAL_SIZE_UINT2", LocalBuffer); var->SetSpace(Uint2Type, 0);
	var = new CLVariable(this, "LOCAL_SIZE_UINT4", LocalBuffer); var->SetSpace(Uint4Type, 0);
	// TODO: add more default variables
}

CLProgram::~CLProgram()
{
	// delete subprograms' allocated data
	ClearSubprograms();

	// delete program variables
	for (std::map<std::string,CLVariable*>::iterator it=variables.begin() ; it != variables.end();)
	{
		CLVariable *var = it->second;
		it++;
		if(var)
		{
			delete var;
			var = NULL;
		}
	}
	variables.clear();
	
	// delete program itself
	if(program)
		clReleaseProgram(program);

	LogDebug("Program destroyed");
}

bool CLProgram::AddSubprogram(CLSubProgram* subprogram)
{
	if(!subprogram)
	{
		Log::Send(Log::Error, "Cannot add NULL subprogram to program");
		return false;
	}

	subprogram->ReleaseKernel();
	subprogram->program = this;
	
	subprograms.push_back(subprogram);
	return true;
}

void CLProgram::SetUnsafeOptimizations(bool enabled)
{
	unsafeMath = enabled;
	isBuilt = false;
}

void CLProgram::AddBuildOption(const std::string& option)
{
	LogDebug("Adding program build/preprocessor option: " + option);
	buildOptions.push_back(option);
}

void CLProgram::ClearBuildOptions()
{
	LogDebug("Clearing program build/preprocessor options");
	buildOptions.clear();
}

bool CLProgram::Build()
{
	LogDebug("Building program.");
	isBuilt = false;
	
	if(!link)
	{
		Log::Send(Log::Error, "No OpenCL devices set to build program on.");
		return false;
	}

	// make the source from subprograms
	source.clear();
	for(size_t i=0; i<subprograms.size(); i++)
	{
		source.append(subprograms[i]->source);
	}

	//Log::Send(Log::Info, source);

	cl_int status;

	// create CL program
	const char* source_cstring = source.c_str();
	size_t source_size = source.size();
	program = clCreateProgramWithSource(link->context, 1, &source_cstring, &source_size, &status); 

	if(status)
	{
		Log::Send(Log::Error, CLSystem::Instance()->ErrorDesc(status));
		return false;
	}

	// set build options
	std::string buildOptionsStr;
	if(unsafeMath)
		buildOptionsStr = "-cl-unsafe-math-optimizations";

	for(size_t i=0; i<buildOptions.size(); i++)
		buildOptionsStr.append(' ' + buildOptions[i]);

	// build on our link (devices)
	if(!link->BuildProgram(program, buildOptionsStr))
		return false;

	// init the variables for devices
	for (std::map<std::string,CLVariable*>::iterator it=variables.begin() ; it != variables.end(); it++)
		if(it->second->needsUpdate)
			it->second->Allocate();

	// create kernels
	for(size_t i=0; i<subprograms.size(); i++)
	{
		if(subprograms[i]->IsKernel())
			subprograms[i]->CreateKernel();
	}

	isBuilt = true;
	return isBuilt;
}

bool CLProgram::Finish()
{
	if(!isBuilt)
		return false;

	if(!link)
		return false;

	return link->Finish();
}

bool CLProgram::SetLink(CLLink* linkToDevices)
{
	LogDebug("Setting new link to devices for program");

	if(!linkToDevices)
	{
		Log::Send(Log::Error, "Cannot set NULL link to devices");
		return false;
	}

	if(!linkToDevices->DeviceCount())
	{
		Log::Send(Log::Error, "Link isn't connected to any device");
		return false;
	}

	link = linkToDevices;
	isBuilt = false;

	return true;
}

void CLProgram::ClearSubprograms()
{
	LogDebug("Clearing program source");

	for(unsigned int i=0; i<subprograms.size(); i++)
		if(subprograms[i])
		{
			subprograms[i]->ReleaseKernel();
			subprograms[i]->program = NULL;
		}

	subprograms.clear();
	isBuilt = false;
}

CLVariable* CLProgram::Variable(const std::string& semantic)
{
	std::map<std::string,CLVariable*>::iterator found = variables.find(semantic);
	if(found != variables.end())
		return found->second;
	return NULL;
}

bool CLProgram::ConnectSemantic(const std::string& semantic, CLVariable* var)
{
	if(!var)
	{
		Log::Send(Log::Error, "Cannot connect a semantic with NULL OpenCL variable.");
		return false;
	}

	if(!semantic.size())
	{
		Log::Send(Log::Error, "Cannot connect an empty semantic with OpenCL variable.");
		return false;
	}

	std::map<std::string,CLVariable*>::iterator found = variables.find(semantic);
	
	if(found != variables.end())
	{
		found->second->semantics.remove(semantic);
	}

	variables[semantic] = var;
	var->semantics.push_back(semantic);

	// if called at run-time of program, update kernels that depend on this semantic
	if(var->Type() == GlobalBuffer && isBuilt)
	{
		// TODO this is temp solution, I'm lazy to search if kernel depend on this semantic, so update ALL
		for(unsigned int i=0; i<subprograms.size(); i++)
			subprograms[i]->SetPersistentArguments();
	}

	return true;
}

size_t CLProgram::UsedMemorySize()
{
	size_t bytes = 0;
	for (std::map<std::string,CLVariable*>::iterator it=variables.begin() ; it != variables.end(); it++)
		bytes += it->second->MemorySize();
	return bytes;
}

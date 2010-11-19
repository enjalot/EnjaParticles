#include "Kernel.h"

namespace rtps {

Kernel::Kernel(CL *cli, std::string name, std::string source)
{
    this->cli = cli;
    this->name = name;
    this->source = source;
    //TODO need to save the program
    kernel = cli->loadKernel(name, source);
	set_profiling = false;
}

//----------------------------------------------------------------------
void Kernel::setProfiling(bool prof)
{
	set_profiling = true;
}
//----------------------------------------------------------------------
void Kernel::execute(int ndrange)
{
    //TODO add error checking
    cli->err = cli->queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(ndrange), cl::NullRange, NULL, &cli->event);

	if (set_profiling == true) {
		cli->queue.finish();
		cl_ulong timeStamp[4]; 
		//timeStamp[0] = cli->event.getProfilingInfo<CL_PROFILING_COMMAND_QUEUED>();
		//timeStamp[1] = cli->event.getProfilingInfo<CL_PROFILING_COMMAND_SUBMIT>();
		timeStamp[2] = cli->event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
		timeStamp[3] = cli->event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
		//printf("queued to end: %f (ms)\n", (timeStamp[3]-timeStamp[0])/1000000.);
		//printf("queued to submit: %f (ms)\n", (timeStamp[1]-timeStamp[0])/1000000.);
		//printf("submit to start: %f (ms)\n", (timeStamp[2]-timeStamp[1])/1000000.);
		printf("start to end: %f (ms)\n", (timeStamp[3]-timeStamp[2])/1000000.);
	}
}
//----------------------------------------------------------------------
void Kernel::execute(int ndrange, int worksize)
{
    //TODO add error checking
    //cli->err = cli->queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(ndrange), cl::NDRange(worksize), NULL, NULL);

    cli->err = cli->queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(ndrange), cl::NDRange(worksize), NULL, &cli->event);

	if (set_profiling == true) {
		cli->queue.finish();
		cl_ulong timeStamp[4]; 
		//timeStamp[0] = cli->event.getProfilingInfo<CL_PROFILING_COMMAND_QUEUED>();
		//timeStamp[1] = cli->event.getProfilingInfo<CL_PROFILING_COMMAND_SUBMIT>();
		timeStamp[2] = cli->event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
		timeStamp[3] = cli->event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
		//printf("queued to end: %f (ms)\n", (timeStamp[3]-timeStamp[0])/1000000.);
		//printf("queued to submit: %f (ms)\n", (timeStamp[1]-timeStamp[0])/1000000.);
		//printf("submit to start: %f (ms)\n", (timeStamp[2]-timeStamp[1])/1000000.);
		printf("start to end: %f (ms)\n", (timeStamp[3]-timeStamp[2])/1000000.);
	}
}
//----------------------------------------------------------------------
void Kernel::setArgShared(int arg, int nb_bytes)
{
    try
    {
        kernel.setArg(arg, nb_bytes, 0);
    }
    catch (cl::Error er) {
        printf("ERROR: %s(%s)\n", er.what(), oclErrorString(er.err()));
    }
}
//----------------------------------------------------------------------
 
}

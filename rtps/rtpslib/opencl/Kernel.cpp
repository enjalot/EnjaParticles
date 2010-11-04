#include "Kernel.h"

namespace rtps {

Kernel::Kernel(CL *cli, std::string name, std::string source)
{
    this->cli = cli;
    this->name = name;
    this->source = source;
    //TODO need to save the program
    kernel = cli->loadKernel(name, source);
}

void Kernel::execute(int ndrange)
{
    //TODO add error checking
    cli->err = cli->queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(ndrange), cl::NullRange, NULL, &cli->event);
    cli->queue.finish();
}

void Kernel::execute(int ndrange, int worksize)
{
    //TODO add error checking
    cli->err = cli->queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(ndrange), cl::NDRange(worksize), NULL, &cli->event);
    cli->queue.finish();
}

 
}

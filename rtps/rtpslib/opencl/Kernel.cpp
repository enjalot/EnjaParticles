#include "Kernel.h"

namespace rtps {

Kernel::Kernel(CL *cli, std::string source, std::string name)
{
    this->cli = cli;
    this->name = name;
    this->source = source;
    //TODO need to save the program
    kernel = cli->loadKernel(source, name);
}
Kernel::Kernel(CL *cli, cl::Program prog, std::string name)
{
    this->cli = cli;
    this->name = name;
    //this->source = source;
    this->program = prog;
    kernel = cli->loadKernel(program, name);
    //TODO need to save the program
    //kernel = cli->loadKernel(source, name);
}

void Kernel::execute(int ndrange)
{
    try
    {
        cli->err = cli->queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(ndrange), cl::NullRange, NULL, &cli->event);
        cli->queue.finish();
    }
    catch (cl::Error er) {
        printf("work group size: %d", ndrange);
        printf("ERROR: %s(%s)\n", er.what(), oclErrorString(er.err()));
    }

}

void Kernel::execute(int ndrange, int worksize)
{
    int global = ndrange / worksize;
    //printf("global: %d\n", global);
    global = worksize*global + worksize;
    //printf("global2: %d\n", global);
    //global = ndrange;

    try
    {
        cli->err = cli->queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(global), cl::NDRange(worksize), NULL, &cli->event);
        cli->queue.finish();


    }
    catch (cl::Error er) {
        printf("ERROR: %s(%s)\n", er.what(), oclErrorString(er.err()));
    }

}

void Kernel::setArgShared(int arg, int nb_bytes)
{
    try
    {
        kernel.setArg(arg, nb_bytes, 0);
        cli->queue.finish();
    }
    catch (cl::Error er) {
        printf("ERROR: %s(%s)\n", er.what(), oclErrorString(er.err()));
    }
}
 
}

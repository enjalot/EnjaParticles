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
    int global = ndrange / worksize;
    //printf("global: %d\n", global);
    global = worksize*global + worksize;
    //printf("global2: %d\n", global);
    //global = ndrange;


    printf("in kernel execute\n");
    try
    {
        //TODO add error checking
        //cli->err = cli->queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(ndrange), cl::NDRange(worksize), NULL, &cli->event);
        cli->err = cli->queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(global), cl::NDRange(worksize), NULL, &cli->event);

    printf("queue finish kernel try\n");
        cli->queue.finish();


    printf("done kernel try\n");
    }
    catch (cl::Error er) {
        printf("ERROR: %s(%s)\n", er.what(), oclErrorString(er.err()));
    }

    printf("done kernel execute\n");
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

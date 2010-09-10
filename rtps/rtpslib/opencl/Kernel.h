#ifndef RTPS_KERNEL_H_INCLUDED
#define RTPS_KERNEL_H_INCLUDED
/*
 * The Kernel class abstracts the OpenCL Kernel class
 * by providing some convenience methods
 *
 * For now we build one program per kernel. In the future
 * we will make it possible to make several kernels per program
 *
 * we pass in an OpenCL instance  to the constructor 
 * which manages the underlying context and queues
 */

#include <string>

#include "CLL.h"

namespace rtps{
class Kernel
{
public:
    Kernel(CL *cli, std::string name, std::string source);

    //we will want to access buffers by name when going accross systems
    std::string name;
    std::string source;

    CL *cli;
    //we need to build a program to have a kernel
    cl::Program program;

    //the actual OpenCL kernel object
    cl::Kernel kernel;

    template <class T> void setArg(int arg, T val);

    void execute();
    
};

}

#endif


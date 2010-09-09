#ifndef RTPS_CL_H_INCLUDED
#define RTPS_CL_H_INCLUDED

#include <CL/cl.hpp>

#include "Buffer.h"

namespace rtps{
class CL
{
public:
    CL();
/*
    std::vector<Buffer> buffers;
    std::vector<Program> programs;
    std::vector<Kernel> kernels;

    int addBuffer(Buffer buff);
    int addProgram(Program prog);
    int addKernel(Kernel kern);
*/
};

}

#endif


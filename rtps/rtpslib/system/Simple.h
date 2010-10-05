#ifndef RTPS_SIMPLE_H_INCLUDED
#define RTPS_SIMPLE_H_INCLUDED

#include <string>

#include "../RTPS.h"
#include "System.h"
#include "../opencl/Kernel.h"
#include "../opencl/Buffer.h"
#include "../opencl/BufferVBO.h"
//#include "../util.h"

namespace rtps {

class Simple : public System
{
public:
    Simple(RTPS *ps, int num);
    ~Simple();

    void update();

    //the particle system framework
    RTPS *ps;

    Kernel k_euler;

    BufferVBO<float4> cl_position;
    BufferVBO<float4> cl_color;
    Buffer<float4> cl_force;
    Buffer<float4> cl_velocity;

};

}

#endif

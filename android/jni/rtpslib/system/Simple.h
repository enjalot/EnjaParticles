#ifndef RTPS_SIMPLE_H_INCLUDED
#define RTPS_SIMPLE_H_INCLUDED

#include <string>

#include "../RTPS.h"
#include "System.h"
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

    std::vector<float4> positions;
    std::vector<float4> colors;
    std::vector<float4> velocities;
    std::vector<float4> forces;

    void cpuEuler();
};

}

#endif

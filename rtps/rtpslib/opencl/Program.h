#ifndef RTPS_PROGRAM_H_INCLUDED
#define RTPS_PROGRAM_H_INCLUDED
/*
 *
*/

#include <string>

#include "CL.h"

namespace rtps{

class Program
{
public:
    Program();

    static cl::Program loadProgram(std::string source);
};

}

#endif


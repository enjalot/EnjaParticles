#ifndef RTPS_PROGRAM_H_INCLUDED
#define RTPS_PROGRAM_H_INCLUDED
/*
 *
*/

#include <string>

#include "CLL.h"
#ifdef WIN32
    #if defined(rtps_EXPORTS)
        #define RTPS_EXPORT __declspec(dllexport)
    #else
        #define RTPS_EXPORT __declspec(dllimport)
	#endif 
#else
    #define RTPS_EXPORT
#endif

namespace rtps{

class RTPS_EXPORT Program
{
public:
    Program();
};

}

#endif


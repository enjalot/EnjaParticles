#ifndef GTOD_WINDOWS_H
#define GTOD_WINDOWS_H
//#include "stdafx.h"
#include <time.h>
#include <windows.h>
#include <iostream>

#ifdef WIN32
    #if defined(rtps_EXPORTS)
        #define RTPS_EXPORT __declspec(dllexport)
    #else
        #define RTPS_EXPORT __declspec(dllimport)
	#endif 
#else
    #define RTPS_EXPORT
#endif

//using namespace System;
using namespace std;
 
#if defined(_MSC_VER) || defined(_MSC_EXTENSIONS)
  #define DELTA_EPOCH_IN_MICROSECS  11644473600000000Ui64
#else
  #define DELTA_EPOCH_IN_MICROSECS  11644473600000000ULL
#endif
 
struct timezone
{
  int  tz_minuteswest; /* minutes W of Greenwich */
  int  tz_dsttime;     /* type of dst correction */
};

// Definition of a gettimeofday function
 
RTPS_EXPORT int gettimeofday(struct timeval *tv, struct timezone *tz);

#endif

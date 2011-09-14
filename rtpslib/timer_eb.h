/****************************************************************************************
* Real-Time Particle System - An OpenCL based Particle system developed to run on modern GPUs. Includes SPH fluid simulations.
* version 1.0, September 14th 2011
* 
* Copyright (C) 2011 Ian Johnson, Andrew Young, Gordon Erlebacher, Myrna Merced, Evan Bollig
* 
* This software is provided 'as-is', without any express or implied
* warranty.  In no event will the authors be held liable for any damages
* arising from the use of this software.
* 
* Permission is granted to anyone to use this software for any purpose,
* including commercial applications, and to alter it and redistribute it
* freely, subject to the following restrictions:
* 
* 1. The origin of this software must not be misrepresented; you must not
* claim that you wrote the original software. If you use this software
* in a product, an acknowledgment in the product documentation would be
* appreciated but is not required.
* 2. Altered source versions must be plainly marked as such, and must not be
* misrepresented as being the original software.
* 3. This notice may not be removed or altered from any source distribution.
****************************************************************************************/


#ifndef _TIME_EB_H_
#define _TIME_EB_H_

// gettimeofday: measured in sec/microsec: wall clock time
// irrespective of CPU/system/threads, etc.


#ifdef WIN32
#include <Windows.h>
#include "gtod_windows.h"
#include <time.h>
//#include <Winsock2.h>
#else
#include <sys/time.h>
#endif

#ifdef WIN32
    #if defined(rtps_EXPORTS)
        #define RTPS_EXPORT __declspec(dllexport)
    #else
        #define RTPS_EXPORT __declspec(dllimport)
	#endif 
#else
    #define RTPS_EXPORT
#endif

#include <string>
#include <vector>
#include <stdio.h>
#include <map> 
#include <string> 

namespace EB {

class RTPS_EXPORT Timer
{
public:
	static std::vector<Timer*> timeList;

private:
#if 1
	struct timeval t_start, t_end;
#endif
	double elapsed;
	float t;
	clock_t t1;
	clock_t t2;
	float scale;
	std::string name;
	std::string unit;
	int count;
	int nbCalls;
	int offset;

public:
	// nbCalls: how many calls before resetting the clock
	// if nbCalls not -1, print time after nbCalls calls
	// offset: how many calls to ignore
	Timer();
	Timer(const char* name, int offset=0, int nbCalls=-1);
	Timer(const Timer&);
	~Timer();
	void reset();
	void begin();
	void end();
	int getCount() { return count;}

	void stop() { end(); }
	void start() { begin(); }

    void set(float t); //add a time from an external timer (GPU)

	static void printAll(FILE* fd=stdout, int label_width=50);
	void print(FILE* fd=stdout, int label_width=50);
    void writeAllToFile(std::string filename="timer_log"); 
	void printReset();
};



class RTPS_EXPORT TimerList : public std::map<std::string, EB::Timer*>
{
    public: 
    void writeToFile(std::string filename) {
        (*(this->begin())).second->writeAllToFile(filename); 
    } 
    void printAll() {
        (*(this->begin())).second->printAll(); 
    } 
};

};
#endif

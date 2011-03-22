#ifndef _TIMEGE_H_
#define _TIMEGE_H_

// gettimeofday: measured in sec/microsec: wall clock time
// irrespective of CPU/system/threads, etc.



#include <string>
#ifdef WIN32
#include <time.h>
#include <Windows.h>
#include "gtod_windows.h"
#else
#include <sys/time.h>
#endif

#include <vector>
//#include "time.h"
#ifdef WIN32
    #if defined(rtps_EXPORTS)
        #define RTPS_EXPORT __declspec(dllexport)
    #else
        #define RTPS_EXPORT __declspec(dllimport)
	#endif 
#else
    #define RTPS_EXPORT
#endif

namespace GE {

class RTPS_EXPORT Time
{
public:
	static std::vector<Time*> timeList;

private:
	struct timeval t_start, t_end;
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
	Time();
	Time(const char* name, int offset=0, int nbCalls=-1);
	Time(const Time&);
	~Time();
	void reset();
	void begin();
	void end();
	int getCount() { return count;}

	void stop() { end(); }
	void start() { begin(); }
	static void printAll();
	void print();
	void printReset();
};

}


#endif

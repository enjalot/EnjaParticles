#ifndef _TIMEGE_H_
#define _TIMEGE_H_

// gettimeofday: measured in sec/microsec: wall clock time
// irrespective of CPU/system/threads, etc.

#include <string>
#include <sys/time.h>
#include <vector>
//#include "time.h"

namespace GE {

class Time
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

public:
	// nbCalls: how many calls before resetting the clock
	// if nbCalls not -1, print time after nbCalls calls
	Time(const char* name, int nbCalls=-1);
	~Time();
	void reset();
	void begin();
	void end();

	void stop() { end(); }
	void start() { begin(); }
	static void printAll();
	void print();
	void printReset();
};

}


#endif

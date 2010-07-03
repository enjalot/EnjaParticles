#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "timege.h"

using namespace GE;

// Must initialize in cpp file to avoid multiple definitions
std::vector<GE::Time*> GE::Time::timeList;

//----------------------------------------------------------------------
Time::Time(const char* name_, int offset, int nbCalls)
{
	name = name_;

	switch (CLOCKS_PER_SEC) {
	case 1000000:
		scale = 1000. / (float) CLOCKS_PER_SEC;
		break;
	case 1000:
		scale = 1. / (float) CLOCKS_PER_SEC;
		break;
	default:
		printf("Time does handle this case\n");
		printf("CLOCKS_PER_SEC= %ld\n", (long) CLOCKS_PER_SEC);
		exit(0);
	}
	count = 0;
	unit = "ms";
	t = 0.0;
	t1 = 0;
	t2 = 0;

    ocount = 0;
    this->offset = offset;
	this->nbCalls = nbCalls;
	timeList.push_back(this);
	//printf("constructor: this= %d, name= %s\n", this, name.c_str());
	reset();
}
//----------------------------------------------------------------------
Time::~Time()
{
}
//----------------------------------------------------------------------
void Time::reset()
{
	t = 0.0;
	t1 = clock();
	count = 0;
}
//----------------------------------------------------------------------
void Time::begin()
{
    if(ocount != offset)
    {
        return;
    }
	gettimeofday(&t_start, NULL);
	t1 = clock();
	t2 = 0.0;
	count++;
}
//----------------------------------------------------------------------
void Time::end()
{
    if(ocount != offset)
    {
        ocount++;
        return;
    }
    ocount = 0;
	gettimeofday(&t_end, NULL);
	double tt = (t_end.tv_sec - t_start.tv_sec) +
	     (t_end.tv_usec - t_start.tv_usec) * 1.e-6;
	//printf("tt= %f\n", tt);
	t += 1000*tt;

	if (count == nbCalls) {
		print();
		reset();
	}

	//t +=  (clock() - t1) * scale;
}
//----------------------------------------------------------------------
void Time::print()
{
	if (count <= 0) return;
	printf("%s: tot (ms): %g, avg: %g, (count=%d)\n", 
		name.c_str(), t, t/count, count);
}
//----------------------------------------------------------------------
void Time::printReset()
{
	//end();
	// I would rather control end() myself
	print();
	reset();
}
//----------------------------------------------------------------------
void Time::printAll()
{
#if 1
	printf("********* printAll\n");
	for (int i=0; i < timeList.size(); i++) {
		Time& tim = *(timeList[i]);
		tim.print();

		/***
		printf("*** tim= %d, msg= %s\n", &tim, tim.name.c_str());
		if (tim.count <= 0) continue;
		printf("tim.count= %d\n", tim.count);
		printf("tim.t= %f\n", tim.t);
		printf("%s: tot: %f, avg: %f (ms), (count=%d)\n", 
			tim.name.c_str(), tim.t, tim.t/tim.count, tim.count);
		***/
	}
#endif
}
//----------------------------------------------------------------------

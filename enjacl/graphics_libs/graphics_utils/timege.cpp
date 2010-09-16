#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "timege.h"

using namespace GE;

// Must initialize in cpp file to avoid multiple definitions
std::vector<GE::Time*> GE::Time::timeList;

//----------------------------------------------------------------------
Time::Time()
{
	name = "";
	scale = 0.;
	count = 0;
	unit = "ms";
	t = 0.0; 
	t1 = 0;
	t2 = 0;

	this->nbCalls = 0;
	this->offset = 0;
	reset();
}
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

	this->nbCalls = nbCalls;
	this->offset = offset;
	timeList.push_back(this);
	//printf("constructor: this= %d, name= %s\n", this, name.c_str());
	reset();
}
//----------------------------------------------------------------------
Time::Time(const Time& t)
{
	name = t.name;
	scale = t.scale;
	count = t.count;
	this->t = t.t;
	this->t1 = t.t1;
	this->t2 = t.t2;
	this->nbCalls = t.nbCalls;
	this->offset = t.offset;
	this->timeList.push_back(this);
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
	if (count < offset) {
		count++;
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
	if (count <= offset) return;

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
	int real_count = count - offset;
	printf("%s: tot (ms): %g, avg: %g, (count=%d)\n", 
		name.c_str(), t, t/real_count, real_count);
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
	for (int i=0; i < timeList.size(); i++) {
		Time& tim = *(timeList[i]);
		tim.print();
	}
#endif
}
//----------------------------------------------------------------------

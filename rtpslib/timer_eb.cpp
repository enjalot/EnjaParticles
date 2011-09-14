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


#include <stdio.h>
#include <stdlib.h>
//#include <sys/time.h>
#include "timer_eb.h"

using namespace EB;

// Modified Gordons timer and ran into compilation and linking 
// problems so Ive been forced to rename to my own timer


// Must initialize in cpp file to avoid multiple definitions
std::vector<EB::Timer*> EB::Timer::timeList;

//----------------------------------------------------------------------
Timer::Timer()
{
#if 1
	static int const_count = 0; 
	if (!const_count) {
		timeList.resize(0); 
	}
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
#endif
}
//----------------------------------------------------------------------
Timer::Timer(const char* name_, int offset, int nbCalls)
{
#if 1
	name = name_;

	switch (CLOCKS_PER_SEC) {
	case 1000000:
		scale = 1000. / (float) CLOCKS_PER_SEC;
		break;
	case 1000:
		scale = 1. / (float) CLOCKS_PER_SEC;
		break;
	default:
		printf("Timer does handle this case\n");
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
#endif
}
//----------------------------------------------------------------------
Timer::Timer(const Timer& t)
{
#if 1
	name = t.name;
	scale = t.scale;
	count = t.count;
	this->t = t.t;
	this->t1 = t.t1;
	this->t2 = t.t2;
	this->nbCalls = t.nbCalls;
	this->offset = t.offset;
	timeList.push_back(this);
	reset();
#endif
}
//----------------------------------------------------------------------
Timer::~Timer()
{
}
//----------------------------------------------------------------------
void Timer::reset()
{
#if 1
	t = 0.0;
	t1 = clock();
	count = 0;
#endif
}
//----------------------------------------------------------------------
void Timer::begin()
{
#if 1
	if (count < offset) {
		count++;
		return;
	}
	gettimeofday(&t_start, NULL);
	t1 = clock();
	t2 = 0.0;
	count++;
#endif
}
//----------------------------------------------------------------------
void Timer::end()
{
#if 1
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
#endif

	//t +=  (clock() - t1) * scale;
}

void Timer::set(float tt)
{
#if 1
    count++;
    if (count <= offset) return;
    t += tt;
    if (count == nbCalls) {
        print();
        reset();
    }
#endif
}
//----------------------------------------------------------------------
void Timer::print(FILE* fd, int label_width)
{
#if 1
	if (count <= 0) return;
	int real_count = count - offset;
	if (name.length() > label_width) { 
        fprintf(fd, "%-*.*s...  |  avg: %10.4f  |  tot: %10.4f  |  count=%6d\n", 
                label_width-3, label_width-3, 
                name.c_str(), t/real_count, t, real_count);
    } else {
	fprintf(fd, "%-*.*s  |  avg: %10.4f  |  tot: %10.4f  |  count=%6d\n", 
            label_width, label_width, 
            name.c_str(), t/real_count, t, real_count);
    }
#endif
}
//----------------------------------------------------------------------
void Timer::printReset()
{
#if 1
	//end();
	// I would rather control end() myself
	print();
	reset();
#endif
}
//----------------------------------------------------------------------
void Timer::printAll(FILE* fd, int label_width)
{
#if 1
	fprintf(fd, "====================================\n"); 
	fprintf(fd, "Timers [All times in ms (1/1000 s)]: \n"); 		
	fprintf(fd, "====================================\n\n");     
	for (int i=0; i < timeList.size(); i++) {
		Timer& tim = *(timeList[i]);
		tim.print(fd, label_width);
	}
	fprintf(fd, "\nNOTE: only timers that have called Timer::start() are shown. \n");
	fprintf(fd, "      [A time of 0.0 may indicate the timer was not stopped.]\n"); 
	fprintf(fd, "====================================\n"); 
#endif
}
//----------------------------------------------------------------------
void Timer::writeAllToFile(std::string filename) 
{
#if 1
    // Get the max label width so we can show all columns the same
    // width and show the FULL label for each timer
    int label_width = 50; 
    for (int i = 0; i < timeList.size(); i++) {
        Timer& tim = *(timeList[i]); 
        if (tim.name.length() > label_width) {
            label_width = tim.name.length(); 
        }
    }

    FILE* fd = fopen(filename.c_str(), "w"); 
    printAll(fd, label_width); 
    fclose(fd); 
#endif
}

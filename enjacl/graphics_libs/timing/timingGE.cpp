#ifndef WINDOWS
#include "timingGE.h"

void Timer::start(int type) 
{
	if (type == CHILDREN) {
		tStart = etimeChildren();
		//printf("tStart= %f\n", tStart);
	} else if (type == SELF) {
		tStart = etimeSelf();
	}
}

void Timer::end(int type) 
{
	float tEnd = -1.;
	//printf(" enter end\n");
	if (type == CHILDREN) {
	//printf(" type = CHILDREN\n");
		tEnd = etimeChildren();
	} else if (type == SELF) {
		tEnd = etimeSelf();
	}
	//printf("tStart, tEnd= %f, %f\n", (float) tStart, (float) tEnd);
	store.addTiming(name, tEnd - tStart);
	tStart = -1.;
}

void Timer::reset()
{
	//printf("store reset: %s\n", name.c_str());
	store.reset(name);
}

double Timer::etimeChildren()
{
    getrusage(RUSAGE_CHILDREN, &ru);

	// time in sec
    tmp[0] = (double) ru.ru_utime.tv_sec + (double) ru.ru_utime.tv_usec * 1e-6;
    tmp[1] = (double) ru.ru_stime.tv_sec + (double) ru.ru_stime.tv_usec * 1e-6;

	//printf("--\nsec,usec= %d, %d\n", ru.ru_utime.tv_sec, ru.ru_utime.tv_usec);
	//printf("sec,usec= %f, %f\n--\n", ru.ru_utime.tv_sec, ru.ru_utime.tv_usec);

	//printf("tmp[0-1]= %f, %f\n", tmp[0], tmp[1]);
    return (tmp[0] + tmp[1]);
}

double Timer::etimeSelf()
{
    getrusage(RUSAGE_SELF, &ru);

    tmp[0] = ru.ru_utime.tv_sec + ru.ru_utime.tv_usec * 1.e-6;
    tmp[1] = ru.ru_stime.tv_sec + ru.ru_stime.tv_usec * 1.e-6;

    return (tmp[0] + tmp[1]);
}

Timer::~Timer() 
{
	if(tStart) {
		end();
	}
}


void Timings::addTiming(std::string name, double diff)
{
	Timing &last = timings[name];
		last.count++;
		last.sum += diff;
		last.max = std::max(last.max, diff);
		last.min = std::min(last.min, diff);
}

void Timings::reset(std::string name)
{
	Timing &last = timings[name];
		last.count = 0;
		last.sum = 0.0;
		last.max = 0.0;
		last.min = 0.0;
		printf("set count = 0\n");
}

void Timings::dumpTimings()
{
	std::cout << "Timings dump\n";
	std::cout << "\"Name\",\"Count\",\"Total (msec)\",\"Mean\",\"Min\",\"Max\"\n";

	for(std::map<std::string, Timing >::const_iterator iter = timings.begin();
		iter != timings.end(); iter++) {
		std::cout << '"' << iter->first 
		<< "\",\"" << iter->second.count
		<< "\",\"" << iter->second.sum *1000
		<< "\",\"" << (iter->second.sum/iter->second.count)*1000
		<< "\",\"" << iter->second.min*1000
		<< "\",\"" << iter->second.max*1000 << "\"\n";
		}
}

#else 

#endif


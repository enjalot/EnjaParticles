#include <stdio.h>
#include <math.h>
#include "timingGE.h"

double s();

int main()
{
	Timings tm;
	Timer t1(tm, "total");
	double f = 0.;

	t1.start();

#if 1
	for (int i=0; i < 10000000; i++) {
		f = cos(i + f);
	}
#else
	f = s();
#endif
	printf("f = %f\n", f);

	t1.end();
	tm.dumpTimings();
}

double s()
{
	double f = 0.;
	for (int i=0; i < 10000000; i++) {
		f = cos(i + f);
	}
	return f;
}

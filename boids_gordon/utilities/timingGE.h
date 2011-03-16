#ifdef WINDOWS
#define _TIMINGGE_H_
#endif

#ifndef _TIMINGGE_H_
#define _TIMINGGE_H_

#include <limits> // For numeric_limits
#include <map>
#include <string>
#include <sys/time.h>
#include <sys/resource.h>
#include <iostream>

using namespace std;


#if 0
extern __inline__ unsigned long long int rdtsc()
{
  unsigned long long int x;
  __asm__ volatile (".byte 0x0f, 0x31" : "=A" (x));
  return x;
}
#endif

class Timings;

//----------------------------------------------------------------------
class Timer {
  Timings &store;
  std::string name;
  struct rusage ru;
  double tmp[2];

  //unsigned long long tStart;
  double tStart;

public:
  enum {CHILDREN=0, SELF=1};
  enum {RUSAGESELF=0, RUSAGECHILDREN=-1};

public:
  Timer(Timings &st, std::string n) : store(st), name(n) 
  { 
  	tStart = 0.; 
	reset();
  }
  //{ start(); } 

  ~Timer();

  void start(int type=SELF);
  void end(int type=SELF);
  void reset();
  double etimeSelf();
  double etimeChildren();
};

//----------------------------------------------------------------------

struct Timing {
    int count;
    double sum;
    double min;
    double max;

    //Timing() : count(0), sum(0), min(std::numeric_limits<double>::max()), max(0.) {}
    Timing() : count(0), sum(0), min(1.e10), max(-1.e10) {}
};

//----------------------------------------------------------------------

class Timings {
public:

  std::map<std::string, Timing > timings;

 public:
  Timings() {}

  void dumpTimings();

  void addTiming(std::string name, double diff);
  void reset(std::string name);
};

#endif

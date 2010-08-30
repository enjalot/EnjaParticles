#ifndef ISPH_CPUBITONICSORT_H
#define ISPH_CPUBITONICSORT_H

#include "vec.h"
#include <cmath>
#include <iostream>
using namespace std;

namespace isph {

class CpuBitonicSort
{
public:
	CpuBitonicSort(unsigned int numElements);
	~CpuBitonicSort();

	void sort(Vec<2,unsigned int> *pData);

private:
	unsigned int  mNumElements;       // Number of elements to be sorted
	unsigned int  mNumElementsP2;     // Closes power of 2 above  number of elements to be sorted
	unsigned int  mP2Exponent;        // Exponent of closest power of 2 
	unsigned int  mNumPhases;         // Number of phases in bitonic sorting
    unsigned int  mNumThreads;        // Emulate number of threads
    unsigned int  mMaxVal;        // Emulate number of threads

	void bitonicStep(Vec<2,unsigned int> *pData, 
					unsigned int phase, 
					unsigned int step, 
					unsigned int inv, 
					unsigned int ssize, 
					unsigned int stride);

	inline unsigned int iLog2(unsigned int value)
    {
      unsigned int l = 0;
      while( (value >> l) > 1 ) ++l;
      return l;
    }

};
}
#endif

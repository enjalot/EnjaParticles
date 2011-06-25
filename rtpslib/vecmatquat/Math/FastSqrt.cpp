// FastSqrt.cpp
//


#include "FastSqrt.h"
#include <math.h>


typedef union FastSqrtUnion FastSqrtUnion, *PFastSqrtUnion, &RFastSqrtUnion;

union FastSqrtUnion
{
	Float  float_val;
	UInt32 uint_val;
};


UInt32 FastSqrtTable[0x10000];


Void BuildSqrtTable()
{
	Int i;
	FastSqrtUnion squareroot;

	for (i = 0; i <= 0x7FFF; i++)
	{
		squareroot.uint_val  = (i << 8) | (0x7F << 23);
		squareroot.float_val = (Float) sqrt(squareroot.float_val);

		FastSqrtTable[i + 0x8000] = (squareroot.uint_val & 0x7FFFFF);

		squareroot.uint_val  = (i << 8) | (0x80 << 23);
		squareroot.float_val = (Float) sqrt(squareroot.float_val);

		FastSqrtTable[i] = (squareroot.uint_val & 0x7FFFFF);
	}
}
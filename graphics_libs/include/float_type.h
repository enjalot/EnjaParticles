#ifndef _FLOAT_TYPE_H_
#define _FLOAT_TYPE_H_

#ifdef DOUBLE

#pragma OPENCL EXTENSION cl_khr_fp64 : enable
typedef double FLOAT;

#else

typedef float FLOAT;

#endif


#endif

//
// platform.h
//
// 2004 Patrick Crawley
//
// this holds some platform specific code for the application
// so far the #else clauses of my #ifdef THIS_OS are always
// specificed to be Linux
//

#ifndef _PLATFORM_H_
#define _PLATFORM_H_

//
// define only one of these or none at all
// this was a bad idea i realise now, i should of made this done
// at runtime so that i could do a check at the begining of the
// application.... TODO
//
// This file should be included before all other user files

#if defined (__APPLE_CC__)
  #define RADEON_VID_CARD
#else 
  #define NVIDIA_VID_CARD
#endif

#ifdef NVIDIA_VID_CARD
	//#define GL_ARB_texture_rectangle 1
	#define TARGET GL_TEXTURE_RECTANGLE_NV
#endif

#ifdef RADEON_VID_CARD
//	#define TARGET GL_TEXTURE_2D
	#define TARGET GL_TEXTURE_RECTANGLE_ARB
#endif

//
// two rand number generators that output the number in the [0,1) range
//
// the 'i' parameter to srand_float is a seed value if 'i' is zero
// then srand_float will try to generate, by some OS dependent way, a
// unique number for the seed
// most of the time just set 'i' to 0
//
//extern float rand_float();
//extern double rand_double(double a, double b);
//extern int rand_float(int a, int b);
extern void  srand_float(int i);
extern bool isNvidiaCard();

#include "glincludes.h"
//#include "globals.h"


#endif


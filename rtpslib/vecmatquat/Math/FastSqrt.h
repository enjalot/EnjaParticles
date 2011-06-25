// FastSqrt.h
//


#ifndef _3D_FASTSQRT_H
#define _3D_FASTSQRT_H


#include "Types.h"


#define FSqrt(x, squareroot)  { extern UInt32 FastSqrtTable[0x10000]; *((UInt *) &(squareroot)) = (FastSqrtTable[(*(Int *) &((Float) x) >> 8) & 0xFFFF] ^ ((((*(Int *) &((Float) x) - 0x3F800000) >> 1) + 0x3F800000) & 0x7F800000)); }

ForceInline Float FastSqrt(Float x)
{
	FSqrt(x, x);
	return x;
}


Void BuildSqrtTable();


#endif
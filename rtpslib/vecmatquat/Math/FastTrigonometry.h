// FastTrigonometry.h
//


#ifndef _3D_FASTTRIGONOMETRY_H
#define _3D_FASTTRIGONOMETRY_H


#include "Types.h"
#include "Math\Vector\Vector 3.h"


ForceInline Float __fastcall SinCos(Float x, PFloat cosval)
{
	x; cosval;						// Just for this shit "unreferenced formal parameter" warning

	__asm
	{
		fld      dword ptr [x]
		fsincos
		fstp     dword ptr [cosval]
	}
}

ForceInline Void __fastcall SinCosVec(PVector3 vec, PVector3 sinvalues, PVector3 cosvalues)
{
	vec; sinvalues; cosvalues;		// Just for this shit "unreferenced formal parameter" warning

	__asm
	{
		mov      eax, vec
		mov      ecx, sinvalues
		mov      edx, cosvalues

		fld      dword ptr [eax]
		fsincos
		fstp     dword ptr [ecx]
		fstp     dword ptr [edx]

		fld      dword ptr [eax+4]
		fsincos
		fstp     dword ptr [ecx+4]
		fstp     dword ptr [edx+4]

		fld      dword ptr [eax+8]
		fsincos
		fstp     dword ptr [ecx+8]
		fstp     dword ptr [edx+8]
	}
}

ForceInline Float __fastcall CoTan(Float x)
{
	x;								// Just for this shit "unreferenced formal parameter" warning

    __asm
	{
		fld      dword ptr [x]
		fptan
		fdivrp   st(1), st
	}
}


#endif
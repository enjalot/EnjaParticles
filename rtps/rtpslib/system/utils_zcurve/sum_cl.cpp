#ifndef _UTILS_CL_H_
#define _UTILS_CL_H_

#define ALMOST_ZERO_BANK_CONFLICTS
#include "bank_conflicts.h"

// single warp of 32 threads summing 32 elements in loc
// unro

inline void sum(__local float4* locc)
{
// blocks of size 32 only

	int lid = get_local_id(0);
	#if 0
	{ LOCC(lid) += LOCC(lid + 16); }
	{ LOCC(lid) += LOCC(lid +  8); }
	{ LOCC(lid) += LOCC(lid +  4); }
	{ LOCC(lid) += LOCC(lid +  2); }
	{ LOCC(lid) += LOCC(lid +  1); }
	#endif

	#if 1
	{ locc[lid] += locc[lid + 16]; }
	{ locc[lid] += locc[lid +  8]; }
	{ locc[lid] += locc[lid +  4]; }
	{ locc[lid] += locc[lid +  2]; }
	{ locc[lid] += locc[lid +  1]; }
	#endif

// there are bank conflicts
// sum in loc[0]

#if 0
	int ai;
	int bi;
	int offset = 1;
	int sz = 16;

	int lid = get_local_id(0);
	for (int d = sz; d > 0; d >>=1) {
		barrier(CLK_LOCAL_MEM_FENCE); // should not be required

		if (lid < d) {
			ai = offset*(2*lid+1)-1;
			bi = offset*(2*lid+2)-1;
			//loc[bi] += loc[ai];
			LOC(bi) += LOC(ai);
		}
		offset <<= 1;
	}
#endif

	//return loc[bi]; // sum should not be required since in local storage

	// manual: 
	//d = 16: 
	// lid < 16
	// offset = 1
	// ai = 1*(2*lid+1)-1 =   2*lid
	// bi = 1*(2*lid+2)-1 = 1+2*lid
	// loc[1+2*lid] += loc[2*lid]
	// loc[1] = loc[1] + loc[0]
	// loc[3] = loc[3] + loc[2]
	// loc[31] = loc[31] + loc[30]
	// offset = 2
	// ...
	// d = 8
	// lid < 8
	// ai = 2*(2*lid+1)-1 = 4*lid+1
	// bi = 2*(2*lid+2)-1 = 4*lid+3
	// loc[4*lid+3] += loc[4*lid+1]
	// loc[3] = loc[3] + loc[1]
	// loc[7] = loc[7] + loc[5]
	// offset = 4
	// ...
	// d = 2
	// offset = 16
	// ...
	// d = 1
	// lid < 1 
	// ai = 16*(2*lid+1)-1 = 32*lid+15
	// bi = 16*(2*lid+2)-1 = 32*lid+31
	// loc[32*lid+31] = loc[32*lid+31] + loc[32*lid+15]
	// loc[31] = loc[31] + loc[15]   // FINAL SUM
}

		// sum 32 elements
		// If float4, we really have 128 floats so we need 64 threads, 
		// so we could use two warps. 
		// a[0] = a[0] + a[1], 
		// a[2] = a[2] + a[3] 
		// a[4] = a[4] + a[5], ..., 
		// a[30] = a[30]+a[31]
		// a[2*i] += a[2*i+1] (i < 5)

		// a[0] = a[0] + a[2]
		// a[4] = a[4] + a[6]
		// a[8] = a[8] + a[10]
		// ...
		// a[28] = a[28] + a[30]
		// a[4*i] += a[4*i+2] (i < 4)

		// a[0] = a[0] + a[4]
		// a[8] = a[8] + a[12]
		// a[16] = a[16] + a[20]
		// a[24] = a[24] + a[28]
		// a[8*i] += a[8*i+4]  (i < 3)

		// a[0] = a[0] + a[8]
		// a[16] = a[16] + a[24]
		// a[16*i] += a[16*i+8]  (i < 2)
		
		// a[0] = a[0] + a[16]
		// a[i] += a[16]   (i < 1)
		// DONE


		//if (lid < 27) {  // only 27 neighbors


#endif

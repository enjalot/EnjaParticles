// HOW TO INCLUDE WHEN COMPILING? ?

#ifndef _CL_MACROS_H_
#define _CL_MACROS_H_


//---------------------------------------------------------------------- 
// Offsets into var_sorted array

#include "../variable_labels.h"

#define numParticles num

// copied from SPHSimLib code
#ifdef USE_TEX
//#define FETCH(a, t, i) tex1Dfetch(t##_tex, i)
#define FETCH(t, i) tex1Dfetch(t##_tex, i)
#define FETCH_NOTEX(a, t, i) a.t[i]
#define FETCH_FLOAT3(a,t,i) make_float4(FETCH(a,t,i))
#define FETCH_MATRIX3(a,t,i) tex1DfetchMatrix3(t##_tex,i)
#define FETCH_MATRIX3_NOTEX(a,t,i) a.t[i]
#else
//#define FETCH(a, t, i) a.t[i]
#define FETCH(t, i) t[i]
#define FETCH_VAR(t, i, ivar) t[i+ivar*numParticles]
#define FETCH_VEL(t, i) t[i+VEL*numParticles]
#define FETCH_DENS(t, i) t[i+DENS*numParticles]
#define FETCH_FOR(t, i) t[i+FOR*numParticles]
#define FETCH_ACC(t, i) t[i+ACC*numParticles]
#define FETCH_POS(t, i) t[i+POS*numParticles]

// change nb of neighbors in GE_SPH::setArrays as well
#define neigh(i, j)  index_neigh[j+50*i] // max of 50 neighbors

#define density(i) 			vars_sorted[i+DENS*numParticles].x
#define pos(i) 				vars_sorted[i+POS*numParticles]
#define vel(i) 				vars_sorted[i+VEL*numParticles]
#define force(i) 			vars_sorted[i+FOR*numParticles]
// accessing density and color requires two memory access. 
// Could be more efficient if stored in local point-based array
#define density_denom(i)   	vars_unsorted[i+DENS_DENOM*numParticles].y
#define color(i)    		vars_sorted[i+COL*numParticles].x
#define surf_tens(i)    	vars_sorted[i+SURF_TENS*numParticles]
#define color_normal(i)    	vars_sorted[i+NORMAL*numParticles]
#define veleval(i)    		vars_sorted[i+VELEVAL*numParticles]
#define xsph(i)    			vars_sorted[i+XSPH*numParticles]

#define unsorted_density(i)   vars_unsorted[i+DENS     *numParticles].x
#define unsorted_pos(i) 	  vars_unsorted[i+POS      *numParticles]
#define unsorted_vel(i) 	  vars_unsorted[i+VEL      *numParticles]
#define unsorted_force(i) 	  vars_unsorted[i+FOR      *numParticles]
#define unsorted_density_denom(i)   vars_unsorted[i+DENS_DENOM*numParticles].y
#define unsorted_color(i)     vars_unsorted[i+COL      *numParticles].x
#define unsorted_surf_tens(i) vars_unsorted[i+SURF_TENS*numParticles]
#define unsorted_color_normal(i)    vars_unsorted[i+NORMAL*numParticles]
#define unsorted_veleval(i)    vars_sorted[i+VELEVAL*numParticles]
#define unsorted_xsph(i)    	vars_sorted[i+XSPH*numParticles]

//#define FETCH_NOTEX(a, t, i) a.t[i]
#define FETCH_NOTEX(t, i) t[i]
//#define FETCH_FLOAT3(a,t,i) make_float4(FETCH(a,t,i))
#define FETCH_FLOAT3(t,i) make_float4(FETCH(t,i))
#define FETCH_MATRIX3(a,t,i) a.t[i]
#define FETCH_MATRIX3_NOTEX(a,t,i) a.t[i]
//#define FETCH(a, t, i) (a + __mul24(i,sizeof(a)) + (void*)offsetof(a, t))
#endif


// FOR DEBUGGING
#define DUMMY_ARGS  , __global float4* clf, __global int4* cli, __global int* index_neigh
#define       ARGS  , clf, cli, index_neigh



#endif

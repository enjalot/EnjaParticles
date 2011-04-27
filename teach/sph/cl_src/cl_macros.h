// HOW TO INCLUDE WHEN COMPILING? ?

#ifndef _CL_SPH_MACROS_H_
#define _CL_SPH_MACROS_H_

#include "cl_common_macros.h"



//All of the below is depracated and not used



//---------------------------------------------------------------------- 
// Offsets into var_sorted array

#define DENS 0
#define POS 1
#define VEL 2
#define FOR 3
#define COL 4
#define DENS_DENOM 5
#define SURF_TENS 6
#define NORMAL 7
#define VELEVAL 8
#define XSPH 9

//#define numParticles num
#define numParticles sphp->max_num

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
//#define surface(i) 			vars_sorted[i+COL*numParticles]
#define pos(i) 				vars_sorted[i+POS*numParticles]
#define vel(i) 				vars_sorted[i+VEL*numParticles]
#define veleval(i)    		vars_sorted[i+VELEVAL*numParticles]
#define force(i) 			vars_sorted[i+FOR*numParticles]
#define xsph(i)    			vars_sorted[i+XSPH*numParticles]
// accessing density and color requires two memory access. 
// Could be more efficient if stored in local point-based array
/*
#define density_denom(i)   	vars_sorted[i+DENS_DENOM*numParticles].y
#define color(i)    		vars_sorted[i+COL*numParticles].x
#define surf_tens(i)    	vars_sorted[i+SURF_TENS*numParticles]
#define color_normal(i)    	vars_sorted[i+NORMAL*numParticles]
*/

#define unsorted_density(i)   		vars_unsorted[i+DENS        *numParticles].x
#define unsorted_pos(i) 	  		vars_unsorted[i+POS         *numParticles]
#define unsorted_vel(i) 	  		vars_unsorted[i+VEL         *numParticles]
#define unsorted_force(i) 	  		vars_unsorted[i+FOR         *numParticles]
#define unsorted_density_denom(i)   vars_unsorted[i+DENS_DENOM  *numParticles].y
#define unsorted_color(i)     		vars_unsorted[i+COL         *numParticles].x
#define unsorted_surf_tens(i) 		vars_unsorted[i+SURF_TENS   *numParticles]
#define unsorted_color_normal(i)    vars_unsorted[i+NORMAL      *numParticles]
#define unsorted_veleval(i)    		vars_unsorted[i+VELEVAL     *numParticles]
#define unsorted_xsph(i)    		vars_unsorted[i+XSPH        *numParticles]



#endif


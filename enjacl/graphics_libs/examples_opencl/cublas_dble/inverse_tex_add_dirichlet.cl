
#ifndef _INVERSE_DIAG_3D_ADD_KERNEL_CU_
#define _INVERSE_DIAG_3D__ADD_KERNEL_CU_

#include "float_type.h"

//#include "macros.h"
#define blockIdxx  get_group_id(0)
#define blockIdxy  get_group_id(1)
#define blockIdxz  get_group_id(2)

#define gridDimx get_num_groups(0)

#define blockDimx get_local_size(0);
#define blockDimy get_local_size(1);
#define blockDimz get_local_size(2);

#define threadIdxx get_local_id(0);
#define threadIdxy get_local_id(1);
#define threadIdxz get_local_id(2);

#define K_P(i)   KP_t[i]
#define K_M(i)   KM_t[i]
#define M_P(i)   MP_t[i]
#define M_M(i)   MM_t[i]
#define L_P(i)   LP_t[i]
#define L_M(i)   LM_t[i]

#define __syncthreads() barrier(CLK_LOCAL_MEM_FENCE)

#define USE_TEX 0

#define OPENCL


#ifdef OPENCL
#define ARGS_OCL   , nx, ny, nz, nb_zblocks
#define ARGS_OTE   , KM_t, KP_t, LM_t, LP_t, MM_t, MP_t 
#define ARGS_CL    , int nx, int ny, int nz, int nb_zblocks
#define ARGS_TE    , __global FLOAT* KM_t, __global FLOAT* KP_t, __global FLOAT* LM_t, __global FLOAT* LP_t, __global FLOAT* MM_t, __global FLOAT* MP_t 
#else
#ifdef CUDA
#define ARGS_OCL  
#define ARGS_OTE 
#define ARGS_CL 
#define ARGS_TE 
#endif
#endif

#ifdef DEBUG
#define ARGS_GLOB   ARGS_CL, FLOAT4* deb
#define ARGS_DUMMY  ARGS_CL  ARGS_TE,  FLOAT4* deb
#define ARGS_REAL   ARGS_OCL ARGS_OTE, deb
#else
#define ARGS_GLOB   ARGS_CL
#define ARGS_DUMMY  ARGS_CL  ARGS_TE
#define ARGS_REAL   ARGS_OCL ARGS_OTE
#endif


/*---------------------------------------------------------------------- */
/* setup a texture for epsilon */

void inverse_diag_3d_z_kernel(__global FLOAT* y, int i_block  ARGS_DUMMY);
void inverse_diag_3d_all_z_kernel(__global FLOAT* y ARGS_DUMMY);


/*---------------------------------------------------------------------- */
__kernel void inverse_tex_add_dirichlet(__global FLOAT* y  ARGS_DUMMY)
{
	#if 0
	__syncthreads();
	return;
	__syncthreads();
	y[1] = ny;
	__syncthreads();
	y[2] = get_local_size(0);
	__syncthreads();
	y[3] = get_local_size(1);
	__syncthreads();
	y[4] = get_local_size(2);
	__syncthreads();
	y[5] = get_global_size(0);
	__syncthreads();
	y[6] = get_global_size(1);
	__syncthreads();
	y[7] = get_global_size(2);
	__syncthreads();
	#endif

	inverse_diag_3d_all_z_kernel(y ARGS_REAL);
}
/* ---------------------------------------------------------------------- */
void inverse_diag_3d_all_z_kernel(__global FLOAT* y ARGS_DUMMY)
{
/* CONSTANTS: nb_zblocks */


	for (int i_block=0; i_block < nb_zblocks; i_block++) {
		/* if (i_block != 0) continue; // TEMPORARY */
		inverse_diag_3d_z_kernel(y, i_block ARGS_REAL);

		__syncthreads();
	}
}
/*---------------------------------------------------------------------- */
void inverse_diag_3d_z_kernel(__global FLOAT* y, int zblock ARGS_DUMMY)
{
	/*
	   xi,yi,zi: coordinates in 2D grid
	   reduce register to 30 by using volatile and moving these
	   higher up in the code
	*/

#if 0
	#define VOLATILE volatile
#else
	#define VOLATILE
#endif

	/* only access points to be updated (interior points if Dirichlet BC) */
	/* include boundary points if Neuman BC */
	//VOLATILE int xi = (get_group_id(0) * get_local_size(0)) + get_local_id(0);
	//VOLATILE int yi = (get_group_id(1) * get_local_size(1)) + get_local_id(1);
	VOLATILE int zi = (zblock    * get_local_size(2))       + get_local_id(2);
	if (zi == 0 || zi == nz-1) return;

	VOLATILE int xi = get_global_id(0);
	if (xi == 0 || xi == nx-1) return;
	VOLATILE int yi = get_global_id(1);
	if (yi == 0 || yi == ny-1) return;

	__syncthreads();

		FLOAT kp  = K_P(xi);
		FLOAT km  = K_M(xi);
		FLOAT lp  = L_P(yi);
		FLOAT lm  = L_M(yi);
		FLOAT mp  = M_P(zi);
		FLOAT mm  = M_M(zi);

	__syncthreads();


// avoid use of registers

#define i threadIdx.x
#define j threadIdx.y
#define k threadIdx.z

/* 3D case */
	FLOAT a;
	a =     km + kp;
	a = a + lm + lp;
	a = a + mm + mp;

/* 
  DIRICHLET CONDITIONS
     xi=nx-1 is the last point. The value of xi(nx+nx*j) is assumed to be given
*/

	y[xi+nx*(yi+ny*zi)] = 1./a;    // correct version
	//y[xi+nx*(yi+ny*zi)] = a;    // non-correct version

#ifdef DEBUG
	#if 1
	int glob = xi+nx*(yi+ny*zi);
	deb[glob].x = a; /* km; //blockDim.x; */
	deb[glob].y = lm+lp; //blockIdx.x;
	deb[glob].z = mm+mp; // zi + 100*yi + 10000*xi;
	deb[glob].w = a;
	/*deb[glob].w = zblocksz; // zi + 100*yi + 10000*xi; */
	/*deb[glob].z = k + 100*j + 10000*i; */
	/*deb[glob].w = nx; */
	/*deb[glob].w = threadIdx.x; */
	#endif
#endif

#undef i
#undef j
#undef k
}

#undef ARGS_GLOB
#undef ARGS_DUMMY
#undef ARGS_REAL
#undef ARGS_OCL
#undef ARGS_OTE
#undef ARGS_CL
#undef ARGS_TE

/*---------------------------------------------------------------------- */

#undef VOLATILE
/*  _INVERSE_DIAG_3D_ADD_KERNEL_CU_ */
#endif

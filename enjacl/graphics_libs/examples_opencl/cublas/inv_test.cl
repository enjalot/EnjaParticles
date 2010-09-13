
#ifndef _INVERSE_TEST_CL_
#define _INVERSE_TEST_CL_

#define OPENCL



/*---------------------------------------------------------------------- */
__kernel void inverse_diag_3d_kernel(__global float* y, __global float* z)
{
	int n = get_global_id(0);
	//z[n] = get_global_id(0);
	z[n] = y[n];
	return;


	#if 0
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
	return;
	#if 0
	inverse_diag_3d_all_z_kernel(y ARGS_REAL);
	#endif
}
/* ---------------------------------------------------------------------- */
#endif

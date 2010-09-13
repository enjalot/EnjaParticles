#ifndef _MAT_MAT_MUL_EL_KERNEL_CU_
#define _MAT_MAT_MUL_EL_KERNEL_CU_

__kernel void mat_mat_mul_el_kernel(__global float* res, __global float* x, __global float* y)
{
// set x = val on the radius rad (for all angles)
// assume that the grid is only in x and y directions
// assume that blockDim.x*gridDim.x + blockDim.y*gridDim.y = nx*ny

		int ix = get_global_id(0);
		res[ix] = x[ix]*y[ix];

	return;
}
//----------------------------------------------------------------------

// #ifndef _MAT_MAT_MUL_EL_KERNEL_CU_
#endif

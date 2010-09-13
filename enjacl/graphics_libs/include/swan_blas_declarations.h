#ifndef  _SWAN_BLAS_DECLARATIONS_H_
#define  _SWAN_BLAS_DECLARATIONS_H_

#include <array_opencl_1d.h>

#include "float_type.h"

extern "C++" {

void cublasSscal (int n, FLOAT alpha, cl_mem x, int incx);
void cublasScopy (int n, cl_mem xsrc, int incx, cl_mem ydest, int incy);
void cublasSaxpy (int n, FLOAT alpha, cl_mem x, int incx, cl_mem y, int incy);
FLOAT cublasSdot (int n, cl_mem x, int incx, cl_mem y, int incy);
void mat_mat_mul_el(cl_mem res, cl_mem a, cl_mem b, int nx, int ny, int nz);

void cublasSscalParams (int n, FLOAT alpha, FLOAT *x, int incx);
void cublasSvecvec (int n, FLOAT *x, FLOAT *y, FLOAT* result);

void matVecKernel(cl_mem a, cl_mem b, int nx, int ny, int nz, 
    int nb_zblocks, cl_mem KP, cl_mem KM, cl_mem LP, cl_mem LM, cl_mem MP, cl_mem MM);

void matVecDirichlet(cl_mem a, cl_mem b, int nx, int ny, int nz, 
    int nb_zblocks, cl_mem KP, cl_mem KM, cl_mem LP, cl_mem LM, cl_mem MP, cl_mem MM);

void inverseDiag3DKernel(cl_mem precond_d, 
	int nx, int ny, int nz, int nb_zblocks, 
	cl_mem KP, cl_mem KM, cl_mem LP, cl_mem LM, cl_mem MP, cl_mem MM);

void invAddDirichlet(cl_mem a, int nx, int ny, int nz, int nb_zblocks, 
		cl_mem KP, cl_mem KM, cl_mem LP, cl_mem LM, cl_mem MP, cl_mem MM);

void inverseTest(cl_mem a, cl_mem b, int nx, int ny, int nz);

};

#endif

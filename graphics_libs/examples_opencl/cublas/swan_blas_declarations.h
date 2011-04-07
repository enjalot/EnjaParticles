#include <array_opencl_1d.h>

extern "C++" {

float cublasSdot (int n, const float* x, int incx, const float* y, int incy);
void cublasSscal (int n, float alpha, cl_mem x, int incx);
void cublasScopy (int n, cl_mem xsrc, int incx, cl_mem ydest, int incy);
void cublasSaxpy (int n, float alpha, cl_mem x, int incx, cl_mem y, int incy);
float cublasSdot (int n, cl_mem x, int incx, cl_mem y, int incy);
void mat_mat_mul_el(cl_mem res, cl_mem a, cl_mem b, int nx, int ny, int nz);

void cublasSscalParams (int n, float alpha, float *x, int incx);
void cublasSvecvec (int n, float *x, float *y, float* result);

void matVecKernel(cl_mem a, cl_mem b, int nx, int ny, int nz, 
    int nb_zblocks, cl_mem KP, cl_mem KM, cl_mem LP, cl_mem LM, cl_mem MP, cl_mem MM);

void matVecDirichlet(cl_mem a, cl_mem b, int nx, int ny, int nz, 
    int nb_zblocks, cl_mem KP, cl_mem KM, cl_mem LP, cl_mem LM, cl_mem MP, cl_mem MM);

void inverseDiag3DKernel(cl_mem precond_d, 
	int nx, int ny, int nz, int nb_zblocks, 
	cl_mem KP, cl_mem KM, cl_mem LP, cl_mem LM, cl_mem MP, cl_mem MM);

void invAddDirichlet(cl_mem precond_d, 
	int nx, int ny, int nz, int nb_zblocks, 
	cl_mem KP, cl_mem KM, cl_mem LP, cl_mem LM, cl_mem MP, cl_mem MM);

void inverseTest(cl_mem a, cl_mem b, int nx, int ny, int nz);



#if 0
void inverseDiag3DMulKernel(cl_mem precond_d, int nx, int ny, int nz, 
    int nb_zblocks, int bx, int cx, 
	cl_mem KP, cl_mem KM, cl_mem LP, cl_mem LM, cl_mem MP, cl_mem MM);
#endif

#if 0
void matrixVecKernel(float* x, float* y, int nx, int ny, int nz, 
    int nb_zblocks, 
	ArrayCuda1D<float>& KP,
	ArrayCuda1D<float>& KM,
	ArrayCuda1D<float>& LP,
	ArrayCuda1D<float>& LM,
	ArrayCuda1D<float>& MP,
	ArrayCuda1D<float>& MM);
#endif
};

#include <stdlib.h>
#include <math.h>
#include <cl.h>
#include <string>
#include <array_opencl_1d.h>
#include "swan_blas_declarations.h"
#include <unistd.h>
// timing routines
#include <timege.h>

#include "float_type.h"

using namespace std;

#include "debug_macros.h"


int nx;
int ny;
int nz;

#include "clocks.h"


//----------------------------------------------------------------------
void tst_inv_tst(int ntests)
{
	int nx,ny,nz;
	nx = ny = nz = 128;
	int ntot = nx*ny*nz; // for macros

	ArrayOpenCL1D<FLOAT> a(nx,ny,nz);
	ArrayOpenCL1D<FLOAT> b(nx,ny,nz);
	a.setTo(4.);
	b.setTo(0.);
	a.copyToDevice();
	b.copyToDevice();

	/* ******************************************************* */

	/*** THERE IS AN ERROR: After calling kernel once, I can no longer
	   access array "a" (copyToHost() and copyToDevice() no longer work 
	****/

	int error = 0;

	#if 1
	for (int i=0; i < ntests; i++) {
		clock_inv_mat_vec.begin();
		inverseTest(a.getDevicePtr(), b.getDevicePtr(), nx, ny, nz);
		clock_inv_mat_vec.end();
	}
	#endif

	DBG_PRINT2(a.getDevicePtr(), "after inverseDiag3DKernel");
}
//----------------------------------------------------------------------
//----------------------------------------------------------------------
void tst_inv_diag_3d(int ntests)
{
#if 1
	int nx,ny,nz;
	int nb_zblocks, bx, cx;
	nx = ny = nz = 128;
	int ntot = nx*ny*nz; // for macros

	ArrayOpenCL1D<FLOAT> a(nx,ny,nz);
	a.setTo(3.);
	a.copyToDevice();
	nb_zblocks = nz / 2;

	/* ******************************************************* */
	// TEXTURES
	ArrayOpenCL1D<FLOAT> KP(nx), KM(nx);
	ArrayOpenCL1D<FLOAT> LP(ny), LM(ny);
	ArrayOpenCL1D<FLOAT> MP(nz), MM(nz);

	for (int i=0; i < nx; i++) {
		KP(i) = 2.2;
		KM(i) = 2.1;
	}
	for (int i=0; i < ny; i++) {
		LP(i) = 3.2;
		LM(i) = 3.1;
	}
	for (int i=0; i < nz; i++) {
		MP(i) = 4.2;
		MM(i) = 4.1;
	}

	KP.copyToDevice();
	KM.copyToDevice();
	LP.copyToDevice();
	LM.copyToDevice();
	MP.copyToDevice();
	MM.copyToDevice();

	/* ******************************************************* */

	/*** THERE IS AN ERROR: After calling kernel once, I can no longer
	   access array "a" (copyToHost() and copyToDevice() no longer work 
	****/

	int error = 0;

	for (int i=0; i < ntests; i++) {
		clock_inv_mat_vec.begin();
		inverseDiag3DKernel(a.getDevicePtr(), nx, ny, nz, nb_zblocks, 
			KP.getDevicePtr(), KM.getDevicePtr(), LP.getDevicePtr(), 
			LM.getDevicePtr(), MP.getDevicePtr(), MM.getDevicePtr());
		clock_inv_mat_vec.end();
	}

	//DBG_PRINT2(a.getDevicePtr(), "after inverseDiag3DKernel");

	a.copyToHost();
	int count = 0;

	for (int k=0; k < nz; k++) {
	for (int j=0; j < ny; j++) {
	for (int i=0; i < nx; i++) {
		FLOAT p = KP(i)+KM(i)+
		         +LP(j)+LM(j)+
		         +MP(k)+MM(k);
		p = 1. / p;
		if (fabs(a(i,j,k)-p) > 1.e-4 && count < 5) {
			printf("(%d,%d,%d) CPU: %g, GPU: %g\n", i,j,k,  p, a(i,j,k));
			error = -1;
			count++;
		}
	}}}

	if (error == 0) {
		printf("inverseDiag3DKernel test successful\n");
	} else {
		printf("inverseDiag3DKernel test failed\n");
	}

	return;

	#if 0
	inverseDiag3DMulKernel(a.getDevicePtr(), nx, ny, nz, nb_zblocks, bx,
	cx, KP.getDevicePtr(), KM.getDevicePtr(), LP.getDevicePtr(), 
	LM.getDevicePtr(), MP.getDevicePtr(), MM.getDevicePtr());

	DBG_PRINT2(a.getDevicePtr(), "after inverseDiag3DMulKernel");

	a.copyToHost();

	error = 0;

	for (int k=0; k < nz; k++) {
	for (int j=0; j < ny; j++) {
	for (int i=0; i < nx; i++) {
		FLOAT p = (KP(i)+KM(i))*
		          (LP(j)+LM(j))*
		          (MP(k)+MM(k));
		p = 1. / p;
		if (fabs(a(i,j,k)-p) > 1.e-4) {
			//printf("(%d,%d,%d) CPU: %g, GPU: %g\n", i,j,k,  p, a(i,j,k));
			error = -1;
		}
	}}}

	if (error == 0) {
		printf("inverseDiag3DMulKernel test successful\n");
	} else {
		printf("inverseDiag3DMulKernel test failed\n");
	}
	#endif

#if 0
//       -----------------------------------------------------
// Check that A*x kernel does not blow up. Cannot check solution
// accuracy since the matrix is built up the KP, ... matrices. 

	ArrayOpenCL1D<FLOAT> b(nx,ny,nz);
	b.setTo(-10.);
	b.copyToDevice();

	DBG_PRINT2(a.getDevicePtr(), "before matrixVecKernel");
	DBG_PRINT2(b.getDevicePtr(), "before matrixVecKernel");

	for (int i=0; i < ntests; i++) {
		clock_mat_vec.begin();
		matrixVecKernel(a.getDevicePtr(), b.getDevicePtr(), 
			nx, ny, nz, nb_zblocks, KP, KM, LP, LM, MP, MM);
		clock_mat_vec.end();
	}

	DBG_PRINT2(a.getDevicePtr(), "after matrixVecKernel");
	DBG_PRINT2(b.getDevicePtr(), "after matrixVecKernel");
#endif

#endif
}
//----------------------------------------------------------------------
void tst_mat_vec(int ntests)
{
printf("====........===\n");
#if 1
	int nx,ny,nz;
	int nb_zblocks, bx, cx;
	nx = ny = nz = 128;
	int ntot = nx*ny*nz; // for macros

	ArrayOpenCL1D<FLOAT> a(nx,ny,nz);
	a.setTo(3.);
	a.copyToDevice();
	nb_zblocks = nx / 2;

	ArrayOpenCL1D<FLOAT> b(nx,ny,nz);
	b.setTo(-10.);
	b.copyToDevice();


	/* ******************************************************* */
	// TEXTURES
	ArrayOpenCL1D<FLOAT> KP(nx), KM(nx);
	ArrayOpenCL1D<FLOAT> LP(ny), LM(ny);
	ArrayOpenCL1D<FLOAT> MP(nz), MM(nz);

	for (int i=0; i < nx; i++) {
		KP(i) = 2.2;
		KM(i) = 2.1;
	}
	for (int i=0; i < ny; i++) {
		LP(i) = 3.2;
		LM(i) = 3.1;
	}
	for (int i=0; i < nz; i++) {
		MP(i) = 4.2;
		MM(i) = 4.1;
	}

	KP.copyToDevice();
	KM.copyToDevice();
	LP.copyToDevice();
	LM.copyToDevice();
	MP.copyToDevice();
	MM.copyToDevice();

// Check that A*x kernel does not blow up. Cannot check solution
// accuracy since the matrix is built up the KP, ... matrices. 

	DBG_PRINT2(a.getDevicePtr(), "before matrixVecKernel");
	DBG_PRINT2(b.getDevicePtr(), "before matrixVecKernel");

	for (int i=0; i < ntests; i++) {
		clock_mat_vec.begin();
		matVecKernel(a.getDevicePtr(), b.getDevicePtr(), 
			nx, ny, nz, nb_zblocks, 
			KP.getDevicePtr(), KM.getDevicePtr(), 
			LP.getDevicePtr(), LM.getDevicePtr(), 
			MP.getDevicePtr(), MM.getDevicePtr());
		clock_mat_vec.end();
	}

	DBG_PRINT2(a.getDevicePtr(), "after matrixVecKernel");
	DBG_PRINT2(b.getDevicePtr(), "after matrixVecKernel");

#endif
}
//----------------------------------------------------------------------
void tst_mat_mat_mul_el(int ntests)
{
#if 1
	int nx,ny,nz;
	nx = ny = nz = 128;

	ArrayOpenCL1D<FLOAT>   a(nx,ny,nz); 
	ArrayOpenCL1D<FLOAT>   b(nx,ny,nz);
	ArrayOpenCL1D<FLOAT> res(nx,ny,nz); 

	a.setTo(4.);
	b.setTo(6.);
	res.setTo(-1.);

	a.copyToDevice();
	b.copyToDevice();

	for (int i=0; i < ntests; i++) {
		clock_mat_mul.begin();
		mat_mat_mul_el(res.getDevicePtr(), a.getDevicePtr(), b.getDevicePtr(), nx, ny, nz);
		clock_mat_mul.end();
	}

	res.copyToHost();

	int error = 0;
	int count = 0;

	for (int k=0; k < nz; k++) {
	for (int j=0; j < ny; j++) {
	for (int i=0; i < nx; i++) {
		if (fabs(res(i,j,k)-a(i,j,k)*b(i,j,k)) > 1.e-5 && count < 5) {
			printf("(%d,%d,%d) a*b=res, %g * %g =? %g\n", i,j,k,  a(i,j,k), b(i,j,k), res(i,j,k));
			error = -1;
			count++;
		}
	}}}

	if (error == 0) {
		printf("mat_mat_mul_el test successful\n");
	} else {
		printf("mat_mat_mul_el test failed\n");
	}
#endif
}
//----------------------------------------------------------------------
void tstSvecvec(int ntests)
{
#if 0
	printf("\nTEST: svecvec\n");

	int nx,ny,nz;
	nx = ny = nz = 128;
	int ntot = nx*ny*nz;

	ArrayOpenCL1D<FLOAT>   a(nx,ny,nz); // works with 128,128,10
	ArrayOpenCL1D<FLOAT>   b(nx,ny,nz); // not with 128,128,11
	ArrayOpenCL1D<FLOAT> res(nx,ny,nz); // not with 128,128,11

	b.setTo(3.0);
	a.setTo(2.5);
	res.setTo(0.);

	a.copyToDevice();
	b.copyToDevice();
	res.copyToDevice();

	printf("a.getSize= %d\n", a.getSize());
	printf("32*32*11= %d\n", 32*32*11);


	// b <- a 
	printf("a.getSize= %d\n", a.getSize());
	for (int i=0; i < ntests; i++) {
		clock_svecvec.begin();
 		cublasSvecvec(a.getSize(), a.getDevicePtr(), b.getDevicePtr(), res.getDevicePtr());
		clock_svecvec.end();
	}

	res.copyToHost();

	#if 1
	int count = 0;
	for (int k=0; k < 128; k++) {
	for (int j=0; j < 128; j++) {
	for (int i=0; i < 128; i++) {
		if (fabs(res(i,j,k)-a(i,j,k)*b(i,j,k)) < 1.e-3 && count < 5) continue;
		printf("i,j,k,a,b,res= %d,%d,%d,   %g, %g, %g\n", i,j,k, a(i,j,k), b(i,j,k), res(i,j,k));
		count++;
	}}}
	#endif

	//FLOAT BB= cublasSdot(a.getSize(), b.getDevicePtr(), 1, b.getDevicePtr(), 1);
	//FLOAT RR= cublasSdot(a.getSize(), a.getDevicePtr(), 1, a.getDevicePtr(), 1); 
	//printf("BB,RR= %g, %g\n", BB ,RR);
#endif
}
//----------------------------------------------------------------------
void tstScopy(int ntests)
{
#if 1
	printf("\nTEST: scopy\n");
	int nx,ny,nz;
	nx = ny = nz = 128;
	int ntot = nx*ny*nz;

	ArrayOpenCL1D<FLOAT> a(nx,ny,nz); // works with 32,32,10
	ArrayOpenCL1D<FLOAT> b(nx,ny,nz); // not with 32,32,11

	b.setTo(3.5);
	a.setTo(2.5);

	a.copyToDevice();
	b.copyToDevice();

	printf("a.getSize= %d\n", a.getSize());
	printf("32*32*11= %d\n", 32*32*11);


	CL cl;
	cl.setProfiling(true);

	// b <- a 
	for (int i=0; i < ntests; i++) {
		clock_scopy.begin();
 		cublasScopy(a.getSize(), a.getDevicePtr(), 1, b.getDevicePtr(), 1);
		clock_scopy.end();
	}
	//clock_scopy.end();
	b.copyToHost();

	int err = 0;
	int count = 0;

	for (int k=0; k < nz; k++) {
	for (int j=0; j < ny; j++) {
	for (int i=0; i < nx; i++) {
		if (fabs(a(i,j,k)-b(i,j,k)) > 1.e-5 && count < 5) {
			err = -1;
			printf("i,j,k,a,b= %d,%d,%d,   %g, %g\n", i,j,k, a(i,j,k), b(i,j,k));
			count++;
		}
	}}}

	if (err == 0) {
		printf("scopy test successful\n");
	} else { 
		printf("scopy test failed\n");
	}
#endif
}
//----------------------------------------------------------------------
void tstSaxpy(int ntests)
{
#if 1
	printf("\nTEST: saxpy\n");

	int nx,ny,nz;
	nx = ny = nz = 128;
	int ntot = nx*ny*nz; // for macros

	ArrayOpenCL1D<FLOAT> a(nx,ny,nz);
	ArrayOpenCL1D<FLOAT> b(nx,ny,nz);
	ArrayOpenCL1D<FLOAT> orig_b(nx,ny,nz);

	a.setTo(2.);
	b.setTo(1.);
	orig_b.setTo(1.);

	a.copyToDevice();
	b.copyToDevice();

	FLOAT alpha = 2.;

	// b <- b + alpha*a 
	// cannot run this test twice in a row without messing up the results
	for (int i=0; i < 1; i++) {
 		cublasSaxpy(a.getSize(), alpha, a.getDevicePtr(), 1, b.getDevicePtr(), 1);
	}

	b.copyToHost();

	int count = 0;
	int err = 0;
	for (int k=0; k < nz; k++) {
	for (int j=0; j < ny; j++) {
	for (int i=0; i < nx; i++) {
		if (fabs(alpha*a(i,j,k)+orig_b(i,j,k)-b(i,j,k)) > 1.e-5 && count < 5) {
			err = -1;
			printf("i,j,k,a,b= %d,%d,%d,   %g, %g\n", i,j,k, a(i,j,k), b(i,j,k));
			count++;
		}
	}}}

	for (int i=0; i < ntests; i++) {
		clock_saxpy.begin();
 		cublasSaxpy(a.getSize(), alpha, a.getDevicePtr(), 1, b.getDevicePtr(), 1);
		clock_saxpy.end();
	}

	if (err == 0) {
		printf("saxpy test successful\n");
	} else { 
		printf("saxpy test failed\n");
	}
#endif
}
//----------------------------------------------------------------------
void tstSdot(int ntests)
{
#if 1
	int nx,ny,nz;
	nx = ny = nz = 128;
	int ntot = nx*ny*nz; // for macros

	printf("\nTEST: sdot\n");
	ArrayOpenCL1D<FLOAT> a(nx,ny,nz);
	ArrayOpenCL1D<FLOAT> b(nx,ny,nz);

	a.setTo(1.);
	b.setTo(1.);

	a.copyToDevice();
	b.copyToDevice();

	FLOAT dot;

	//CL cl;

	for (int i=0; i < ntests; i++) {
		clock_sdot.begin();
 		dot = cublasSdot(a.getSize(), a.getDevicePtr(),1, b.getDevicePtr(), 1);
		//cl.waitForKernelsToFinish();
		clock_sdot.end();
	}
	printf("dot= %f\n", dot);

	if (fabs(dot-a.getSize()) > 1.e-5) {
		printf("sdot failed\n");
	} else {
		printf("sdot successful\n");
	}
	printf("===========================\n");
#endif
}
//----------------------------------------------------------------------
void tstSscal(int ntests)
{
#if 1
	printf("\nTEST: sscal\n");

	int nx,ny,nz;
	nx = ny = nz = 128;
	int ntot = nx*ny*nz; // for macros

	ArrayOpenCL1D<FLOAT> a(nx,ny,nz);
	ArrayOpenCL1D<FLOAT> b(nx,ny,nz);

	FLOAT alpha = 1.0;
	a.setTo(.02);
	b.setTo(.02*alpha);

	a.copyToDevice();
	b.copyToDevice();

	CL cl;
	cl.setProfiling(true);


	for (int i=0; i < ntests; i++) {
		clock_scal.begin(); // SCREWS CL UP
 		cublasSscal(a.getSize(), alpha, a.getDevicePtr(),1);
		clock_scal.end();
		//cl.waitForKernelsToFinish();
	}
	//clock_scal.end(); // SCREWS CL UP
	a.copyToHost();


	int error = 0;

	for (int k=0; k < nz; k++) {
	for (int j=0; j < ny; j++) {
	for (int i=0; i < nx; i++) {
		if (fabs(a(i,j,k)-b(i,j,k)) > 1.e-5) {
			error = -1;
		}
	}}}

	if (error == 0) {
		printf("sscal test successful\n");
	} else {
		printf("sscal test failed\n");
	}

	for (int i=0; i < 10; i++) {
		printf("a,b= %f, %f\n", a(i,0,0), b(i,0,0));
	}
#endif
}
//----------------------------------------------------------------------
void tstSscalParams(int ntests)
{
#if 0
	printf("\nTEST: sscalParams\n");

	int nx,ny,nz;
	nx = ny = nz = 128;
	int ntot = nx*ny*nz; // for macros

	ArrayOpenCL1D<FLOAT> a(nx,ny,nz);
	ArrayOpenCL1D<FLOAT> b(nx,ny,nz);

	FLOAT alpha = 1.;

	a.setTo(.02);
	b.setTo(.02*alpha);

    swanMemset(a.getDevicePtr(), 0, a.getSize()*sizeof(FLOAT)); 


	a.copyToDevice();

	for (int i=0; i < ntests; i++) {
		clock_scal_params.begin();
 		cublasSscalParams(a.getSize(), alpha, a.getDevicePtr(),1);
 		//cublasSscal(a.getSize(), alpha, a.getDevicePtr(),1);
		clock_scal_params.end();
	}
	a.copyToHost();

	int error = 0;

	for (int k=0; k < nz; k++) {
	for (int j=0; j < ny; j++) {
	for (int i=0; i < nx; i++) {
		if (fabs(a(i,j,k)-b(i,j,k)) > 1.e-5) {
			error = -1;
		}
	}}}

	if (error == 0) {
		printf("sscalParams test successful\n");
	} else {
		printf("sscalParams test failed\n");
	}

	for (int i=0; i < 10; i++) {
		printf("a= %f, %f\n", a(i), b(i));
	}
#endif
}
//----------------------------------------------------------------------
int main()
{
	nx = 128;
	ny = 128;
	nz = 128;

	int ntests = 40;

	//clock_ssdot = GE::Time("sdot", -1, 5);
	//clock_ssdot_cpu = GE::Time("sdot_cpu");
	//clock_ssdot_gpu = GE::Time("sdot_gpu");

	printf("current directory\n");
	size_t sz = 255;
	char* path = new char[sz];
	char* base_dir =  getenv("BASE_DIR");
	printf("cwd= %s\n", base_dir);

	// Do not know how to test this
	//tst_mat_vec(ntests);
	//tst_inv_diag_3d(ntests);
	//tst_inv_tst(ntests);
	//exit(0);

	#if 1
	tstSscal(ntests);
	#endif 

	#if 1
	tstSdot(ntests);
	tstScopy(ntests);
	tstSaxpy(ntests);
	tst_mat_vec(ntests);
	tst_mat_mat_mul_el(ntests);
	tst_inv_diag_3d(ntests);
	#endif


	//tstSscalParams(ntests);
	//tstSvecvec(ntests);

	clock_scal_params.print();

	printf("\n");
	clock_svecvec.print();
	clock_mat_vec.print();

	printf("\n");
	clock_mat_vec.print();
	clock_sdot.print();
	clock_scal.print();
	clock_scopy.print();
	clock_saxpy.print();
	clock_mat_mul.print();
	clock_inv_mat_vec.print();

	printf("\n");
	clock_mat_vec_cpu.print();
	clock_sdot_cpu.print();
	clock_scale_cpu.print();
	clock_scopy_cpu.print();
	clock_saxpy_cpu.print();
	clock_mat_mul_cpu.print();
	clock_inv_mat_vec_cpu.print();

	printf("\n");
	clock_mat_vec_gpu.print();
	clock_sdot_gpu.print();
	clock_scale_gpu.print();
	clock_scopy_gpu.print();
	clock_saxpy_gpu.print();
	clock_mat_mul_gpu.print();
	clock_inv_mat_vec_gpu.print();

	//clock_dbg.print();

	return 0;
}
//----------------------------------------------------------------------

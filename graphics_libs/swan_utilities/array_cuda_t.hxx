
#include <string.h>
#include <stdlib.h>
#include <typeinfo>
#include "array_cuda_t.h"

//#define EXIT_FAILURE  0
//#define EXIT_SUCCESS  -1

//----------------------------------------------------------------------
template <class T>
ArrayCudaT<T>::ArrayCudaT() {
	//printf("inside arrayCudaT consructor 1\n");
	init();
}
//----------------------------------------------------------------------
template <class T>
ArrayCudaT<T>::ArrayCudaT(T* array, const Vec3i& n_) : AF(array, n_) {
	//printf("array_cuda_t, inside arrayCudaT consructor 2\n");
	init();
}
//----------------------------------------------------------------------
template <class T>
ArrayCudaT<T>::ArrayCudaT(T* array, const Vec3i& n_, const Vec3i& nm_) :
    AF(array, n_, nm_) {
	//printf("inside arrayCudaT consructor 3\n");
	init();
}
//----------------------------------------------------------------------
template <class T>
ArrayCudaT<T>::ArrayCudaT(const Vec3i& n_) : AF(n_) {
	//printf("inside arrayCudaT consructor 4\n");
	init();
}
//----------------------------------------------------------------------
template <class T>
ArrayCudaT<T>::ArrayCudaT(const Vec3i& n_, const Vec3i& nm_) : AF(n_, nm_) {
	//printf("inside arrayCudaT consructor 4\n");
	init();
}
//----------------------------------------------------------------------
template <class T>
ArrayCudaT<T>::ArrayCudaT(int n1_, int n2_, int n3_, int n1m_, int n2m_, int n3m_) :
    AF(n1_, n2_, n3_, n1m_, n2m_, n3m_) {
	init();
}
//----------------------------------------------------------------------
template <class T>
ArrayCudaT<T>::ArrayCudaT(T* array, const int* n_) : AF(array, n_) {
	init();
}
//----------------------------------------------------------------------
template <class T>
ArrayCudaT<T>::ArrayCudaT(T* array, int n1_, int n2_, int n3_, int n1m_, int n2m_, int n3m_) : AF(array, n1_, n2_, n3_, n1m_, n2m_, n3m_) {
	init();
}
//----------------------------------------------------------------------
template <class T>
ArrayCudaT<T>::~ArrayCudaT()
{
    //printf("inside ArrayCudaT destructor\n");
}
//----------------------------------------------------------------------
template <class T>
void ArrayCudaT<T>::init()
{
	//printf("default nb channels: 1\n");
	nbChannels = 1; // default

	// check type of T (int or float or float2 or float 4)
	const char* tname = typeid(T).name();
	if (strcmp(typeid(float4).name(), tname) == 0) {
		nbChannels = 4;
	} else if (strcmp(typeid(float2).name(), tname) == 0) {
		nbChannels = 2;
	} else if (strcmp(typeid(float).name(), tname) == 0) {
		nbChannels = 1;
	} else if (strcmp(typeid(int).name(), tname) == 0) {
		nbChannels = 1;
	}
	//printf("*** enter ArrayCudaT::init, nbChannels= %d\n", nbChannels);

	//d_data = (CUdeviceptr) 0;
	//cu_array = (CUarray) 0;
	//printf("before call createDeviceArray\n");
	//this->createDeviceArray(nbChannels); // pure virtual called during the construction of the object? Illegal?
	//printf("after call createDeviceArray\n");

	//printf("*** exit init, nbChannels= %d\n", nbChannels);
}
//----------------------------------------------------------------------
#if 0
template <class T>
void ArrayCudaT<T>::createPlan2D()
{
	//printf("np2,np3= %d, %d\n", this->np2, this->np3);
	if (this->np1 == 2) {
	} else {
    	fft_status = cufftPlan2d(&plan, this->np1, this->np2, CUFFT_DATA_C2C); // complex to complex
	}
	if (fft_status != CUFFT_SUCCESS) {
		printf("error in createPlan2D\n");
	}
}
//----------------------------------------------------------------------
template <class T>
void ArrayCudaT<T>::createPlan1D(int nbBatches)
{
	//printf("np2,np3= %d, %d\n", this->np2, this->np3);
    fft_status = cufftPlan1d(&plan, this->np2, CUFFT_DATA_C2C, nbBatches); // complex to complex
	if (fft_status != CUFFT_SUCCESS) {
		printf("error in createPlan2D\n");
	}
}
//----------------------------------------------------------------------
template <class T>
void ArrayCudaT<T>::ifft2D(ArrayCudaT* B)
{
	if (B == 0) {
    	fft_status = cufftExecC2C(plan, (cufftComplex*) getDevicePtr(), (cufftComplex*) getDevicePtr(), CUFFT_INVERSE);
	} else {
    	fft_status = cufftExecC2C(plan, (cufftComplex*) getDevicePtr(), (cufftComplex*) B->getDevicePtr(), CUFFT_INVERSE);
	}
	if (fft_status != CUFFT_SUCCESS) {
		printf("error in ifft2D\n");
	}
}
//----------------------------------------------------------------------
template <class T>
void ArrayCudaT<T>::ifft2D(int offset, ArrayCudaT* B)
{
	int nbBytes = offset*sizeof(T);

	if (B == 0) {
    	fft_status = cufftExecC2C(plan, (cufftComplex*) (getDevicePtr()+nbBytes), (cufftComplex*) (getDevicePtr()+nbBytes), CUFFT_INVERSE);
	} else {
    	fft_status = cufftExecC2C(plan, (cufftComplex*) (getDevicePtr()+nbBytes), (cufftComplex*) B->getDevicePtr(), CUFFT_INVERSE);
	}
	if (fft_status != CUFFT_SUCCESS) {
		printf("error in ifft2D\n");
	}
}
//----------------------------------------------------------------------
//----------------------------------------------------------------------
template <class T>
void ArrayCudaT<T>::fft2D(ArrayCudaT* B)
{
	if (B == 0) {
    	fft_status = cufftExecC2C(plan, (cufftComplex*) getDevicePtr(), (cufftComplex*) getDevicePtr(), CUFFT_FORWARD);
	} else {
    	fft_status = cufftExecC2C(plan, (cufftComplex*) getDevicePtr(), (cufftComplex*) B->getDevicePtr(), CUFFT_FORWARD);
	}
	if (fft_status != CUFFT_SUCCESS) {
		printf("error in fft2D\n");
	}
}
//----------------------------------------------------------------------
template <class T>
void ArrayCudaT<T>::fft1D(ArrayCudaT* B) // ideally, batch should be the last dimension
{
	if (B == 0) {
    	fft_status = cufftExecC2C(plan, (cufftComplex*) getDevicePtr(), (cufftComplex*) getDevicePtr(), CUFFT_FORWARD);
	} else {
    	fft_status = cufftExecC2C(plan, (cufftComplex*) getDevicePtr(), (cufftComplex*) B->getDevicePtr(), CUFFT_FORWARD);
	}
	if (fft_status != CUFFT_SUCCESS) {
		printf("error in fft1D\n");
	}
}
//----------------------------------------------------------------------
template <class T>
void ArrayCudaT<T>::setPlan2D_of(ArrayCudaT& A)
{
	plan = A.getPlan2D();
}
//----------------------------------------------------------------------
template <class T>
void ArrayCudaT<T>::setPlan1D_of(ArrayCudaT& A)
{
	plan = A.getPlan1D();
}
//----------------------------------------------------------------------
template <class T>
cufftHandle ArrayCudaT<T>::getPlan2D()
{
	return plan;
}
//----------------------------------------------------------------------
template <class T>
cufftHandle ArrayCudaT<T>::getPlan1D()
{
	return plan;
}
#endif
//----------------------------------------------------------------------
#if 1
template <class T>
void ArrayCudaT<T>::setRandom_float2(float a, float b)
{
	float2* f = this->getDataPtr();
	float fac = 1. / (float) RAND_MAX;

    for (int i = 0; i < this->getSize(); i++) {
        f[i].x = (b-a)*fac*rand() + a;
        f[i].y = (b-a)*fac*rand() + a;
    }
}
//----------------------------------------------------------------------
template <class T>
void ArrayCudaT<T>::setRandom(float a, float b)
{
	float* f = this->getDataPtr();
	float fac = 1. / (float) RAND_MAX;

    for (int i = 0; i < this->getSize(); i++) {
        f[i] = (b-a)*fac*rand() + a;
    }
}
#endif
//----------------------------------------------------------------------
#if 0
template <class T>
void ArrayCudaT<T>::getType(int* nbComponents
	const char* tname = typeid(t).name();

	if (strcmp(typeid(float).name(), tname) == 0) {
#endif
//----------------------------------------------------------------------
#if 0
template <class T>
const char* ArrayCudaT<T>::getType()
{
	// check type of T 
	const char* tname = typeid(T).name();
	if (strcmp(typeid(float2).name(), tname) == 0) {
		return "float2";
	} else if (strcmp(typeid(float4).name(), tname) == 0) {
		return "float4";
	if (strcmp(typeid(int2).name(), tname) == 0) {
		return "int2";
	} else if (strcmp(typeid(int4).name(), tname) == 0) {
		return "int4";
	} else
		return getBaseType();
	}
}
#endif
//----------------------------------------------------------------------
template <class T>
void ArrayCudaT<T>::setToFloat2(float2 f) // { printf("cannot set float 2 (gordon)\n"); }
{
	int np = this->npts;
	T* data = this->data;

	for (int i=0; i < np; i++) {
		data[i].x = f.x;
		data[i].y = f.y;
	}
}
//----------------------------------------------------------------------
template <class T>
void ArrayCudaT<T>::setToFloat4(float4 f) // { printf("cannot set float 2 (gordon)\n"); }
{
	int np = this->npts;
	T* data = this->data;

	for (int i=0; i < np; i++) {
		data[i].x = f.x;
		data[i].y = f.y;
		data[i].z = f.z;
		data[i].w = f.w;
	}
}
//----------------------------------------------------------------------
template <class T>
void ArrayCudaT<T>::setToInt2(int2 f)
{
	int np = this->npts;
	T* data = this->data;

	for (int i=0; i < np; i++) {
		data[i].x = f.x;
		data[i].y = f.y;
	}
}
//----------------------------------------------------------------------
template <class T>
void ArrayCudaT<T>::setToInt4(int4 f) 
{
	int np = this->npts;
	T* data = this->data;

	for (int i=0; i < np; i++) {
		data[i].x = f.x;
		data[i].y = f.y;
		data[i].z = f.z;
		data[i].w = f.w;
	}
}
//----------------------------------------------------------------------

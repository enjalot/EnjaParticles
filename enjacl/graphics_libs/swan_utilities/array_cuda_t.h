//  May 2007
//  Author: Gordon Erlebacher
// 
// By default create an array on Cuda
//
//----------------------------------------------------------------------

#ifndef _ARRAY_CUDA_H_
#define _ARRAY_CUDA_H_

//#include <cuda.h>
//#include <cutil.h>
//#include <cublas.h>
//#include <cufft.h>
//
//GE #include <typeinfo>

// CUDA
// GE
//#include <host_defines.h>
//#include "vector_types.h" 

#include "ArrayT.h"
#include "Vec3i.h"
#include "swan_types.h"
#include "swan_defines.h"
//swan_types should be included from swan_defines.h through swan_stub.h

#define CUFFT_DATA_C2C CUFFT_C2C

#if 0
// HOW TO SPECIALIZE templatized classes? Ask Bruno
class ArrayCudaT<int>  : public ArrayT<int>
....
#endif

template <class T>
class ArrayCudaT : public ArrayT<T>
{
protected:
	typedef ArrayT<T> AF;
    //CUDA_ARRAY_DESCRIPTOR desc;
	//CUDA_MEMCPY2D copyParam;
    //CUdeviceptr d_data; // 1D or 2D array

	// FOR SOME REASON, this generates an error
    //__align__(256) T*  d_data; // 1D or 2D array of type T (not a cudaArray) (on the device)
    __align__(128) T*  d_data; // 1D or 2D array of type T (not a cudaArray) (on the device)
    //CUarray cu_array;
	float* d_blas;
	unsigned int pitch; // used in copy functions. Also might be required as
	           // arguments to entry.run(). Leads to complications if the 
			   // pitch is actually required and I am using polymorphism, 
			   // unless a downcast is used (not good practice)
			   // Alternative: include "pitch" in base class although not used. 
			   // I guess this is ok since the pitch for 1D array is zero, and 
			   // the pitch for Cuda array is zero (not used). It would be better
			   // to only use Arrays in arguments to entry.run(), which are expanded
			   // into arguments. That puts restrictions on the structure of kernel
			   // programs, not necessarily desirable. 
	unsigned int readSize;  // can also be 8 or 16
	unsigned int status;
    //cufftHandle plan;
	//cufftResult fft_status;
	int nbChannels;

public:
	ArrayCudaT();
    ArrayCudaT(T* array, const Vec3i& n_);
    ArrayCudaT(T* array, const Vec3i& n_, const Vec3i& nm_);
    ArrayCudaT(const Vec3i& n_);
    ArrayCudaT(const Vec3i& n_, const Vec3i& nm_);
    ArrayCudaT(int n1_, int n2_=1, int n3_=1, int n1m_=0, int n2m_=0, int n3m_=0);
    ArrayCudaT(T* array, const int* n_);
    ArrayCudaT(T* array, int n1_, int n2_, int n3_=1, int n1m_=0, int n2m_=0, int n3m_=0);
    virtual ~ArrayCudaT();
	void init();

    // The next four routines are poor programming, but useful functions. Perhaps they
    // should be typdefs, or macros. USE with caution. 

    // only works if type T is float2, float3, float4
    void setToFloat2(float2 f); // { printf("cannot set float 2 (gordon)\n"); }

    // only works if type T is float4
    void setToFloat4(float4 f); // { printf("cannot set float 2 (gordon)\n"); }

    // only works if type T is int2, int3, int4
    void setToInt2(int2 f); // { printf("cannot set float 2 (gordon)\n"); }

    // only works if type T is int4
    void setToInt4(int4 f); // { printf("cannot set float 2 (gordon)\n"); }

	// Must be implemented in each subclass
	virtual void copyToDevice() = 0;
	virtual void copyToHost() = 0;
	//virtual void createDeviceArray(int nbChannels = 1) = 0; // 1D, 2D, cuda
	// I should really use the pure virtual, but ... in the meantime ...


private:
	virtual void createDeviceArray(int nbChannels) = 0;
	// Channels are not required
	virtual void createDeviceArray() = 0;
	//virtual void createDeviceArray(int nbChannels = 1) {
		//printf("arrayCudaT: inside createDeviceaArray\n");
	//} // 1D, 2D, cuda

	// QUESTION: pitch is only needed for 2D arrays. 

	// also need routines that create an array out of a subset of the base array stored in ArrayCudaT
	//CUarray& getCudaArray() { 
		//printf("cu_array= %d\n", cu_array);
		//return cu_array;
	//}

public:
	T* getDevicePtr() {
		return d_data;
	}

	T** getDeviceHdl() {
		return &d_data;
	}

	// for 2D arrays
	unsigned int getPitch() { return pitch; }

	float* getBlasPtr() { return d_blas; }

	// inverse fft using an array that starts at offset elements from the base (base = getDataPtr())
	// if the fft is not in place (B != 0), the result starts at B->getDataPtr()
	//void ifft2D(int offset, ArrayCudaT* B=0); // fft in place

	//void ifft2D(ArrayCudaT* B=0);
	//void fft2D(ArrayCudaT* B=0);
	//void fft1D(ArrayCudaT* B=0);
	//void createPlan1D(int nbBatches);
	//void createPlan2D();
	//void setPlan1D_of(ArrayCudaT& A);
	//void setPlan2D_of(ArrayCudaT& A);
	//void setPlan1D(cufftHandle plan_) { this->plan = plan_; }
	//void setPlan2D(cufftHandle plan_) { this->plan = plan_; }
	//cufftHandle getPlan2D();
	//cufftHandle getPlan1D();
	int getNbChannels() {
		return nbChannels;
	}
	virtual void clear() {printf("inside array_cuda_t::clear();\n"); } // clear all elements to zero

	// Use specialized classes for this
	// How to handle int, int8, float, half, etc. using templates?
	void setRandom(float a=0., float b=1.);
	void setRandom_float2(float a=0., float b=1.);
	//virtual const char* getType();

	//uses function from superclass
	//virtual char* getBaseType(); 

   // status = cublasAlloc(n2, sizeof(d_A[0]), (void**) &d_A);
};

#include "array_cuda_t.hxx"

#endif

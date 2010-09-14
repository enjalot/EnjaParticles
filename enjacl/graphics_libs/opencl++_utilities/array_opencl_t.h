//  June 2010
//  Author: Gordon Erlebacher
// 
//----------------------------------------------------------------------

#ifndef _ARRAY_CL_T_H_
#define _ARRAY_CL_T_H_

// OpenCL

#include "ArrayT.h"
#include "Vec3i.h"

#include "cl.h" // opencl interface to arrays

template <class T>
class ArrayOpenCLT : public ArrayT<T>
{
protected:
	typedef ArrayT<T> AF;
	CL cl;

    //__align__(128) T*  d_data; // 1D or 2D array of type T (not a cudaArray) (on the device)
    //T*  d_data; // 1D or 2D array of type T (not a cudaArray) (on the device)
	cl_mem d_data;
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
	int nbChannels;

public:
	ArrayOpenCLT();
    ArrayOpenCLT(T* array, const Vec3i& n_);
    ArrayOpenCLT(T* array, const Vec3i& n_, const Vec3i& nm_);
    ArrayOpenCLT(const Vec3i& n_);
    ArrayOpenCLT(const Vec3i& n_, const Vec3i& nm_);
    ArrayOpenCLT(int n1_, int n2_=1, int n3_=1, int n1m_=0, int n2m_=0, int n3m_=0);
    ArrayOpenCLT(T* array, const int* n_);
    ArrayOpenCLT(T* array, int n1_, int n2_, int n3_=1, int n1m_=0, int n2m_=0, int n3m_=0);
    virtual ~ArrayOpenCLT();
	void init();

    // The next four routines are poor programming, but useful functions. Perhaps they
    // should be typdefs, or macros. USE with caution. 

	#if 0
    // only works if type T is float2, float3, float4
    void setToFloat2(float2 f); // { printf("cannot set float 2 (gordon)\n"); }

    // only works if type T is float4
    void setToFloat4(float4 f); // { printf("cannot set float 2 (gordon)\n"); }

    // only works if type T is int2, int3, int4
    void setToInt2(int2 f); // { printf("cannot set float 2 (gordon)\n"); }

    // only works if type T is int4
    void setToInt4(int4 f); // { printf("cannot set float 2 (gordon)\n"); }
	#endif

	// Must be implemented in each subclass
	virtual void copyToDevice() = 0;
	virtual void copyToHost() = 0;
	//virtual void createDeviceArray(int nbChannels = 1) = 0; // 1D, 2D, cuda
	// I should really use the pure virtual, but ... in the meantime ...


private:
	virtual void createDeviceArray(int nbChannels) = 0;
	// Channels are not required
	virtual void createDeviceArray() = 0;

	// QUESTION: pitch is only needed for 2D arrays. 

	// also need routines that create an array out of a subset of the base array stored in ArrayOpenCLT

public:
	cl_mem getDevicePtr() {
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
	//void ifft2D(int offset, ArrayOpenCLT* B=0); // fft in place

	int getNbChannels() {
		return nbChannels;
	}
	virtual void clear() {printf("inside array_cuda_t::clear();\n"); } // clear all elements to zero

	// Use specialized classes for this
	// How to handle int, int8, float, half, etc. using templates?
	//void setRandom(float a=0., float b=1.);
	//void setRandom_float2(float a=0., float b=1.);
};

#include "array_opencl_t.hxx"

#endif

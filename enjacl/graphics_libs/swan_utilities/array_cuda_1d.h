#ifndef _ARRAY_CUDA_1D_H_
#define _ARRAY_CUDA_1D_H_

#include "array_cuda_t.h"

class Vec3i;

/// How to use pitch in ArrayCuda2D. Ok

//----------------------------------------------------------------------
template <class T>
class ArrayCuda1D : public ArrayCudaT<T>
{
private: 
	//const char* isLocked;

public:
    ArrayCuda1D(T* array, int n1_, int n2_=1, int n3_=1, int n1m_=0, int n2m_=0, int n3m_=0); // use memory (locked before calling the class)
    ArrayCuda1D(T* array, Vec3i& n_); // use memory (locked before calling the class)
	ArrayCuda1D(const Vec3i& n_);
    ArrayCuda1D(int n1_, int n2_=1, int n3_=1, int n1m_=0, int n2m_=0, int n3m_=0);
	ArrayCuda1D(int n1_, const char* locked);
	~ArrayCuda1D();
	void copyToDevice();
	void copyToHost();
	void clear(); // clear array zero on the device
	//virtual void print(const char* msg, int xo, int yo, int w, int h);
	//virtual const char* getType();


private:
	void createDeviceArray(int nbChannels); // 1D, 2D, cuda
	void createDeviceArray(); // 1D, 2D, cuda
};

//----------------------------------------------------------------------

#include "array_cuda_1d.hxx"

#endif


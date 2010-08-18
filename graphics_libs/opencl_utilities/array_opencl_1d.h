#ifndef _ARRAY_OPENCL_1D_H_
#define _ARRAY_OPENCL_1D_H_

#include "array_opencl_t.h"

class Vec3i;


//----------------------------------------------------------------------
template <class T>
class ArrayOpenCL1D : public ArrayOpenCLT<T>
{
private: 
	//const char* isLocked;

public:
    ArrayOpenCL1D(T* array, int n1_, int n2_=1, int n3_=1, int n1m_=0, int n2m_=0, int n3m_=0); // use memory (locked before calling the class)
    ArrayOpenCL1D(T* array, Vec3i& n_); // use memory (locked before calling the class)
	ArrayOpenCL1D(const Vec3i& n_);
    ArrayOpenCL1D(int n1_, int n2_=1, int n3_=1, int n1m_=0, int n2m_=0, int n3m_=0);
	ArrayOpenCL1D(int n1_, const char* locked);
	~ArrayOpenCL1D();
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

#include "array_opencl_1d.hxx"

#endif


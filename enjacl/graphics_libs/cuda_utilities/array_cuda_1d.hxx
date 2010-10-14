//#include "array_cuda_1d.h"
#include <device_types.h>
#include <vector_types.h>
#include <cutil.h>

//----------------------------------------------------------------------
extern "C" void copyToDeviceFromHost_1d(void* dst, const void* src, size_t count);
extern "C" void copyToHostFromDevice(void* dst, const void* src, size_t count);
extern "C" void cudaMalloc_1d(void** data , int nbBytes);
extern "C" void clear_1d(void* data, size_t count); // in bytes 
//----------------------------------------------------------------------

template <class T>
ArrayCuda1D<T>::ArrayCuda1D(T* array, int n1_, int n2_, int n3_, int n1m_, int n2m_, int n3m_) : ArrayCudaT<T>(array, n1_, n2_, n3_, n1m_, n2m_, n3m_)
{
	//printf("** array1D constructor: array, n1,n2,n3\n");
	//printf("new --- inside arrayCuda1D consructor array 2, nbChannels= %d\n", this->nbChannels);
	//this->isLocked = "";
	//printf("call createDeviceArray\n");
	createDeviceArray();
	//createDeviceArray(this->nbChannels); // pure virtual called during the construction of the object? Illegal?
}
//----------------------------------------------------------------------
template <class T>
ArrayCuda1D<T>::ArrayCuda1D(T* array, Vec3i& n_) : ArrayCudaT<T>(array, n_)
{
	//printf("inside arrayCuda1D consructor array 1\n");
	//this->isLocked = "";
	createDeviceArray(array);
	//createDeviceArray(this->nbChannels); // pure virtual called during the construction of the object? Illegal?
}
//----------------------------------------------------------------------
template <class T>
ArrayCuda1D<T>::ArrayCuda1D(const Vec3i& n_) : ArrayCudaT<T>(n_)
{
	//printf("inside arrayCuda1D consructor 1\n");
	//this->isLocked = "";
	//createDeviceArray(this->nbChannels); // pure virtual called during the construction of the object? Illegal?
	createDeviceArray();
}
//----------------------------------------------------------------------
template <class T>
ArrayCuda1D<T>::ArrayCuda1D(int n1_, const char* locked) :
    ArrayCudaT<T>(n1_, 1, 1, 0, 0, 0)
{
	//printf("inside arrayCuda1D consructor 2, locked\n");
	//this->isLocked = locked;
	//printf("channels: %d\n", this->nbChannels);
	createDeviceArray();
	//createDeviceArray(this->nbChannels); // pure virtual called during the construction of the object? Illegal?
}
//----------------------------------------------------------------------
template <class T>
ArrayCuda1D<T>::ArrayCuda1D(int n1_, int n2_, int n3_, int n1m_, int n2m_, int n3m_) :
    ArrayCudaT<T>(n1_, n2_, n3_, n1m_, n2m_, n3m_)
{
	//this->isLocked = "";
	//printf("new ** -- inside arrayCuda1D consructor 3\n");
	//printf("channels: %d\n", this->nbChannels);
	createDeviceArray();
	//createDeviceArray(this->nbChannels); // pure virtual called during the construction of the object? Illegal?
	//this->init();
}
//----------------------------------------------------------------------
template <class T>
ArrayCuda1D<T>::~ArrayCuda1D()
{
	cudaFree(this->d_data); // this-> is necessary due to templates
	//printf("inside arrayCuda1D destructor\n");
}
//----------------------------------------------------------------------
template <class T>
void ArrayCuda1D<T>::copyToDevice()
{
	//printf("d_data= %d, dataPtr= %d, size= %d\n", this->d_data, this->getDataPtr(), this->getSize()*
	    //sizeof(T));
	copyToDeviceFromHost_1d(this->d_data, (const void*) this->getDataPtr(), sizeof(T)*this->getSize());
}
//----------------------------------------------------------------------
#if 0
template <class T>
void ArrayCuda1D<T>::copyDtoDfrom(ArrayCuda1D<T>& fromArray)
{
// copy from device to device from the 1D array fromArray into this
// make sure that the pitch is equal to the array width. There might be 
// problems since the pitch is not obtained from 2D memory allocation
// However, if the array width is a multiple of 16 there should be no problems. 
// Dangerous since pitching is dependent on the specific card. 

    //printf("==== copyDtoD(from) ===\n");
	CUDA_MEMCPY2D& copyParam = this->copyParam;
	memset(&copyParam, 0, sizeof(copyParam));
	copyParam.dstMemoryType = CU_MEMORYTYPE_DEVICE;
	copyParam.dstPitch = this->pitch; // in array_cuda_t.h superclass
	//printf("dstPitch= %d\n", this->pitch);
	copyParam.dstDevice = this->d_data;
	copyParam.dstXInBytes = 0; // zero by default?
	copyParam.dstY = 0;

	//copyParam.srcMemoryType = CU_MEMORYTYPE_SYSTEM;
	copyParam.srcMemoryType = CU_MEMORYTYPE_DEVICE;
	copyParam.srcDevice = fromArray.getDevicePtr();
	copyParam.srcXInBytes = 0; // zero by default?
	copyParam.srcY = 0;
	//
	// copy might be inefficient if alignments not ok
	// do some tests with different size host arrays 
	int* dims = this->getDims();
	//copyParam.srcPitch = dims[1] * sizeof(T); // what if 3D array?
	copyParam.srcPitch = fromArray.getPitch();
	//printf("fromArray: srcPitch = %d\n", fromArray.getPitch());
	//printf("height: %d\n", dims[0]);

	copyParam.WidthInBytes = copyParam.srcPitch;
	copyParam.Height = dims[0];
    CU_SAFE_CALL(cuMemcpy2D(&copyParam));
	//printf("ArrayCuda2D: copyDtoD\n");
}
#endif
//----------------------------------------------------------------------
template <class T>
void ArrayCuda1D<T>::copyToHost()
{
	//CU_SAFE_CALL( cuMemcpyDtoH((void*) this->getDataPtr(), this->d_data, sizeof(T) * this->getSize()) );
	//cudaMemcpy((void*) this->getDataPtr(), this->d_data, sizeof(T)*this->getSize());
	copyToHostFromDevice((void*) this->getDataPtr(), this->d_data, sizeof(T)*this->getSize());
	//printf("copy to host: %d\n", (int) sizeof(T)*this->getSize());
}
//----------------------------------------------------------------------
template <class T>
void ArrayCuda1D<T>::createDeviceArray() // 1D, 2D, cuda
{
	int nbBytes = this->getSize() * sizeof(T);
	cudaMalloc_1d((void**) &this->d_data, nbBytes);

	if (this->d_data == 0) {
		printf("*** createDeviceArray:: cannot allocate memory (ArrayCuda1D) (%d) bytes\n", nbBytes);
		exit(0);
	}
}
//----------------------------------------------------------------------
#if 1
template <class T>
void ArrayCuda1D<T>::createDeviceArray(int nbChannels) // 1D, 2D, cuda
{
	int nbBytes = this->getSize() * sizeof(T);
	//printf("sizeof(T)= %d\n", (int) sizeof(T));
	//printf("nbBytes= %d\n", nbBytes);
	//printf("nbChannels= %d\n", nbChannels);
	cudaMalloc_1d((void**) &this->d_data, nbBytes);

	if (this->d_data == 0) {
		printf("*** createDeviceArray(%d):: cannot allocate memory (ArrayCuda1D) (%d) bytes\n", nbChannels, nbBytes);
		exit(0);
	}
	//printf("inside ArrayCuda1D<T>::createDeviceArray, d_data= %ld\n", (long) this->d_data);
}
#endif
//----------------------------------------------------------------------
#if 1
template <class T>
void ArrayCuda1D<T>::clear()
{
	// last argument is the number of 8 bit elements
	//printf("cuda1D::clear\n");
	//CU_SAFE_CALL(cuMemsetD8(this->getDevicePtr(), 0, this->getSize() * sizeof(T) / 2));
	clear_1d(this->getDevicePtr(), this->getSize() * sizeof(T)); // count bytes
}
#endif
//----------------------------------------------------------------------
#if 0
template <class T>
void ArrayCuda1D<T>::print(const char* msg, int xo, int yo, int w, int h)
{
	if (msg) {
	   printf("\n---------- %s -----------------\n", msg);
	}

	const char* type = getType();
	const int* dims = this->getDims();

	if (type == "float") {
		for (int j=0; j < h; j++) {
		for (int i=0; i < w; i++) {
			T& t = (*this)(xo+i, yo+j);
			printf("(%d,%d), %f\n", xo+i, yo+j, t);
		}}
	} else if (type == "float2") {
		float2* t = (this->data);
		for (int j=0; j < h; j++) {
		for (int i=0; i < w; i++) {
			int ix = (xo+i) + (yo+i)*dims[0];
			printf("(%d,%d), %f, %f\n", xo+i, yo+j, t[ix].x, t[ix].y);
		}}
	} else if (type == "float4") {
		float4* t = (this->data);
		for (int j=0; j < h; j++) {
		for (int i=0; i < w; i++) {
			int ix = (xo+i) + (yo+i)*dims[0];
			printf("(%d,%d), %f, %f, %f, %f\n", xo+i, yo+j, t[ix].x, t[ix].y, t[ix].z, t[ix].w);
		}}
	} else if (type == "int") {
		for (int j=0; j < h; j++) {
		for (int i=0; i < w; i++) {
			T& t = (*this)(xo+i, yo+j);
			printf("(%d,%d), %d\n", xo+i, yo+j, t);
		}}
	} else if (type == "int2") {
		for (int j=0; j < h; j++) {
		for (int i=0; i < w; i++) {
			T& t = (*this)(xo+i, yo+j);
			printf("(%d,%d), %d, %d\n", xo+i, yo+j, t.x, t.y);
		}}
	} else if (type == "int4") {
		for (int j=0; j < h; j++) {
		for (int i=0; i < w; i++) {
			T& t = (*this)(xo+i, yo+j);
			printf("(%d,%d), %d, %d, %d, %d\n", xo+i, yo+j, t.x, t.y, t.z, t.w);
		}}
	}

	printf("----------------------------------\n", msg);
}
#endif
//----------------------------------------------------------------------

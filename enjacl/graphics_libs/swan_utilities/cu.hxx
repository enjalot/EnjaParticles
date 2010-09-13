#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <typeinfo>

//#include <cuda.h>
//#include <cutil.h>

#include "ArrayT.h"

//#include <driver_types.h>

extern "C" void cudaConfigureCall_ge(Vec3i& g, Vec3i& b, size_t shared, int tokens);
extern "C" void cudaLaunch_ge(const char* entry); 
extern "C" void cudaSetupArgument_ge(void* arg, size_t count, size_t offset);

//----------------------------------------------------------------------
template <class T>
unsigned int CUentry::getOffset(T t)
{
	const char* tname = typeid(t).name();
	int sz;

	#if 0
	const char* Tname = typeid(T).name();
	printf("*** tname= %s, %s\n", tname, Tname);
	printf("typeid(void*).name= %s\n", typeid(void*).name());
	printf("typeid(float*).name= %s\n", typeid(float*).name());
	printf("typeid(float).name= %s\n", typeid(float).name());
	printf("typeid(int).name= %s\n", typeid(int).name());
	printf("typeid(unsigned int).name= %s\n", typeid(unsigned int).name());
	printf("size(void*)=  %d\n", sizeof(void*));
	printf("size(unsigned int)=  %d\n", sizeof(unsigned int));
	#endif

	if (strcmp(typeid(float2).name(), tname) == 0) {
		printf("type: float2, sz= %d\n", sizeof(float2));
		sz = 8* ((sizeof(unsigned int)-1) % 8 + 1); // boundary is multiple of 8
		cudaSetupArgument_ge(&t, sz, curOffset);
		return sz;
	} else if (strcmp(typeid(float).name(), tname) == 0) {
		printf("type: float, sz= %d\n", sizeof(float));
		sz = 8* ((sizeof(unsigned int)-1) % 8 + 1); // boundary is multiple of 8
		cudaSetupArgument_ge(&t, sz, curOffset);
		return sizeof(float);
	} else if (strcmp(typeid(int).name(), tname) == 0) {
		printf("type: int, sz= %d\n", sizeof(int));
		sz = 8* ((sizeof(unsigned int)-1) % 8 + 1); // boundary is multiple of 8
		cudaSetupArgument_ge(&t, sz, curOffset);
		return sz;
	} else if (strcmp(typeid(unsigned int).name(), tname) == 0) {
		printf("type: unsigned int, sz= %d\n", sizeof(unsigned int));
		sz = 8* ((sizeof(unsigned int)-1) % 8 + 1); // boundary is multiple of 8
		cudaSetupArgument_ge(&t, sz, curOffset);
		return sz;
	} else if (strcmp(typeid(float *).name(), tname) == 0) {
		printf("type: void*, sz= %d\n", sizeof(float*));
		sz = 8* ((sizeof(float*)-1) % 8 + 1); // boundary is multiple of 8
		cudaSetupArgument_ge(&t, sz, curOffset);
		return sz; // in previous versions, I did not have this line. DO NOT KNOW WHY
	} else if (strcmp(typeid(void *).name(), tname) == 0) {
		printf("type: void*, sz= %d\n", sizeof(void*));
		sz = 8* ((sizeof(void*)-1) % 8 + 1); // boundary is multiple of 8
		cudaSetupArgument_ge(&t, sz, curOffset);
		return sz; // in previous versions, I did not have this line. DO NOT KNOW WHY
	} else {
		//printf("type: CUarray");
		//cuParamSetArray(cuFunction, curOffset, t);
		//setParam(cuFunction, curOffset, t);
		//printf("sizeof(CUarray): %d\n", sizeof(CUarray));
//		return sizeof(CUarray);
		printf("TYPE NOT FOUND\n");
		exit(0);
		return 0;
	}
	//printf("*** getOffset, return 0\n");
	return 0;
}
//----------------------------------------------------------------------
#if 0
template <class T1>
void CUentry::run(T1 t1)
{
	curOffset = getOffset(t1);
	CU_SAFE_CALL(cuParamSetSize(cuFunction, curOffset));
	CU_SAFE_CALL(cuLaunchGrid(cuFunction, grid.x(), grid.y()));
}
//----------------------------------------------------------------------
template <class T1, class T2> 
void CUentry::run(T1 t1, T2 t2)
{
	const char* t1name = typeid(t1).name();
	const char* t2name = typeid(t2).name();

	curOffset  = getOffset(t1);
	curOffset += getOffset(t2);
	CU_SAFE_CALL(cuParamSetSize(cuFunction, curOffset));
	CU_SAFE_CALL(cuLaunchGrid(cuFunction, grid.x(), grid.y()));
}
//----------------------------------------------------------------------
template <class T1, class T2, class T3> 
void CUentry::run(T1 t1, T2 t2, T3 t3)
{
	curOffset  = getOffset(t1);
	curOffset += getOffset(t2);
	curOffset += getOffset(t3);
	CU_SAFE_CALL(cuParamSetSize(cuFunction, curOffset));
	CU_SAFE_CALL(cuLaunchGrid(cuFunction, grid.x(), grid.y()));
}
#endif
//----------------------------------------------------------------------
#if 0
template <class T1, class T2, class T3, class T4> void CUentry::run(T1 t1, T2 t2, T3 t3, T4 t4)
{
	// make sure alignment is 8 bytes on 64 bit machine and 4 bytes on 
	// 32-bit machine
	int shared = 0;
	int tokens = 0;
	//cudaConfigureCall(gridDim, blockDim, shared, tokens);
	cudaConfigureCall_ge(grid, block, shared, tokens);
	curOffset  = getOffset(t1);
	printf("curOffset= %d\n", curOffset);
	curOffset += getOffset(t2);
	printf("curOffset= %d\n", curOffset);
	curOffset += getOffset(t3);
	printf("curOffset= %d\n", curOffset);
	curOffset += getOffset(t4);
	printf("curOffset= %d\n", curOffset);
	cudaLaunch_ge("addKernel"); // assumes func_name is defined
	//cudaLaunch_ge(name.c_str()); // assumes func_name is defined
}
#endif
//----------------------------------------------------------------------
#if 0
template <class T1, class T2, class T3, class T4, class T5> 
void CUentry::run(T1 t1, T2 t2, T3 t3, T4 t4, T5 t5)
{
	curOffset  = getOffset(t1);
	curOffset += getOffset(t2);
	curOffset += getOffset(t3);
	curOffset += getOffset(t4);
	curOffset += getOffset(t5);
	printf("curOffset: %d\n", curOffset);
	printf("grid(xy)= %d, %d\n", grid.x(), grid.y());
	CU_SAFE_CALL(cuParamSetSize(cuFunction, curOffset));
	CU_SAFE_CALL(cuLaunchGrid(cuFunction, grid.x(), grid.y()));
}
//----------------------------------------------------------------------
template <class T1, class T2, class T3, class T4, class T5, class T6> 
void CUentry::run(T1 t1, T2 t2, T3 t3, T4 t4, T5 t5, T6 t6)
{
	curOffset  = getOffset(t1);
	curOffset += getOffset(t2);
	curOffset += getOffset(t3);
	curOffset += getOffset(t4);
	curOffset += getOffset(t5);
	curOffset += getOffset(t6);
	CU_SAFE_CALL(cuParamSetSize(cuFunction, curOffset));
	CU_SAFE_CALL(cuLaunchGrid(cuFunction, grid.x(), grid.y()));
}
//----------------------------------------------------------------------
template <class T1, class T2, class T3, class T4, class T5, class T6, class T7> 
void CUentry::run(T1 t1, T2 t2, T3 t3, T4 t4, T5 t5, T6 t6, T7 t7)
{
	curOffset  = getOffset(t1);
	curOffset += getOffset(t2);
	curOffset += getOffset(t3);
	curOffset += getOffset(t4);
	curOffset += getOffset(t5);
	curOffset += getOffset(t6);
	curOffset += getOffset(t7);
	CU_SAFE_CALL(cuParamSetSize(cuFunction, curOffset));
	CU_SAFE_CALL(cuLaunchGrid(cuFunction, grid.x(), grid.y()));
}
//----------------------------------------------------------------------
template <class T1, class T2, class T3, class T4, class T5, class T6, class T7, class T8> 
void CUentry::run(T1 t1, T2 t2, T3 t3, T4 t4, T5 t5, T6 t6, T7 t7, T8 t8)
{
	curOffset  = getOffset(t1);
	curOffset += getOffset(t2);
	curOffset += getOffset(t3);
	curOffset += getOffset(t4);
	curOffset += getOffset(t5);
	curOffset += getOffset(t6);
	curOffset += getOffset(t7);
	curOffset += getOffset(t8);
	CU_SAFE_CALL(cuParamSetSize(cuFunction, curOffset));
	CU_SAFE_CALL(cuLaunchGrid(cuFunction, grid.x(), grid.y()));
}
//----------------------------------------------------------------------
template <class T1, class T2, class T3, class T4, class T5, class T6, class T7, class T8, class T9> 
void CUentry::run(T1 t1, T2 t2, T3 t3, T4 t4, T5 t5, T6 t6, T7 t7, T8 t8, T9 t9)
{
	curOffset  = getOffset(t1);
	curOffset += getOffset(t2);
	curOffset += getOffset(t3);
	curOffset += getOffset(t4);
	curOffset += getOffset(t5);
	curOffset += getOffset(t6);
	curOffset += getOffset(t7);
	curOffset += getOffset(t8);
	curOffset += getOffset(t9);
	CU_SAFE_CALL(cuParamSetSize(cuFunction, curOffset));
	CU_SAFE_CALL(cuLaunchGrid(cuFunction, grid.x(), grid.y()));
}
//----------------------------------------------------------------------
template <class T1, class T2, class T3, class T4, class T5, class T6, class T7, class T8, class T9, class T10> 
void CUentry::run(T1 t1, T2 t2, T3 t3, T4 t4, T5 t5, T6 t6, T7 t7, T8 t8, T9 t9, T10 t10)
{
	curOffset  = getOffset(t1);
	curOffset += getOffset(t2);
	curOffset += getOffset(t3);
	curOffset += getOffset(t4);
	curOffset += getOffset(t5);
	curOffset += getOffset(t6);
	curOffset += getOffset(t7);
	curOffset += getOffset(t8);
	curOffset += getOffset(t9);
	curOffset += getOffset(t10);
	CU_SAFE_CALL(cuParamSetSize(cuFunction, curOffset));
	CU_SAFE_CALL(cuLaunchGrid(cuFunction, grid.x(), grid.y()));
}
#endif
//----------------------------------------------------------------------
template <class T> 
void CUentry::setGrid(ArrayCudaT<T>& A)
{
	int* dims = A.getDims();
	int gx = (dims[0]-1) / block.x + 1; // assumes that was is multiple of nbx
	int gy = (dims[1]-1) / block.y + 1;
	setGrid(gx, gy);
}
//----------------------------------------------------------------------
//CUentry::CUentry(const char* moduleName, const char* functionName, ArrayCudaT<float>& data)
template <class T>
CUentry::CUentry(ArrayCudaT<T>& data)
{
	setGridBlock(data);
}
//----------------------------------------------------------------------
template <class T>
void CUentry::setGridBlock(ArrayCudaT<T>& A)
{
	int* dims = A.getDims();
	setGridBlock(dims[0], dims[1]);
}
//----------------------------------------------------------------------
//----------------------------------------------------------------------
//----------------------------------------------------------------------
//----------------------------------------------------------------------

#ifndef _CU_H_
#define _CU_H_

//#include <cuda.h>
#include <cutil.h>
#include "Vec3i.h"
#include "array_cuda_t.h"
#include <string>
#include <vector_types.h>

//class ArrayFloat;

class CU {
private:
	//CUdevice cuDevice;
	//CUcontext cuContext;
	//CUmodule cuModule;
	//CUfunction cuFunction;
	char *image_filename;
	float angle;   // angle to rotate image by (in radians)

public:
	CU();
	//CUmodule loadModule(const char* moduleName);
	//CUfunction loadFunction(const char* functionName);
	//CUresult createDevice(); /// creates device 0
	//CUresult deleteDevice(); 
};

//----------------------------------------------------------------------

class CUentry {
private:
	dim3 grid;
	dim3 block;
	unsigned int curOffset;
	unsigned int offset;
	std::string name; // function name

public:
	CUentry();
	CUentry(int width, int height);

	template <class T>
	CUentry(ArrayCudaT<T>& A);

	void setName(std::string func_name) {
		name = func_name;
	}

	void init(const char* moduleName, const char* functionName);
	//void loadFunction(const char* functionName);
	void setBlock(int x, int y=1, int z=1);
	void setGrid(int x, int y=1, int z=1);
	dim3&  getGrid() { return grid; }
	dim3&  getBlock() { return block; }
	
	template <class T> void setGrid(ArrayCudaT<T>& A);
	template <class T> 
		unsigned int getOffset(T type_name);
	void run();
	//template <class T1> void run(T1 t1);
	//template <class T1, class T2> void run(T1 t1, T2 t2);
	//template <class T1, class T2, class T3> void run(T1 t1, T2 t2, T3 t3);
	template <class T1, class T2, class T3, class T4> 
		void run(T1 t1, T2 t2, T3 t3, T4 t4);
	//template <class T1, class T2, class T3, class T4, class T5> 
		//void run(T1 t1, T2 t2, T3 t3, T4 t4, T5 t5);
	//template <class T1, class T2, class T3, class T4, class T5, class T6> 
		//void run(T1 t1, T2 t2, T3 t3, T4 t4, T5 t5, T6 t6);
	//template <class T1, class T2, class T3, class T4, class T5, class T6, class T7> 
		//void run(T1 t1, T2 t2, T3 t3, T4 t4, T5 t5, T6 t6, T7 t7);
	//template <class T1, class T2, class T3, class T4, class T5, class T6, class T7, class T8> 
		//void run(T1 t1, T2 t2, T3 t3, T4 t4, T5 t5, T6 t6, T7 t7, T8 t8);
	//template <class T1, class T2, class T3, class T4, class T5, class T6, class T7, class T8, class T9> 
		//void run(T1 t1, T2 t2, T3 t3, T4 t4, T5 t5, T6 t6, T7 t7, T8 t8, T9 t9);
	//template <class T1, class T2, class T3, class T4, class T5, class T6, class T7, class T8, class T9, class T10> 
		//void run(T1 t1, T2 t2, T3 t3, T4 t4, T5 t5, T6 t6, T7 t7, T8 t8, T9 t9, T10 t10);

	//void paramSet(CUfunction fct, unsigned int offset, float2 val);
	//void paramSet(CUfunction fct, unsigned int offset, float val);
	//void paramSet(CUfunction fct, unsigned int offset, int val);
	//void paramSet(CUfunction fct, unsigned int offset, unsigned int  val);
	//void paramSet(CUfunction fct, unsigned int offset, void* val);

	// set default thread and grid sizes based on width/height pair
	void setGridBlock(int width, int height); // 16 x 8 and grid
	void setGridBlock(dim3& block, int width, int height);

	template <class T>
	void setGridBlock(ArrayCudaT<T>& A); // 16 x 8 and grid
};


// template functions
#include "cu.hxx"

#endif

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <typeinfo>

//#include <cuda.h>
//#include <cutil.h>

#include "array_cuda_t.h"
//#include "ArrayFloat.h"
//#include "ArrayT.h"
//#include "Array3D.h"
#include "cu.h"


//----------------------------------------------------------------------
CU::CU()
{
}
//----------------------------------------------------------------------
//CUresult
#if 0
CUmodule
CU::loadModule(const char* moduleName)
{
    CUresult status;
    char* module_path = cutFindFilePath(moduleName, 0);
    //char* module_path = cutFindFilePath("simpleTexture_kernel.cubin", 0);
    //module_path = cutFindFilePath("simpleTexture_kernel.cubin", executablePath);

    if (module_path == 0) {
        status = CUDA_ERROR_NOT_FOUND;
        goto Error;
    }

    status = CU_SAFE_CALL(cuModuleLoad(&cuModule, module_path));
    cutFree(module_path);

    if ( CUDA_SUCCESS != status ) {
        goto Error;
    }
    return cuModule;
    //return CUDA_SUCCESS;

Error:
    //cuCtxDetach( );
	printf("loadModule: error\n"); exit(0);
}
#endif
//----------------------------------------------------------------------
#if 0
CUfunction
CU::loadFunction(const char* functionName)
{
	CU_SAFE_CALL(cuModuleGetFunction( &cuFunction, cuModule, functionName));
    return cuFunction;
    //return CUDA_SUCCESS;

//Error:
	//CU_SAFE_CALL(cuCtxDetach());
	//printf("loadFunction: error\n"); exit(0);
}
#endif
//----------------------------------------------------------------------
#if 0
CUresult
CU::deleteDevice()
{
    cuCtxDetach(cuContext);
    //return status;
	return (CUresult) 0;
}
#endif
//----------------------------------------------------------------------
#if 0
CUresult
CU::createDevice()
{
    cuDevice = 0;

    CU_SAFE_CALL(cuInit(0));
    CU_SAFE_CALL(cuDeviceGet( &cuDevice, 0 ) );
    CU_SAFE_CALL( cuCtxCreate( &cuContext, 0, cuDevice ) );
	return (CUresult) 0;

//Error:
	//cuCtxDetach();
	printf("error in createDevice\n"); exit(0);
	//return status;
	return (CUresult) 0;
}
#endif
//----------------------------------------------------------------------
#if 0
CUentry::CUentry(const char* moduleName, const char* functionName)
{
	init(moduleName, functionName);
}
#endif
//----------------------------------------------------------------------
CUentry::CUentry()
{
	//init(moduleName, functionName);
	this->setBlock(16,8,1);
	this->setGrid(1,1,1);
	curOffset = 0;
}
//----------------------------------------------------------------------
//CUentry::CUentry(const char* moduleName, const char* functionName, int width, int height)
CUentry::CUentry(int width, int height)
{
	//init(moduleName, functionName);
	setGridBlock(width, height);
}
//----------------------------------------------------------------------
#if 0
void CUentry::init(const char* moduleName, const char* functionName)
{
	block = Vec3i(16, 8, 1); // default 
	loadModule(moduleName);
	loadFunction(functionName);
	curOffset = 0;
}
#endif
//----------------------------------------------------------------------
#if 0
CUmodule CUentry::getModule()
{
	return cuModule;
}
#endif
//----------------------------------------------------------------------
#if 0
CUfunction CUentry::getFunction()
{
	return cuFunction;
}
#endif
//----------------------------------------------------------------------
#if 0
void CUentry::loadModule(const char* moduleName)
{
    CUresult status;
    char* module_path = cutFindFilePath(moduleName, 0);
    //char* module_path = cutFindFilePath("simpleTexture_kernel.cubin", 0);
    //module_path = cutFindFilePath("simpleTexture_kernel.cubin", executablePath);

    if (module_path == 0) {
        status = CUDA_ERROR_NOT_FOUND;
		printf("error 1\n");
        goto Error;
    }

    CU_SAFE_CALL(cuModuleLoad(&cuModule, module_path));
    cutFree(module_path);

	return;
    //return CUDA_SUCCESS;

Error:
    //cuCtxDetach( );
	printf("CUentry: loadModule: error\n"); exit(0);
}
#endif
//----------------------------------------------------------------------
#if 0
void CUentry::loadFunction(const char* functionName)
{
	CU_SAFE_CALL(cuModuleGetFunction( &cuFunction, cuModule, functionName));
	return;
    //return CUDA_SUCCESS;

//Error:
	//cuCtxDetach();
	printf("loadFunction: error\n"); exit(0);
	//return status;
}
#endif
//----------------------------------------------------------------------
void CUentry::setBlock(int x, int y, int z)
{
	//cuFuncSetBlockShape(cuFunction, x,y,z);
	block = dim3(x,y,z);
}
//----------------------------------------------------------------------
void CUentry::setGrid(int x, int y, int z)
{
	grid = dim3(x,y,z);
}
//----------------------------------------------------------------------
//void CUentry::paramSet(CUfunction fct, unsigned int offset, float2 val) {
//	cuParamSetf(fct, offset, val);
//}
//----------------------------------------------------------------------
#if 0
void CUentry::paramSet(CUfunction fct, unsigned int offset, float val) {
	cuParamSetf(fct, offset, val);
}
//----------------------------------------------------------------------
void CUentry::paramSet(CUfunction fct, unsigned int offset, unsigned int val) {
	cuParamSeti(fct, offset, (unsigned int) val);
}
//----------------------------------------------------------------------
void CUentry::paramSet(CUfunction fct, unsigned int offset, int val)
{
	cuParamSeti(fct, offset, (unsigned int) val);
}
//----------------------------------------------------------------------
void CUentry::paramSet(CUfunction fct, unsigned int offset, void* val)
{
	cuParamSeti(fct, offset, (unsigned long) val); // for 64 bit machines
	//cuParamSeti(fct, offset, (unsigned long) val); // for 32 bit machines
}
#endif
//----------------------------------------------------------------------
void CUentry::setGridBlock(int width, int height) // 16 x 8 and grid
{
	int block_x = 16;
	int block_y = 8; // optimum size (16 in x is minimum, 8 in y allows multiple 
	setBlock(block_x, block_y, 1);

	int gx = (width-1) / block_x + 1; // assumes that wa is multiple of nbx
	int gy = (height-1) / block_y + 1;
	setGrid(gx, gy);
}
//----------------------------------------------------------------------
void CUentry::setGridBlock(dim3& block, int width, int height) 
{
	setBlock(block.x, block.y, block.z);

	int gx = (width-1) / block.x + 1; // assumes that wa is multiple of nbx
	int gy = (height-1) / block.y + 1;
	setGrid(gx, gy);
}
//----------------------------------------------------------------------
void CUentry::run()
{
	//CU_SAFE_CALL(cuLaunchGrid(cuFunction, grid.x(), grid.y()));
}
//----------------------------------------------------------------------

#ifndef __SimCudaHelper_h__
#define __SimCudaHelper_h__

#include "Config.h" 
//#define SPHSIMLIB_3D_SUPPORT

#include <cutil.h>
#include <cutil_inline.h>

#ifdef SPHSIMLIB_3D_SUPPORT
#include <GL/glew.h>
#include <GL/glu.h>

#include <cuda_gl_interop.h>
#include <cutil_gl_inline.h>
#endif

#ifdef SPHSIMLIB_3D_SUPPORT
#ifdef _WIN32
//#include <d3dx9.h>
#include <D3D9.h>
#include <cuda_d3d9_interop.h>
#include <cudad3d9.h>
#endif
#endif


namespace SimLib
{
	class SimCudaHelper
	{
	public:
		SimCudaHelper();
		~SimCudaHelper();

		void Initialize(int cudaDevice);
#ifdef SPHSIMLIB_3D_SUPPORT
		void InitializeGL(int cudaGLDevice);
		void InitializeD3D9(int cudaGLDevice, IDirect3DDevice9 *pDxDevice);

		// CUDA REGISTER
		static cudaError_t RegisterGLBuffer(GLuint vbo);
		static cudaError_t UnregisterGLBuffer(GLuint vbo);

#ifdef _WIN32
		static cudaError_t RegisterD3D9Buffer(IDirect3DResource9 * pResource);
		static cudaError_t UnregisterD3D9Buffer(IDirect3DResource9 * pResource);
#endif
		// CUDA MAPPING
		static cudaError_t MapBuffer(void **devPtr, IDirect3DResource9* pResource);
		static cudaError_t UnmapBuffer(void **devPtr, IDirect3DResource9* pResource);

		static cudaError_t MapBuffer(void **devPtr, GLuint bufObj);
		static cudaError_t UnmapBuffer(void **devPtr, GLuint bufObj);

#endif

		int PrintDevices(int deviceSelected);

	private:

		int Init(int cudaDevice);
		void CheckError(const char *msg);
		void CheckError(cudaError_t err, const char *msg);


	};
}


#endif

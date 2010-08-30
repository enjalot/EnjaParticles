#ifndef ISPH_CLDEVICE_H
#define ISPH_CLDEVICE_H

#include <string>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

namespace isph
{
	class CLSystem;
	class CLPlatform;

	/*!
	 *	\class	CLDevice
	 *	\brief	OpenCL enabled device.
	 */
	class CLDevice
	{
	public:

		/*!
		 *	\brief	Called only by CLSystem.
		 */
		CLDevice(CLPlatform* parentPlatform, cl_device_id device);

		~CLDevice();

		/*!
		 *	\brief	Get the device type (CPU, GPU, etc).
		 */
		inline cl_device_type Type() { return type; }

		/*!
		 *	\brief	Check is it CPU device.
		 */
		inline bool IsCPU() { return type == CL_DEVICE_TYPE_CPU; }

		/*!
		 *	\brief	Check is it GPU device.
		 */
		inline bool IsGPU() { return type == CL_DEVICE_TYPE_GPU; }

		/*!
		 *	\brief	Get the OpenCL ID of the device.
		 */
		inline const cl_device_id& ID() { return id; }

		/*!
		 *	\brief	Get the name of the device.
		 */
		inline const std::string& Name() { return name; }

		/*!
		 *	\brief	Get the vendor name of the device.
		 */
		inline const std::string& Vendor() { return vendor; }

		/*!
		 *	\brief	Get the number of compute units.
		 */
		inline unsigned int Cores() { return (unsigned int)maxComputeUnits; }

		/*!
		 *	\brief	Get the maximum clock frequency of device.
		 */
		inline unsigned int MaxFrequency() { return (unsigned int)maxClock; }

		/*!
		 *	\brief	Check if device supports double precision.
		 */
		inline bool DoublePrecision() { return fp64; }

		/*!
		 *	\brief	Check if device supports half precision.
		 */
		inline bool HalfPrecision() { return fp16; }

		/*!
		 *	\brief	Check if device supports atomic functions.
		 */
		inline bool Atomics() { return globalAtomics && localAtomics; }

		/*!
		 *	\brief	Check if device supports atomic functions in global memory.
		 */
		inline bool GlobalAtomics() { return globalAtomics; }

		/*!
		 *	\brief	Check if device supports atomic functions in local memory.
		 */
		inline bool LocalAtomics() { return localAtomics; }

		/*!
		 *	\brief	Get the size in bytes of global memory that device has.
		 */
		inline unsigned int GlobalMemorySize() { return (unsigned int)globalMemSize; }

		/*!
		 *	\brief	Get the size in bytes of local memory that device has.
		 */
		inline unsigned int LocalMemorySize() { return (unsigned int)localMemSize; }

		/*!
		 *	\brief	Get the performance index of the device.
		 */
		inline unsigned int PerformanceIndex() { return (unsigned int)performanceIndex; }

		/*!
		 *	\brief	Get device maximum work-group size.
		 */
		inline unsigned int MaxWorkGroupSize() { return (unsigned int)maxWorkGroupSize; }

		/*!
		 *	\brief	Get device maximum work-item size for a dimension.
		 */
		inline unsigned int MaxWorkItemSize(unsigned int dimension) { return (unsigned int)maxWorkItemSize[dimension]; }

		/*!
		 *	\brief	Get maximum size that can be allocated on the device.
		 */
		inline unsigned int MaxAllocSize() { return (unsigned int)maxAllocSize; }

	private:
		friend class CLSystem;

		CLPlatform* platform;
		cl_device_id id;
		cl_device_type type;
		std::string name;
		std::string vendor;
		cl_uint maxComputeUnits;
		cl_uint maxClock;
		cl_uint performanceIndex;
		size_t maxWorkGroupSize;
		size_t maxWorkItemSize[3];
		cl_ulong globalMemSize;
		cl_ulong localMemSize;
		cl_ulong maxAllocSize;
		bool fp16;
		bool fp64;
		bool globalAtomics;
		bool localAtomics;
	};

}

#endif

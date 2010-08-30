#ifndef ISPH_CLPLATFORM_H
#define ISPH_CLPLATFORM_H

#include <string>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

namespace isph
{
	class CLSystem;
	class CLDevice;
	class CLProgram;

	/*!
	 *	\class	CLPlatform
	 *	\brief	Platform that has OpenCL enabled devices.
	 */
	class CLPlatform
	{
	public:

		/*!
		 *	\brief	Called only by CLSystem.
		 */
		CLPlatform(cl_platform_id platform);

		~CLPlatform();

		/*!
		 *	\brief	Get the OpenCL ID of the platform.
		 */
		inline cl_platform_id ID() { return id; }

		/*!
		 *	\brief	Get the name of the platform.
		 */
		inline const std::string& Name() { return name; }

		/*!
		 *	\brief	Get the vendor of the platform.
		 */
		inline const std::string& Vendor() { return vendor; }

		/*!
		 *	\brief	Get the OpenCL version.
		 */
		inline const std::string& CLVersion() { return clVersion; }

		/*!
		 *	\brief	Get the number of OpenCL enabled devices in platform.
		 */
		inline unsigned int DeviceCount() { return deviceCount; }

		/*!
		 *	\brief	Get one of the devices in the platform.
		 */
		inline CLDevice*& Device(unsigned int i) { return devices[i]; }

	private:
		friend class CLSystem;

		cl_platform_id id;
		std::string name;
		std::string vendor;
		std::string clVersion;

		cl_uint deviceCount;
		CLDevice** devices;
	};
}

#endif

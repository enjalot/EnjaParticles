#ifndef ISPH_CLLINK_H
#define ISPH_CLLINK_H

#include <string>
#include <vector>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

namespace isph
{
	class CLProgram;
	class CLPlatform;
	class CLDevice;
	class CLVariable;

	/*!
	 *	\brief	Connection to OpenCL devices and their command queues.
	 */
	class CLLink
	{
	public:

		/*!
		 *	\brief	Create new context with all devices in specified platform.
		 */
		CLLink(CLPlatform* platform);

		/*!
		 *	\brief	Create new context with one device.
		 */
		CLLink(CLDevice* device);

		/*!
		 *	\brief	Create new context with devices.
		 */
		CLLink(std::vector<CLDevice*>& devices);

		/*!
		 *	\brief	Class destructor.
		 */
		~CLLink();

		/*!
		 *	\brief	Get the number of devices in context.
		 */
		inline unsigned int DeviceCount() { return deviceCount; }
		
		/*!
		 *	\brief	Get one of the devices in context.
		 */
		inline CLDevice* Device(unsigned int i) { return devices[i]; }

		/*!
		 *	\brief	Get the OpenCL command queue of one device in context.
		 */
		inline const cl_command_queue& Queue(unsigned int i) { return queues[i]; }

		/*!
		 *	\brief	Get the performace factor of the device for the context.
		 */
		inline double PerformanceFactor(unsigned int i) { return deviceFactors[i]; }

		/*!
		 *	\brief	Get the OpenCL context object.
		 */
		inline const cl_context& Context() { return context; }

	private:

		friend class CLProgram;

		bool Initialize(CLDevice** devices, unsigned int size);
		bool BuildProgram(cl_program& program, const std::string& buildOptions);
		bool Finish();

		cl_context context;

		unsigned int deviceCount;
		CLDevice** devices;
		cl_command_queue *queues;
		double* deviceFactors;

		std::vector<CLVariable*> variables;
	};
}

#endif

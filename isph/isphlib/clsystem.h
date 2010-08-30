#ifndef ISPH_CLSYSTEM_H
#define ISPH_CLSYSTEM_H

#include <string>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

namespace isph
{
	class CLPlatform;
	class CLDevice;
	class CLProgram;

	/*!
	 *	\class	CLSystem
	 *	\brief	System of OpenCL enabled platforms (singleton class).
	 */
	class CLSystem
	{
	public:
		CLSystem();
		~CLSystem();

		/*!
		 *	\brief	Get the one and only instance of this class.
		 */
		static CLSystem* Instance();

		/*!
		 *	\brief	Get the number of OpenCL enabled platforms.
		 */
		inline unsigned int PlatformCount() { return platformCount; }

		/*!
		 *	\brief	Get one of the platforms in the system.
		 */
		inline CLPlatform* Platform(unsigned int i) { return platforms[i]; }

		/*!
		 *	\brief	Get the first availible OpenCL supported platform.
		 */
		inline CLPlatform* FirstPlatform() { return platformCount ? platforms[0] : NULL; }

		/*!
		 *	\brief	Get description of OpenCL runtime error.
		 */
		std::string ErrorDesc(cl_int status);

	private:

		friend class CLProgram;

		/*!
		 *	\brief	Initialize OpenCL system.
		 */
		void Initialize();

		static CLSystem* instance;

		cl_uint platformCount;
		CLPlatform** platforms;

	};
}

#endif

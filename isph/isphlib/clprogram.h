#ifndef ISPH_CLPROGRAM_H
#define ISPH_CLPROGRAM_H

#include <string>
#include <vector>
#include <map>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

namespace isph {

	class CLSubProgram;
	class CLPlatform;
	class CLDevice;
	class CLVariable;
	class CLLink;

	/*!
	 *	\class	CLProgram
	 *	\brief	Compound of CLSubProgram and CLVariable objects ready to build and run on devices.
	 */
	class CLProgram
	{
	public:

		CLProgram();
		~CLProgram();

		/*!
		 *	\brief	Set link to devices that will run the program.
		 *	\param	linkToDevices	Link to OpenCL enabled devices that will run the program.
		 */
		bool SetLink(CLLink* linkToDevices);

		/*!
		 *	\brief	Get link to devices that will run the program.
		 */
		inline CLLink* Link() { return link; }

		/*!
		 *	\brief	Add a part of program.
		 *	\param	subprogram	Pointer to the part of the source code.
		 */
		bool AddSubprogram(CLSubProgram* subprogram);

		/*!
		 *	\brief	Remove all parts of program.
		 */
		void ClearSubprograms();

		/*!
		 *	\brief	Set wether to use unsafe math optimizations.
		 */
		void SetUnsafeOptimizations(bool enabled);

		/*!
		 *	\brief	Get wether are unsafe math optimizations enabled.
		 */
		inline bool UnsafeOptimizations() { return unsafeMath; }

		/*!
		 *	\brief	Add custom build options (e.g. custom preprocessor define).
		 */
		void AddBuildOption(const std::string& option);

		/*!
		 *	\brief	Remove added custom build options.
		 */
		void ClearBuildOptions();

		/*!
		 *	\brief	Compile and link the program on devices.
		 *	\remarks Set devices (SetDevice/SetDevices) before calling this function.
		 */
		bool Build();

		/*!
		 *	\brief	Check if program is successfuly compiled and linked.
		 */
		inline bool IsBuilt() { return isBuilt; }

		/*!
		 *	\brief	Wait until program has finished.
		 */
		bool Finish();

		/*!
		 *	\brief	Get the map of program's variables (with its semantics as keys).
		 */
		inline const std::map<std::string,CLVariable*>& Variables() { return variables; }

		/*!
		 *	\brief	Get the program variable from its semantic.
		 */
		CLVariable* Variable(const std::string& semantic);

		/*!
		 *	\brief	Set (yet another) semantic that will represent specific variable.
		 */
		bool ConnectSemantic(const std::string& semantic, CLVariable* var);

		/*!
		 *	\brief	Get the amount of used memory in bytes program has allocated on devices.
		 */
		size_t UsedMemorySize();

	private:

		friend class CLVariable;
		friend class CLSubProgram;

		CLLink *link;
		bool unsafeMath;
		std::vector<std::string> buildOptions;
		std::string source;
		std::map<std::string,CLVariable*> variables;
		std::vector<CLSubProgram*> subprograms;
		bool isBuilt;
		cl_program program;
		
	};

}

#endif

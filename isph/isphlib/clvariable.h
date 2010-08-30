#ifndef ISPH_CLVARIABLE_H
#define ISPH_CLVARIABLE_H

#include <string>
#include <list>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

namespace isph
{
	class CLProgram;
	class CLSubProgram;

	/*!
	 *	\enum	VariableDataType
	 *	\brief	Different data types.
	 */
	enum VariableDataType
	{
		FloatType,
		Float2Type,
		Float4Type,
		DoubleType,
		Double2Type,
		Double4Type,
		UintType,
		Uint2Type,
		Uint4Type,
		IntType,
		Int2Type,
		Int4Type
	};

	/*!
	 *	\enum	VariableType
	 *	\brief	Different types of OpenCL variables.
	 */
	enum VariableType
	{
		ProgramConstant,
		KernelArgument,
		GlobalBuffer,
		LocalBuffer
	};

	/*!
	 *	\class	CLVariable
	 *	\brief	OpenCL variable (buffer, scalar, etc); used across CLProgram, manipulated in CLSubProgram.
	 */
	class CLVariable
	{
	public:

		/*!
		 *	\brief	Create variable to be used in CLProgram.
		 */
		CLVariable(CLProgram* program, const std::string& semantic, VariableType type);

		/*!
		 *	\brief	Class destructor.
		 */
		~CLVariable();

		/*!
		 *	\brief	Get the type of the variable.
		 */
		inline VariableType Type() { return varType; }

		/*!
		 *	\brief	Get the semantic associated to the variable.
		 */
		inline const std::string& Semantic() { return semantics.front(); }

		/*!
		 *	\brief	Read the data from devices to the host.
		 *	\param	memoryPos	Position in memory with size of MemorySize(); space where data from devices will be copied.
		 */
		bool ReadTo(void* memoryPos);

		/*!
		 *	\brief	Write the data from host to devices.
		 *	\param	memoryPos	Position in memory with size of MemorySize(); data that will be copied to devices.
		 */
		bool WriteFrom(const void* memoryPos);

		/*!
		 *	\brief	Copy data from another buffer.
		 *	\param	var	Variable data that will be copied to this variable.
		 */
		bool WriteFrom(CLVariable* var);

		/*!
		 *	\brief	Set variable memory space in bytes.
		 *	\param	elements	Number of elements to allocate. 
		 *	\param	dataType	Type of data.
		 */
		void SetSpace(VariableDataType dataType, unsigned int elements);

		/*!
		 *	\brief	Get the data type of the variable.
		 */
		inline VariableDataType DataType() { return varDataType; }

		/*!
		 *	\brief	Get the size of variable data type in bytes.
		 */
		size_t DataTypeSize();

		/*!
		 *	\brief	Get the number of elements.
		 */
		inline size_t ElementCount() { return elementCount; }

		/*!
		 *	\brief	Get the allocated memory: ElementCount()*TypeSize().
		 */
		inline size_t MemorySize() { return memorySize; }

		/*!
		 *	\brief	Return OpenCL memory object of the variable.
		 */
		inline cl_mem Buffer(unsigned int device) { return clBuffers[device]; }

	private:

		inline bool IsSplit() { return bufferCount > 1; }

		bool Allocate();
		void Release();

		bool SetAsArgument(CLSubProgram *kernel, unsigned int argID, unsigned int deviceID = 0, size_t kernelLocalSize = 0);

		inline size_t Offset(unsigned int deviceID) { return IsSplit() ? offsets[deviceID] : 0; }
		inline size_t ElementCount(unsigned int deviceID) { return IsSplit() ? partElementCount[deviceID] : varType == KernelArgument ? (size_t)*simpleData : elementCount; }

		friend class CLProgram;
		friend class CLSubProgram;

		CLProgram *parentProgram;

		// OpenCL buffer data
		cl_mem* clBuffers;
		unsigned int bufferCount;
		size_t* partElementCount;
		size_t* offsets;

		// simple constant data
		char* simpleData;
		
		// general data
		VariableType varType;
		VariableDataType varDataType;
		size_t elementCount;
		size_t memorySize;
		bool needsUpdate;
		std::list<std::string> semantics;
	};
}

#endif

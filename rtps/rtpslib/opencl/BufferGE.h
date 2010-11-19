#ifndef RTPS_BUFFERGE_H_INCLUDED
#define RTPS_BUFFERGE_H_INCLUDED
/*
 * The BufferGE class abstracts the OpenCL BufferGE and BufferGEGL classes
 * by providing some convenience methods
 *
 * we pass in an OpenCL instance  to the constructor 
 * which manages the underlying context and queues
 */

#include <string>

#include "CLL.h"

namespace rtps {


template <class T>
class BufferGE
{
private:
	T* data;
	//GLuint vbo_id;
	bool externalPtr;
	int nb_el;
	/// Total storage allocated (GPU + CPU)
	static int total_bytes_GPU;
	static int total_bytes_CPU;

//public:
//enum IOType {READ_WRITE=0, READ, WRITE};

public:
    BufferGE(){ cli=NULL; data = 0;}

    //allocate memory
	//do not transfer data to GPU by default
	BufferGE(CL *cli, int sz, int io_type=CL_MEM_READ_WRITE);

    //create an OpenCL buffer from existing external data
	//do not transfer data to GPU by default
	BufferGE(CL *cli, T* data, int sz, int io_type=CL_MEM_READ_WRITE);
	//BufferGE(CL *cli, std::vector<T>& data);
		

    //BufferGE(CL *cli, const std::vector<T> &data);
    //create a OpenCL BufferGL from a vbo_id
    //if managed is true then the destructor will delete the VBO
    //BufferGE(CL *cli, GLuint vbo_id=-1);

    ~BufferGE();

    //we will want to access buffers by name when going across systems
    //std::string name;
    //the actual buffer handled by the Khronos OpenCL c++ header
    //cl::Memory cl_buffer;
    std::vector<cl::Memory> cl_buffer;

	// I would make this static, unless there is a reason to have more than one, for 
	// example when working on multiple GPUs. 
    CL *cli;


    //need to acquire and release arrays from OpenGL context if we have a VBO
	// put in BufferVBO class
    //void acquire();
    //void release();

    void copyToDevice();
    void copyToHost();

	T& operator[](int i) { return data[i]; }
	const T& operator[](const int i) const { return data[i]; }

	T& operator()(int i) { return data[i]; }
	const T& operator()(const int i) const { return data[i]; }

    //void set(T val);
    //void set(const std::vector<T> &data);

	T* getHostPtr() { return data; }
	// return a cl_mem object
	//cl::Memory getDevicePtr() { return cl_buffer[0](); }
	cl_mem getDevicePtr() { return cl_buffer[0](); }

	static int getNbBytesGPU() { return total_bytes_GPU; } 
	static int getNbBytesCPU() { return total_bytes_CPU; } 

	int getSize() { return nb_el; }

private:
	BufferGE(const BufferGE&); // copy constructor
	const BufferGE& operator=(const BufferGE&) const;
	BufferGE& operator=(BufferGE&);

};

#include "BufferGE.cpp"

}
#endif



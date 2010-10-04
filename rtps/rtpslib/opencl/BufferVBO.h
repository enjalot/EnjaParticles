#ifndef RTPS_BUFFERVBO_H_INCLUDED
#define RTPS_BUFFERVBO_H_INCLUDED
/*
 * The BufferVBO class abstracts the OpenCL BufferVBO and BufferVBOGL classes
 * by providing some convenience methods
 *
 * we pass in an OpenCL instance  to the constructor 
 * which manages the underlying context and queues
 */

#include <string>

//for glGenBuffers
//#include <GL/glew.h>

#include "CLL.h"


namespace rtps {

template <class T>
class BufferVBO
{
private:
	//T* data;
	GLuint vbo_id;
	//bool externalPtr;
	//int nb_el;

public:
    BufferVBO(){ cli=NULL; vbo_id=0; }

    //create a OpenCL BufferGL from a vbo_id
    //if managed is true then the destructor will delete the VBO
    BufferVBO(CL *cli, GLuint vbo_id=-1);

    ~BufferVBO();

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
    void acquire();
    void release();

	// I should be able to copy contents of vbo to the host into an array, in this case external
    //void copyToDevice();
    //void copyToHost();

	GLuint getVboId() { return vbo_id; }

	cl_mem getDevicePtr() { return cl_buffer[0](); }
	//cl_mem getHostPtr() { return cl_buffer[0](); }
};

//#include "BufferVBO.cpp"
template <class T>
BufferVBO<T>::BufferVBO(CL *cli, GLuint vbo_id)
{
	//data = 0;
	//externalPtr = false;
	//this->nb_el = 0;

    this->cli = cli;
	if (vbo_id == -1) {
		//glGenBuffers(1, &vbo_id);
        vbo_id = registerVBO();
	}

	this->vbo_id = vbo_id;
    cl_buffer.push_back(cl::BufferGL(cli->context, CL_MEM_READ_WRITE, vbo_id, &cli->err));
}

//----------------------------------------------------------------------
template <class T>
BufferVBO<T>::~BufferVBO()
{
}

//----------------------------------------------------------------------
template <class T>
void BufferVBO<T>::acquire()
{
    cli->err = cli->queue.enqueueAcquireGLObjects(&cl_buffer, NULL, &cli->event);
    cli->queue.finish();
}

//----------------------------------------------------------------------
template <class T>
void BufferVBO<T>::release()
{
    cli->err = cli->queue.enqueueReleaseGLObjects(&cl_buffer, NULL, &cli->event);
    cli->queue.finish();
}

}
#endif


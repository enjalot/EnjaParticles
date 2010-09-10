#ifndef RTPS_BUFFER_H_INCLUDED
#define RTPS_BUFFER_H_INCLUDED
/*
 * The Buffer class abstracts the OpenCL Buffer and BufferGL classes
 * by providing some convenience methods
 *
 * we pass in an OpenCL instance  to the constructor 
 * which manages the underlying context and queues
 */

#include <string>

#include "CLL.h"

namespace rtps {

template <class T>
class Buffer
{
public:
    Buffer(){ cli=NULL; vbo_id=0; };
    //create an OpenCL buffer from existing data
    Buffer(CL *cli, const std::vector<T> &data);
    //create a OpenCL BufferGL from a vbo_id
    //if managed is true then the destructor will delete the VBO
    Buffer(CL *cli, GLuint vbo_id);
    ~Buffer();

    //we will want to access buffers by name when going across systems
    //std::string name;
    //the actual buffer handled by the Khronos OpenCL c++ header
    //cl::Memory cl_buffer;
    std::vector<cl::Memory> cl_buffer;

    CL *cli;

    //if this is a VBO we store its id
    GLuint vbo_id;

    //need to acquire and release arrays from OpenGL context if we have a VBO
    void acquire();
    void release();

    void copyToDevice(const std::vector<T> &data);
    std::vector<T>* copyToHost(int num);

    void set(T val);
    void set(const std::vector<T> &data);

};

#include "Buffer.cpp"

}
#endif


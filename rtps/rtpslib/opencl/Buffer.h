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
    //create an OpenCL buffer from existing data
    Buffer(CL *cli, std::string name, std::vector<T> data, int num_elements, bool vbo);
    //create a OpenCL BufferGL from a vbo_id
    Buffer(CL *cli, std::string name, GLuint vbo_id, int num_elements);

    //we will want to access buffers by name when going accross systems
    std::string name;
    //the actual buffer handled by the Khronos OpenCL c++ header
    cl::Memory cl_buffer;

    std::vector<T> data;
    int num_elements;

    CL *cli;

    //if this is a VBO we store its id
    GLuint vbo_id;
    //need to acquire and release arrays from OpenGL context if we have a VBO
    void acquire();
    void release();

    void copyToDevice();
    void copyToHost();

    //set all the data to a single value
    void set(T t);
    //copy a vector to data
    void set(std::vector<T> t);


};

}

#endif


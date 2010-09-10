#include "Buffer.h"

namespace rtps {

template <class T>
Buffer<T>::Buffer(CL *_cli, std::string name, std::vector<T> data, int num_elements, bool vbo)
{
    cli = _cli;
}


template <class T>
Buffer<T>::Buffer(CL *_cli, std::string name, GLuint vbo_id, int num_elements)
{
    cli = _cli;
}


template <class T>
void Buffer<T>::acquire()
{

}


template <class T>
void Buffer<T>::release()
{
}


template <class T>
void Buffer<T>::copyToDevice()
{

}


template <class T>
void Buffer<T>::copyToHost()
{

}


template <class T>
void Buffer<T>::set(T t)
{

}


template <class T>
void Buffer<T>::set(std::vector<T> t)
{

}



}

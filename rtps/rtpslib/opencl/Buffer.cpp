//#include "Buffer.h"

//namespace rtps {

template <class T>
Buffer<T>::Buffer(CL *cli, const std::vector<T> &data)
{
    this->cli = cli;
    //this->data = data;

    cl_buffer.push_back(cl::Buffer(cli->context, CL_MEM_READ_WRITE, data.size()*sizeof(T), NULL, &cli->err));
    copyToDevice(data);


}

template <class T>
Buffer<T>::Buffer(CL *cli, const std::vector<T> &data, unsigned int memtype)
{
    this->cli = cli;
    //this->data = data;

    cl_buffer.push_back(cl::Buffer(cli->context, memtype, data.size()*sizeof(T), NULL, &cli->err));
    copyToDevice(data);


}


template <class T>
Buffer<T>::Buffer(CL *cli, GLuint vbo_id)
{
    this->cli = cli;
    cl_buffer.push_back(cl::BufferGL(cli->context, CL_MEM_READ_WRITE, vbo_id, &cli->err));
}

template <class T>
Buffer<T>::~Buffer()
{
}

template <class T>
void Buffer<T>::acquire()
{
    cli->err = cli->queue.enqueueAcquireGLObjects(&cl_buffer, NULL, &cli->event);
    cli->queue.finish();
}


template <class T>
void Buffer<T>::release()
{
    cli->err = cli->queue.enqueueReleaseGLObjects(&cl_buffer, NULL, &cli->event);
    cli->queue.finish();
}


template <class T>
void Buffer<T>::copyToDevice(const std::vector<T> &data)
{
    //TODO clean up this memory/buffer issue (nasty pointer casting)
    cli->err = cli->queue.enqueueWriteBuffer(*((cl::Buffer*)&cl_buffer[0]), CL_TRUE, 0, data.size()*sizeof(T), &data[0], NULL, &cli->event);
    cli->queue.finish();

}

template <class T>
void Buffer<T>::copyToDevice(const std::vector<T> &data, int start)
{
    //TODO clean up this memory/buffer issue (nasty pointer casting)
    cli->err = cli->queue.enqueueWriteBuffer(*((cl::Buffer*)&cl_buffer[0]), CL_TRUE, start*sizeof(T), data.size()*sizeof(T), &data[0], NULL, &cli->event);
    cli->queue.finish();

}

template <class T>
std::vector<T> Buffer<T>::copyToHost(int num)
{
    //TODO clean up this memory/buffer issue
    std::vector<T> data(num);
    //TODO pass back a pointer instead of a copy
    //std::vector<T> data = new std::vector<T>(num);
    cli->err = cli->queue.enqueueReadBuffer(*((cl::Buffer*)&cl_buffer[0]), CL_TRUE, 0, data.size()*sizeof(T), &data[0], NULL, &cli->event);
    cli->queue.finish();
    return data;

}

template <class T>
std::vector<T> Buffer<T>::copyToHost(int start, int num)
{
    //TODO clean up this memory/buffer issue
    std::vector<T> data(num);
    //TODO pass back a pointer instead of a copy
    //std::vector<T> data = new std::vector<T>(num);
    cli->err = cli->queue.enqueueReadBuffer(*((cl::Buffer*)&cl_buffer[0]), CL_TRUE, start*data.size()*sizeof(T), data.size()*sizeof(T), &data[0], NULL, &cli->event);
    cli->queue.finish();
    return data;

}

//}

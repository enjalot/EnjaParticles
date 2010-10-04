//#include "BufferVBO.h"

//namespace rtps {

//----------------------------------------------------------------------
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

//----------------------------------------------------------------------
//----------------------------------------------------------------------


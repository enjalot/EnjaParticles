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
#include <vector>

#include "CLL.h"
#ifdef WIN32
    //#if defined(rtps_EXPORTS)
	//This needs to be handled better. For some reason the above ifdef works
    // in all the other include files except this one.

        #define RTPS_EXPORT __declspec(dllexport)

    //#else
    //    #define RTPS_EXPORT __declspec(dllimport)
	//#endif 

#else
    #define RTPS_EXPORT
#endif

namespace rtps
{
    
    template <class T>
    class RTPS_EXPORT Buffer
    {
    public:
        Buffer(){ cli=NULL; vbo_id=0; };
        //create an OpenCL buffer from existing data
        Buffer(CL *cli, const std::vector<T> &data);
        Buffer(CL *cli, const std::vector<T> &data, unsigned int memtype);
        //create a OpenCL BufferGL from a vbo_id
        Buffer(CL *cli, GLuint vbo_id);
        Buffer(CL *cli, GLuint vbo_id, int type);
        ~Buffer();

        cl_mem getDevicePtr() { return cl_buffer[0](); }
        cl::Memory& getBuffer(int index) {return cl_buffer[index];};
       
        //need to acquire and release arrays from OpenGL context if we have a VBO
        void acquire();
        void release();

        void copyToDevice(const std::vector<T> &data);
        //pastes the data over the current array starting at [start]
        void copyToDevice(const std::vector<T> &data, int start);

        //really these should take in a presized vector<T> to be filled
        //these should be factored out
        std::vector<T> copyToHost(int num);
        std::vector<T> copyToHost(int num, int start);
        //correct way (matches copyToDevice
        void copyToHost(std::vector<T> &data);
        void copyToHost(std::vector<T> &data, int start);


        void copyFromBuffer(Buffer<T> dst, size_t start_src, size_t start_dst, size_t size);
        

        //these don't appear to be implemented. need to revisit
        void set(T val);
        void set(const std::vector<T> &data);
		void getSize() { return nb_el; }
		void getNbBytes() { return nb_bytes; }

    private:
         //we will want to access buffers by name when going across systems
        //std::string name;
        //the actual buffer handled by the Khronos OpenCL c++ header
        //cl::Memory cl_buffer;
        std::vector<cl::Memory> cl_buffer;
		int nb_el; // measured in units of <T>
		int nb_bytes; 

        CL *cli;

        //if this is a VBO we store its id
        GLuint vbo_id;


    };

    #include "Buffer.cpp"

}
#endif


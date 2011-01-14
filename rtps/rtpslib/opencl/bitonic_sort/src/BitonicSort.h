#include "../opencl/Kernel.h"
#include "../opencl/Buffer.h"
#include "../opencl/Scopy.h


namespace rtps
{

class Bitonic
{
public:
    Bitonic(){ cli=NULL; };
    //create an OpenCL buffer from existing data
    Bitonic(    CL *cli,
                Buffer<T> dstkey, Buffer<T> dstval, 
                Buffer<T> srckey, Buffer<T> srcval, 
                int batch,          
                int arrayLength,
                int dir);

private:
    //template <class T> 
    template <class T> void Sort();

}

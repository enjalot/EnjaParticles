/**
 * C++ port of NVIDIA's Bitonic Sort implementation
 */


template <class T>
Bitonic<T>::Bitonic(std::string source_dir, CL *cli )
{
    this->cli = cli;
    /*
    this->cl_dstkey = dstkey;
    this->cl_dstval = dstval;
    this->cl_srckey = srckey;
    this->cl_srcval = srcval;
    */
    loadKernels(source_dir);
}

template <class T>
void Bitonic<T>::loadKernels(std::string source_dir)
{
    source_dir += "/bitonic.cl";

    std::string options = "-D LOCAL_SIZE_LIMIT=512";
    cl::Program prog = cli->loadProgram(source_dir, options);
    k_bitonicSortLocal = Kernel(cli, prog, "bitonicSortLocal");
    //k_bitonicSortLocal = Kernel(cli, path, "bitonicSortLocal");
    k_bitonicSortLocal1 = Kernel(cli, prog, "bitonicSortLocal1");
    //k_bitonicSortLocal1 = Kernel(cli, path, "bitonicSortLocal1");
    k_bitonicMergeLocal = Kernel(cli, prog, "bitonicMergeLocal");
    //k_bitonicMergeLocal = Kernel(cli, path, "bitonicMergeLocal");
    k_bitonicMergeGlobal = Kernel(cli, prog, "bitonicMergeGlobal");
    //k_bitonicMergeGlobal = Kernel(cli, path, "bitonicMergeGlobal");

    //TODO: implement this check with the C++ API    
    //if( (szBitonicSortLocal < (LOCAL_SIZE_LIMIT / 2)) || (szBitonicSortLocal1 < (LOCAL_SIZE_LIMIT / 2)) || (szBitonicMergeLocal < (LOCAL_SIZE_LIMIT / 2)) ){
            //shrLog("\nERROR !!! Minimum work-group size %u required by this application is not supported on this device.\n\n", LOCAL_SIZE_LIMIT / 2);

    /*
    printf("bitonic dev pointers: %d\n", cl_dstkey->getDevicePtr());
    printf("bitonic dev pointers: %d\n", cl_dstval->getDevicePtr());
    printf("bitonic dev pointers: %d\n", cl_srckey->getDevicePtr());
    printf("bitonic dev pointers: %d\n", cl_srcval->getDevicePtr());
    */


}

static cl_uint factorRadix2(cl_uint& log2L, cl_uint L){
    if(!L){
        log2L = 0;
        return 0;
    }else{
        for(log2L = 0; (L & 1) == 0; L >>= 1, log2L++);
        return L;
    }
}



template <class T>
int Bitonic<T>::Sort(int batch, int arrayLength, int dir,
                    Buffer<T> *cl_dstkey, Buffer<T> *cl_dstval, 
                    Buffer<T> *cl_srckey, Buffer<T> *cl_srcval)
{

    if(arrayLength < 2)
        return 0;


    int arg = 0;
    k_bitonicSortLocal.setArg(arg++, cl_dstkey->getDevicePtr());
    k_bitonicSortLocal.setArg(arg++, cl_dstval->getDevicePtr());
    k_bitonicSortLocal.setArg(arg++, cl_srckey->getDevicePtr());
    k_bitonicSortLocal.setArg(arg++, cl_srcval->getDevicePtr());

    arg = 0;
    k_bitonicSortLocal1.setArg(arg++, cl_dstkey->getDevicePtr());
    k_bitonicSortLocal1.setArg(arg++, cl_dstval->getDevicePtr());
    k_bitonicSortLocal1.setArg(arg++, cl_srckey->getDevicePtr());
    k_bitonicSortLocal1.setArg(arg++, cl_srcval->getDevicePtr());

    arg = 0;
    k_bitonicMergeGlobal.setArg(arg++, cl_dstkey->getDevicePtr());
    k_bitonicMergeGlobal.setArg(arg++, cl_dstval->getDevicePtr());
    k_bitonicMergeGlobal.setArg(arg++, cl_dstkey->getDevicePtr());
    k_bitonicMergeGlobal.setArg(arg++, cl_dstval->getDevicePtr());

    arg = 0;
    k_bitonicMergeLocal.setArg(arg++, cl_dstkey->getDevicePtr());
    k_bitonicMergeLocal.setArg(arg++, cl_dstval->getDevicePtr());
    k_bitonicMergeLocal.setArg(arg++, cl_dstkey->getDevicePtr());
    k_bitonicMergeLocal.setArg(arg++, cl_dstval->getDevicePtr());






    //Only power-of-two array lengths are supported so far
    cl_uint log2L;
    cl_uint factorizationRemainder = factorRadix2(log2L, arrayLength);
    //printf("bitonic factorization remainder: %d\n", factorizationRemainder);
    
    dir = (dir != 0);
    //printf("dir: %d\n", dir);

    int localWorkSize;
    int globalWorkSize;

    if(arrayLength <= LOCAL_SIZE_LIMIT)
    {
         //Launch bitonicSortLocal
        k_bitonicSortLocal.setArg(4, arrayLength);
        k_bitonicSortLocal.setArg(5, dir); 

        localWorkSize  = LOCAL_SIZE_LIMIT / 2;
        globalWorkSize = batch * arrayLength / 2;
        k_bitonicSortLocal.execute(globalWorkSize, localWorkSize);
  
    }
    else
    {
        //Launch bitonicSortLocal1
        
        localWorkSize  = LOCAL_SIZE_LIMIT / 2;
        globalWorkSize = batch * arrayLength / 2;
        k_bitonicSortLocal1.execute(globalWorkSize, localWorkSize);

        for(uint size = 2 * LOCAL_SIZE_LIMIT; size <= arrayLength; size <<= 1)
        {
            for(unsigned stride = size / 2; stride > 0; stride >>= 1)
            {
                if(stride >= LOCAL_SIZE_LIMIT)
                {
                    //Launch bitonicMergeGlobal
                    k_bitonicMergeGlobal.setArg(4, arrayLength);
                    k_bitonicMergeGlobal.setArg(5, size);
                    k_bitonicMergeGlobal.setArg(6, stride);
                    k_bitonicMergeGlobal.setArg(7, dir); 

                    globalWorkSize = batch * arrayLength / 2;
                    k_bitonicMergeGlobal.execute(globalWorkSize);
                }
                else
                {
                    //Launch bitonicMergeLocal
                    
                    
                    k_bitonicMergeLocal.setArg(4, arrayLength);
                    k_bitonicMergeLocal.setArg(5, stride);
                    k_bitonicMergeLocal.setArg(6, size);
                    k_bitonicMergeLocal.setArg(7, dir); 

                    localWorkSize  = LOCAL_SIZE_LIMIT / 2;
                    globalWorkSize = batch * arrayLength / 2;
                    
                    k_bitonicMergeLocal.execute(globalWorkSize, localWorkSize);
                    break;
                }
                //printf("globalWorkSize: %d\n", globalWorkSize);
            }
        }




        
    }

    return localWorkSize;

    


    //scopy(num, cl_sort_output_hashes.getDevicePtr(), 
	//             cl_sort_hashes.getDevicePtr());
	//scopy(num, cl_sort_output_indices.getDevicePtr(), 
	//             cl_sort_indices.getDevicePtr());
    


}

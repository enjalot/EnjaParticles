/**
 * C++ port of NVIDIA's Radix Sort implementation with Key-Value instead of Keys Only
 */


template <class T>
Radix<T>::Radix(std::string source_dir, CL *cli, int max_elements, int cta_size )
{
    this->cli = cli;

    WARP_SIZE = 128;//32
    SCAN_WG_SIZE = 128;//256
    MIN_LARGE_ARRAY_SIZE = 4 * SCAN_WG_SIZE;
    bit_step = 4;
    //maybe cta_size should be passed to the call instead of the constructor
    this->cta_size = cta_size;
    uintsz = sizeof(T);

    loadKernels(source_dir);

    int num_blocks;
    if ((max_elements % (cta_size * 4)) == 0)
    {
        num_blocks = max_elements / (cta_size * 4);
    }
    else
    {
        num_blocks = max_elements / (cta_size * 4) + 1;
    }

    vector<unsigned int> tmp(max_elements);
    d_tempKeys = Buffer<unsigned int>(cli, tmp);
    d_tempValues = Buffer<unsigned int>(cli, tmp);

    tmp.resize(WARP_SIZE * num_blocks);
    mCounters = Buffer<unsigned int>(cli, tmp);
    mCountersSum = Buffer<unsigned int>(cli, tmp);
    mBlockOffsets = Buffer<unsigned int>(cli, tmp);

    int numscan = max_elements/2/cta_size*16;
    if (numscan >= MIN_LARGE_ARRAY_SIZE)
    {
    //#MAX_WORKGROUP_INCLUSIVE_SCAN_SIZE 1024
        tmp.resize(numscan / 1024);
        scan_buffer = Buffer<unsigned int>(cli, tmp);
    }



}

template <class T>
void Radix<T>::loadKernels(std::string source_dir)
{
    string radix_source = source_dir + "/RadixSort.cl";

    std::string options = "-D LOCAL_SIZE_LIMIT=512";
    cl::Program prog = cli->loadProgram(radix_source, options);
    //printf("radix sort\n");
    
    //printf("load scanNaive\n");
    k_scanNaive = Kernel(cli, prog, "scanNaive");
    //printf("load radixSortBlockKeysValues\n");
    //k_radixSortBlockKeysValues = Kernel(cli, prog, "radixSortBlockKeysValues");
    //printf("load radixSortBlocksKeysValues\n");
    k_radixSortBlocksKeysValues = Kernel(cli, prog, "radixSortBlocksKeysValues");
    //printf("load reorderDataKeysValues\n");
    k_reorderDataKeysValues = Kernel(cli, prog, "reorderDataKeysValues");
    //printf("load findRadixOffsets\n");
    k_findRadixOffsets = Kernel(cli, prog, "findRadixOffsets");

    string scan_source = source_dir + "/Scan_b.cl";

    options = "-D LOCAL_SIZE_LIMIT=512";
    prog = cli->loadProgram(scan_source, options);
    
    k_scanExclusiveLocal1 = Kernel(cli, prog, "scanExclusiveLocal1");
    k_scanExclusiveLocal2 = Kernel(cli, prog, "scanExclusiveLocal2");
    k_uniformUpdate = Kernel(cli, prog, "uniformUpdate");
    
 

    //TODO: implement this check with the C++ API    
    //if( (szRadixSortLocal < (LOCAL_SIZE_LIMIT / 2)) || (szRadixSortLocal1 < (LOCAL_SIZE_LIMIT / 2)) || (szRadixMergeLocal < (LOCAL_SIZE_LIMIT / 2)) ){
            //shrLog("\nERROR !!! Minimum work-group size %u required by this application is not supported on this device.\n\n", LOCAL_SIZE_LIMIT / 2);
}

template <class T>
void Radix<T>::sort(int num, Buffer<T>* keys, Buffer<T>* values)
{
        //printf("radix sort routine\n");
        this->keys = keys;
        this->values = values;
        int key_bits = sizeof(T) * 8;
        //printf("sorting %d\n", num);
        //printf("key_bits %d\n", key_bits);
        //printf("bit_step %d\n", bit_step);

        int i = 0;
        while(key_bits > i*bit_step)
        {
            //printf("i*bit_step %d\n", i*bit_step)
            step(bit_step, i*bit_step, num);
            i += 1;
        }
        cli->queue.finish();
}

template <class T>
void Radix<T>::step(int nbits, int startbit, int num)
{
        blocks(nbits, startbit, num);
        cli->queue.finish();

        find_offsets(startbit, num);
        cli->queue.finish();

        int array_length = num/2/cta_size*16;
        if(array_length < MIN_LARGE_ARRAY_SIZE)
        {
            naive_scan(num);
        }
        else
        {
            scan(&mCountersSum, &mCounters, 1, array_length);
        }
        cli->queue.finish();

        reorder(startbit, num);
        cli->queue.finish();
}

template <class T>
void Radix<T>::blocks(int nbits, int startbit, int num)
{
        int totalBlocks = num/4/cta_size;
        int global_size = cta_size*totalBlocks;
        int local_size = cta_size;

        int arg = 0;
        k_radixSortBlocksKeysValues.setArg(arg++, keys->getDevicePtr());
        k_radixSortBlocksKeysValues.setArg(arg++, values->getDevicePtr());
        k_radixSortBlocksKeysValues.setArg(arg++, d_tempKeys.getDevicePtr());
        k_radixSortBlocksKeysValues.setArg(arg++, d_tempValues.getDevicePtr());
        k_radixSortBlocksKeysValues.setArg(arg++, nbits);
        k_radixSortBlocksKeysValues.setArg(arg++, startbit);
        k_radixSortBlocksKeysValues.setArg(arg++, num);
        k_radixSortBlocksKeysValues.setArg(arg++, totalBlocks);
        k_radixSortBlocksKeysValues.setArgShared(arg++, 4 * cta_size * sizeof(T));
        k_radixSortBlocksKeysValues.setArgShared(arg++, 4 * cta_size * sizeof(T));

        k_radixSortBlocksKeysValues.execute(global_size, local_size);

}
template <class T>
void Radix<T>::find_offsets(int startbit, int num)
{
        int totalBlocks = num/2/cta_size;
        int global_size = cta_size*totalBlocks;
        int local_size = cta_size;
        int arg = 0;
        k_findRadixOffsets.setArg(arg++, d_tempKeys.getDevicePtr());
        k_findRadixOffsets.setArg(arg++, d_tempValues.getDevicePtr());
        k_findRadixOffsets.setArg(arg++, mCounters.getDevicePtr());
        k_findRadixOffsets.setArg(arg++, mBlockOffsets.getDevicePtr());
        k_findRadixOffsets.setArg(arg++, startbit);
        k_findRadixOffsets.setArg(arg++, num);
        k_findRadixOffsets.setArg(arg++, totalBlocks);
        k_findRadixOffsets.setArgShared(arg++, 2 * cta_size * sizeof(T));

        k_findRadixOffsets.execute(global_size, local_size);
}

template <class T>
void Radix<T>::naive_scan(int num)
{
        int nhist = num/2/cta_size*16;
        int global_size = nhist;
        int local_size = nhist;
        int extra_space = nhist / 16;// #NUM_BANKS defined as 16 in RadixSort.cpp (original NV implementation)
        int shared_mem_size = sizeof(T) * (nhist + extra_space);
        int arg = 0;
        k_scanNaive.setArg(arg++, mCountersSum.getDevicePtr()); 
        k_scanNaive.setArg(arg++, mCounters.getDevicePtr()); 
        k_scanNaive.setArg(arg++, nhist); 
        k_scanNaive.setArgShared(arg++, 2*shared_mem_size); 
        
        k_scanNaive.execute(global_size, local_size);
}
template <class T>
void Radix<T>::scan( Buffer<T>* dst, Buffer<T>* src, int batch_size, int array_length)
{
        scan_local1(dst, 
                    src, 
                    batch_size * array_length / (4 * SCAN_WG_SIZE), 
                    4 * SCAN_WG_SIZE);
        
        cli->queue.finish();
        scan_local2(dst, 
                    src, 
                    batch_size,
                    array_length / (4 * SCAN_WG_SIZE));
        cli->queue.finish();
        scan_update(dst, batch_size * array_length / (4 * SCAN_WG_SIZE));
        cli->queue.finish();

}

template <class T>
void Radix<T>::scan_local1( Buffer<T>* dst, Buffer<T>* src, int n, int size)
{
    int global_size = n * size / 4;
    int local_size = SCAN_WG_SIZE;
    int arg = 0;
    k_scanExclusiveLocal1.setArg(arg++, dst->getDevicePtr());
    k_scanExclusiveLocal1.setArg(arg++, src->getDevicePtr());
    k_scanExclusiveLocal1.setArgShared(arg++, 2 * SCAN_WG_SIZE * sizeof(T));
    k_scanExclusiveLocal1.setArg(arg++, size);
    
    k_scanExclusiveLocal1.execute(global_size, local_size);
}

template <class T>
void Radix<T>::scan_local2( Buffer<T>* dst, Buffer<T>* src, int n, int size)
{
    int elements = n * size;
    int dividend = elements;
    int divisor = SCAN_WG_SIZE;
    int global_size;
    if (dividend % divisor == 0)
        global_size = dividend;
    else
        global_size = dividend - dividend % divisor + divisor;
    int local_size = SCAN_WG_SIZE;

    int arg = 0;
    k_scanExclusiveLocal2.setArg(arg++, scan_buffer.getDevicePtr());
    k_scanExclusiveLocal2.setArg(arg++, dst->getDevicePtr());
    k_scanExclusiveLocal2.setArg(arg++, src->getDevicePtr());
    k_scanExclusiveLocal2.setArgShared(arg++, 2 * SCAN_WG_SIZE * sizeof(T));
    k_scanExclusiveLocal2.setArg(arg++, elements);
    k_scanExclusiveLocal2.setArg(arg++, size);
    k_scanExclusiveLocal2.execute(global_size, local_size);
}

template <class T>
void Radix<T>::scan_update( Buffer<T>* dst, int n)
{
    int global_size = n * SCAN_WG_SIZE;
    int local_size = SCAN_WG_SIZE;
    int arg = 0;
    k_uniformUpdate.setArg(arg++, dst->getDevicePtr());
    k_uniformUpdate.setArg(arg++, scan_buffer.getDevicePtr());
    k_uniformUpdate.execute(global_size, local_size);
}

template <class T>
void Radix<T>::reorder( int startbit, int num)
{
        int totalBlocks = num/2/cta_size;
        int global_size = cta_size*totalBlocks;
        int local_size = cta_size;
        int arg = 0;
        k_reorderDataKeysValues.setArg(arg++, keys->getDevicePtr());
        k_reorderDataKeysValues.setArg(arg++, values->getDevicePtr());
        k_reorderDataKeysValues.setArg(arg++, d_tempKeys.getDevicePtr());
        k_reorderDataKeysValues.setArg(arg++, d_tempValues.getDevicePtr());
        k_reorderDataKeysValues.setArg(arg++, mBlockOffsets.getDevicePtr());
        k_reorderDataKeysValues.setArg(arg++, mCountersSum.getDevicePtr());
        k_reorderDataKeysValues.setArg(arg++, mCounters.getDevicePtr());
        k_reorderDataKeysValues.setArg(arg++, startbit);
        k_reorderDataKeysValues.setArg(arg++, num);
        k_reorderDataKeysValues.setArg(arg++, totalBlocks);
        k_reorderDataKeysValues.setArgShared(arg++, 2 * cta_size * sizeof(T));
        k_reorderDataKeysValues.setArgShared(arg++, 2 * cta_size * sizeof(T));
        k_reorderDataKeysValues.execute(global_size, local_size);
}


#if 0
if __name__ == "__main__":

    n = 1048576*2
    #n = 32768*2
    #n = 16384
    #n = 8192
    hashes = np.ndarray((n,1), dtype=np.uint32)
    indices = np.ndarray((n,1), dtype=np.uint32)
    
    for i in xrange(0,n): 
        hashes[i] = n - i
        indices[i] = i
    
    npsorted = np.sort(hashes,0)

    print "hashes before:", hashes[0:20].T
    print "indices before: ", indices[0:20].T 

    radix = Radix(n, 128, hashes.dtype)
    #num_to_sort = 32768
    num_to_sort = n
    hashes, indices = radix.sort(num_to_sort, hashes, indices)

    print "hashes after:", hashes[0:20].T
    print "indices after: ", indices[0:20].T 

    print np.linalg.norm(hashes - npsorted)

    print timings


#endif





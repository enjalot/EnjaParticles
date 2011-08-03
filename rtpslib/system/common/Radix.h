#ifndef RTPS_RADIX_SORT_H
#define RTPS_RADIX_SORT_H

#include "CLL.h"
#include "Kernel.h"
#include "Buffer.h"

#include <vector>
using namespace std;

#ifndef uint
#define uint unsigned int
#endif


namespace rtps 
{

template <class T>
class Radix 
{
public:
    static const uint LOCAL_SIZE_LIMIT = 512U;
    Radix(){ cli=NULL; };
    //create an OpenCL buffer from existing data
    Radix( std::string source_dir, CL *cli, int max_elements, int cta_size );

    void loadKernels(std::string source_dir);

    void sort(int num, Buffer<T>* keys, Buffer<T>* values);
    void step(int nbits, int startbit, int num);
    void blocks(int nbits, int startbit, int num);
    void find_offsets(int startbit, int num);
    void naive_scan(int num);

    void scan( Buffer<T>* dst, Buffer<T>* src, int batch_size, int array_length);
    void scan_local1( Buffer<T>* dst, Buffer<T>* src, int n, int size);
    void scan_local2( Buffer<T>* dst, Buffer<T>* src, int n, int size);
    void scan_update( Buffer<T>* dst, int n);
    void reorder( int startbit, int num);

private:
    Kernel k_scanNaive;
    //Kernel k_radixSortBlockKeysValues; 
    Kernel k_radixSortBlocksKeysValues;
    Kernel k_reorderDataKeysValues;
    Kernel k_findRadixOffsets;
    Kernel k_scanExclusiveLocal1;
    Kernel k_scanExclusiveLocal2;
    Kernel k_uniformUpdate;
 

    CL *cli;

    int WARP_SIZE;
    int SCAN_WG_SIZE;
    int MIN_LARGE_ARRAY_SIZE;
    int bit_step;
    int cta_size;
    size_t uintsz;

    Buffer<T>* keys;
    Buffer<T>* values;
    //these should probably be type T
    Buffer<unsigned int> d_tempKeys;
    Buffer<unsigned int> d_tempValues;
    Buffer<unsigned int> mCounters;
    Buffer<unsigned int> mCountersSum;
    Buffer<unsigned int> mBlockOffsets;
    Buffer<unsigned int> scan_buffer;

};


#include "Radix.cpp"
}
#endif

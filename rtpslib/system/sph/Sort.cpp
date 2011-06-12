#include "SPH.h"

namespace rtps
{

    void SPH::bitonic_sort()
    {
        try
        {
            int dir = 1;        // dir: direction
            //int batch = num;

            int arrayLength = nlpo2(num);
            //printf("num: %d\n", num);
            //printf("nlpo2(num): %d\n", arrayLength);
            //int arrayLength = max_num;
            //int batch = max_num / arrayLength;
            int batch = 1;

            //printf("about to try sorting\n");
            bitonic.Sort(batch, 
                        arrayLength, 
                        dir,
                        &cl_sort_output_hashes,
                        &cl_sort_output_indices,
                        &cl_sort_hashes,
                        &cl_sort_indices );

        }
        catch (cl::Error er)
        {
            printf("ERROR(bitonic sort): %s(%s)\n", er.what(), oclErrorString(er.err()));
            exit(0);
        }

        ps->cli->queue.finish();

        /*
        int nbc = 10;
        std::vector<int> sh = cl_sort_hashes.copyToHost(nbc);
        std::vector<int> eci = cl_cell_indices_end.copyToHost(nbc);
    
        for(int i = 0; i < nbc; i++)
        {
            printf("before[%d] %d eci: %d\n; ", i, sh[i], eci[i]);
        }
        printf("\n");
        */


        cl_sort_hashes.copyFromBuffer(cl_sort_output_hashes, 0, 0, num);
        cl_sort_indices.copyFromBuffer(cl_sort_output_indices, 0, 0, num);

        /*
        scopy(num, cl_sort_output_hashes.getDevicePtr(), 
              cl_sort_hashes.getDevicePtr());
        scopy(num, cl_sort_output_indices.getDevicePtr(), 
              cl_sort_indices.getDevicePtr());
        */

        ps->cli->queue.finish();
#if 0
    
        printf("********* Bitonic Sort Diagnostics **************\n");
        int nbc = 20;
        //sh = cl_sort_hashes.copyToHost(nbc);
        //eci = cl_cell_indices_end.copyToHost(nbc);
        std::vector<unsigned int> sh = cl_sort_hashes.copyToHost(nbc);
        std::vector<unsigned int> si = cl_sort_indices.copyToHost(nbc);
        //std::vector<int> eci = cl_cell_indices_end.copyToHost(nbc);

    
        for(int i = 0; i < nbc; i++)
        {
            //printf("after[%d] %d eci: %d\n; ", i, sh[i], eci[i]);
            printf("sh[%d] %d si: %d\n ", i, sh[i], si[i]);
        }

#endif


    }
	//----------------------------------------------------------------------
    void SPH::cloud_bitonic_sort()    // GE
    {
        try
        {
            int dir = 1;        // dir: direction
            //int batch = num;

            int arrayLength = nlpo2(num);
            //printf("num: %d\n", num);
            //printf("nlpo2(num): %d\n", arrayLength);
            //int arrayLength = max_num;
            //int batch = max_num / arrayLength;
            int batch = 1;

            //printf("about to try sorting\n");
            bitonic.Sort(batch, 
                        arrayLength,  //GE?? ???
                        dir,
                        &cl_cloud_sort_output_hashes,
                        &cl_cloud_sort_output_indices,
                        &cl_cloud_sort_hashes,
                        &cl_cloud_sort_indices );

        }
        catch (cl::Error er)
        {
            printf("ERROR(bitonic sort): %s(%s)\n", er.what(), oclErrorString(er.err()));
            exit(0);
        }

        ps->cli->queue.finish();

        /*
        int nbc = 10;
        std::vector<int> sh = cl_sort_hashes.copyToHost(nbc);
        std::vector<int> eci = cl_cell_indices_end.copyToHost(nbc);
    
        for(int i = 0; i < nbc; i++)
        {
            printf("before[%d] %d eci: %d\n; ", i, sh[i], eci[i]);
        }
        printf("\n");
        */


		// NOT SURE HOW THIS WORKS!! GE
        cl_cloud_sort_hashes.copyFromBuffer(cl_cloud_sort_output_hashes, 0, 0, cloud_num);
        cl_cloud_sort_indices.copyFromBuffer(cl_cloud_sort_output_indices, 0, 0, cloud_num);

        /*
        scopy(num, cl_sort_output_hashes.getDevicePtr(), 
              cl_sort_hashes.getDevicePtr());
        scopy(num, cl_sort_output_indices.getDevicePtr(), 
              cl_sort_indices.getDevicePtr());
        */

        ps->cli->queue.finish();
#if 0
    
        printf("********* Bitonic Sort Diagnostics **************\n");
        int nbc = 20;
        //sh = cl_sort_hashes.copyToHost(nbc);
        //eci = cl_cell_indices_end.copyToHost(nbc);
        std::vector<unsigned int> sh = cl_cloud_sort_hashes.copyToHost(nbc);
        std::vector<unsigned int> si = cl_cloud_sort_indices.copyToHost(nbc);
        //std::vector<int> eci = cl_cell_indices_end.copyToHost(nbc);

    
        for(int i = 0; i < nbc; i++)
        {
            //printf("after[%d] %d eci: %d\n; ", i, sh[i], eci[i]);
            printf("sh[%d] %d si: %d\n ", i, sh[i], si[i]);
        }

#endif

    }

    //void SPH::radix_sort()
    //{
    //}
}


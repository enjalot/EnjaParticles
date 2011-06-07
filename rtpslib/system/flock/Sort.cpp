#include "FLOCK.h"

namespace rtps
{

    void FLOCK::bitonic_sort()
    {
        try
        {
            int dir = 1;        // dir: direction
            
            int arrayLength = max_num;
            int batch = max_num / arrayLength;

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

        cl_sort_hashes.copyFromBuffer(cl_sort_output_hashes, 0, 0, num);
        cl_sort_indices.copyFromBuffer(cl_sort_output_indices, 0, 0, num);

        ps->cli->queue.finish();
    }
}

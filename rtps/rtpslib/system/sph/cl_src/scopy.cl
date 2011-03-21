

__kernel void scopy(int n, 
                    __global float* sxorig, 
                    __global float* sydest)
{
    int i, tid, totalThreads, ctaStart;

    tid = get_local_id(0);
    int locsiz = get_local_size(0);
    int gridDimx = get_num_groups(0);
    int gid = get_group_id(0);

    totalThreads = gridDimx * locsiz;
    ctaStart = locsiz*gid; 

    for (i = ctaStart + tid; i < n; i += totalThreads)
    {
        sydest[i] = sxorig[i];
    }
}
//----------------------------------------------------------------------

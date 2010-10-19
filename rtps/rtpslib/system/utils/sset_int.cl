

__kernel void sset_int(int n, int val, 
				  __global int* sxdest)
{
	return;

	#if 0
    int i, tid, totalThreads, ctaStart;

    tid = get_local_id(0);
	int locsiz = get_local_size(0);
	int gridDimx = get_num_groups(0);
	int gid = get_group_id(0);

    totalThreads = gridDimx * locsiz;
    ctaStart = locsiz*gid; 

	for (i = ctaStart + tid; i < n; i += totalThreads) {
		sxdest[i] = *val;
	}
	#endif
}
//----------------------------------------------------------------------

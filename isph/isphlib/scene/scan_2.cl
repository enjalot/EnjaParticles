__kernel void scanExclusiveLocal2
(
    __global uint *d_Buf		: SCAN_BUFFER,
    __global const uint *d_Dst	: RADIX_COUNTERS_SUM,
    __global const uint *d_Src	: RADIX_COUNTERS,
    __local uint* l_Data		: LOCAL_SIZE_UINT2,
    uint N						: SCAN_SIZE
)
{
	uint data;
	int globalId = get_global_id(0);
	int localSize4 = 4*get_local_size(0);
    
	if(globalId < N)
		data = d_Dst[(localSize4-1) + localSize4*globalId] + d_Src[(localSize4-1) + localSize4*globalId];
	else
		data = 0;

    uint odata = scan1Inclusive(data, l_Data, N) - data;

    if(globalId < N)
        d_Buf[globalId] = odata;
}

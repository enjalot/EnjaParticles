uint scan1Inclusive(uint idata, __local uint *l_Data, uint size)
{
    uint pos = 2 * get_local_id(0) - (get_local_id(0) & (size - 1));
    l_Data[pos] = 0;
    pos += size;
    l_Data[pos] = idata;

    for(uint offset = 1; offset < size; offset <<= 1)
	{
        barrier(CLK_LOCAL_MEM_FENCE);
        uint t = l_Data[pos] + l_Data[pos - offset];
        barrier(CLK_LOCAL_MEM_FENCE);
        l_Data[pos] = t;
	}

	return l_Data[pos];
}

uint4 scan4Inclusive(uint4 data4, __local uint *l_Data, uint size)
{
    data4.y += data4.x;
    data4.z += data4.y;
    data4.w += data4.z;

    uint val = scan1Inclusive(data4.w, l_Data, size / 4) - data4.w;
    return (data4 + (uint4)val);
}

uint4 scan4Exclusive(uint4 data4, __local uint *l_Data, uint size)
{
    return scan4Inclusive(data4, l_Data, size) - data4;
}

__kernel void scanExclusiveLocal1
(
    __global uint4 *d_Dst		: RADIX_COUNTERS_SUM,
    __global const uint4 *d_Src	: RADIX_COUNTERS,
    __local uint* l_Data		: LOCAL_SIZE_UINT2
)
{
	size_t globalId = get_global_id(0);
    uint4 idata4 = d_Src[globalId];
    uint4 odata4 = scan4Exclusive(idata4, l_Data, 4*get_local_size(0));
    d_Dst[globalId] = odata4;
}

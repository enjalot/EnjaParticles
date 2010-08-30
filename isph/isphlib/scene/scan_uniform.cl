__kernel void uniformUpdate
(
    __global uint4 *d_Data		: RADIX_COUNTERS_SUM,
    __global const uint *d_Buf	: SCAN_BUFFER
)
{
    __local uint buf[1];

    if(get_local_id(0) == 0)
        buf[0] = d_Buf[get_group_id(0)];

    barrier(CLK_LOCAL_MEM_FENCE);

    d_Data[get_global_id(0)] += (uint4)buf[0];
}

__kernel void FindRadixOffsets
(
	__global const uint2* keys	: CELLS_HASH_TEMP,
	__global uint* counters		: RADIX_COUNTERS,
	__global uint* blockOffsets	: RADIX_BLOCK_OFFSETS,
	uint startbit				: RADIX_STARTBIT,
	uint totalBlocks			: RADIX_BLOCK_COUNT,
	__local uint* sRadix1		: LOCAL_SIZE_UINT2
)
{
	__local uint  sStartPointers[16];

    uint groupId = get_group_id(0);
    uint localId = get_local_id(0);
    uint groupSize = get_local_size(0);

    uint2 radix2 = keys[get_global_id(0)];
        

    sRadix1[2 * localId]     = (radix2.x >> startbit) & 0xF;
    sRadix1[2 * localId + 1] = (radix2.y >> startbit) & 0xF;

    if(localId < 16) 
    {
        sStartPointers[localId] = 0; 
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if((localId > 0) && (sRadix1[localId] != sRadix1[localId - 1]) ) 
    {
        sStartPointers[sRadix1[localId]] = localId;
    }
    if(sRadix1[localId + groupSize] != sRadix1[localId + groupSize - 1]) 
    {
        sStartPointers[sRadix1[localId + groupSize]] = localId + groupSize;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if(localId < 16) 
    {
        blockOffsets[groupId*16 + localId] = sStartPointers[localId];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if((localId > 0) && (sRadix1[localId] != sRadix1[localId - 1]) ) 
    {
        sStartPointers[sRadix1[localId - 1]] = 
            localId - sStartPointers[sRadix1[localId - 1]];
    }
    if(sRadix1[localId + groupSize] != sRadix1[localId + groupSize - 1] ) 
    {
        sStartPointers[sRadix1[localId + groupSize - 1]] = 
            localId + groupSize - sStartPointers[sRadix1[localId + groupSize - 1]];
    }
        

    if(localId == groupSize - 1) 
    {
        sStartPointers[sRadix1[2 * groupSize - 1]] = 
            2 * groupSize - sStartPointers[sRadix1[2 * groupSize - 1]];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if(localId < 16) 
    {
        counters[localId * totalBlocks + groupId] = sStartPointers[localId];
    }
}

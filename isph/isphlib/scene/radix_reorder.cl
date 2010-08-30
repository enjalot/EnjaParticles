__kernel void RadixReorderData
(
	__global uint *outKeys				: CELLS_HASH, 
	__global const uint2 *keys			: CELLS_HASH_TEMP,
	__global uint *outValues			: HASHES_PARTICLE, 
	__global const uint2 *values		: HASHES_PARTICLE_TEMP,
	__global const uint *blockOffsets	: RADIX_BLOCK_OFFSETS, 
	__global const uint *offsets		: RADIX_COUNTERS_SUM, 
	uint startbit						: RADIX_STARTBIT,
	uint numElements					: ALLOCATED_PARTICLE_COUNT,
	uint totalBlocks					: RADIX_BLOCK_COUNT,
	__local uint2* sKeys2				: LOCAL_SIZE_UINT2,
	__local uint2* sValues2				: LOCAL_SIZE_UINT2
)
{
	__local uint sOffsets[16];
	__local uint sBlockOffsets[16];

	__local uint *sKeys1 = (__local uint*)sKeys2;
	__local uint *sValues1 = (__local uint*)sValues2; 

	uint groupId = get_group_id(0);
	uint globalId = get_global_id(0);
	uint localId = get_local_id(0);
	uint groupSize = get_local_size(0);

	sKeys2[localId]   = keys[globalId];
	sValues2[localId] = values[globalId];

	if(localId < 16)  
	{
		sOffsets[localId]      = offsets[localId * totalBlocks + groupId];
		sBlockOffsets[localId] = blockOffsets[groupId * 16 + localId];
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	int radix = (sKeys1[localId] >> startbit) & 0xF;
	int globalOffset = sOffsets[radix] + localId - sBlockOffsets[radix];
	
	if (globalOffset < numElements)
	{
		outKeys[globalOffset] = sKeys1[localId];
		outValues[globalOffset] = sValues1[localId];
	}

	radix = (sKeys1[localId + groupSize] >> startbit) & 0xF;
	globalOffset = sOffsets[radix] + localId + groupSize - sBlockOffsets[radix];

	if (globalOffset < numElements)
	{
		outKeys[globalOffset]   = sKeys1[localId + groupSize];
		outValues[globalOffset] = sValues1[localId + groupSize];
	}
}

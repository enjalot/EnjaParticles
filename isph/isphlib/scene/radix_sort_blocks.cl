#define WARP_SIZE 32
#define BITSTEP 4

uint scanwarp(uint val, __local uint* sData, int maxlevel)
{
    int localId = get_local_id(0);
    int idx = 2 * localId - (localId & (WARP_SIZE - 1));
    sData[idx] = 0;
    idx += WARP_SIZE;
    sData[idx] = val;     

    if (0 <= maxlevel) { sData[idx] += sData[idx - 1]; }
    if (1 <= maxlevel) { sData[idx] += sData[idx - 2]; }
    if (2 <= maxlevel) { sData[idx] += sData[idx - 4]; }
    if (3 <= maxlevel) { sData[idx] += sData[idx - 8]; }
    if (4 <= maxlevel) { sData[idx] += sData[idx -16]; }

    return sData[idx] - val;
}

uint4 scan4(uint4 idata, __local uint* ptr)
{    
    
    uint idx = get_local_id(0);

    uint4 val4 = idata;
    uint sum[3];
    sum[0] = val4.x;
    sum[1] = val4.y + sum[0];
    sum[2] = val4.z + sum[1];
    
    uint val = val4.w + sum[2];
    
    val = scanwarp(val, ptr, 4);
    barrier(CLK_LOCAL_MEM_FENCE);

    if ((idx & (WARP_SIZE - 1)) == WARP_SIZE - 1)
    {
        ptr[idx >> 5] = val + val4.w + sum[2];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

	if (idx < WARP_SIZE)
		ptr[idx] = scanwarp(ptr[idx], ptr, 2);
    
    barrier(CLK_LOCAL_MEM_FENCE);

    val += ptr[idx >> 5];

    val4.x = val;
    val4.y = val + sum[0];
    val4.z = val + sum[1];
    val4.w = val + sum[2];

    return val4;
}

uint4 rank4(uint4 preds, __local uint* sMem)
{
	int localId = get_local_id(0);
	int localSize = get_local_size(0);

	uint4 address = scan4(preds, sMem);
	
	__local uint numtrue[1];
	if (localId == localSize - 1) 
	{
		numtrue[0] = address.w + preds.w;
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	
	uint4 rank;
	int idx = localId*4;
	rank.x = (preds.x) ? address.x : numtrue[0] + idx - address.x;
	rank.y = (preds.y) ? address.y : numtrue[0] + idx + 1 - address.y;
	rank.z = (preds.z) ? address.z : numtrue[0] + idx + 2 - address.z;
	rank.w = (preds.w) ? address.w : numtrue[0] + idx + 3 - address.w;
	
	return rank;
}

__kernel void RadixSortBlocks
(
	__global const uint4* keysIn	: CELLS_HASH, 
	__global uint4* keysOut 		: CELLS_HASH_TEMP,
	__global const uint4* valuesIn	: HASHES_PARTICLE,
	__global uint4* valuesOut		: HASHES_PARTICLE_TEMP,
	uint startbit					: RADIX_STARTBIT,
	__local uint* sMem				: LOCAL_SIZE_UINT4
)
{
	int globalId = get_global_id(0);
	int localId = get_local_id(0);
    int localSize = get_local_size(0);
	
	uint4 key = keysIn[globalId];
	uint4 value = valuesIn[globalId];
	
	barrier(CLK_LOCAL_MEM_FENCE);
	
	for(uint shift = startbit; shift < (startbit + BITSTEP); ++shift)
	{
		uint4 lsb;
		lsb.x = !((key.x >> shift) & 0x1);
		lsb.y = !((key.y >> shift) & 0x1);
        lsb.z = !((key.z >> shift) & 0x1);
        lsb.w = !((key.w >> shift) & 0x1);
        
		uint4 r = rank4(lsb, sMem);

        sMem[(r.x & 3) * localSize + (r.x >> 2)] = key.x;
        sMem[(r.y & 3) * localSize + (r.y >> 2)] = key.y;
        sMem[(r.z & 3) * localSize + (r.z >> 2)] = key.z;
        sMem[(r.w & 3) * localSize + (r.w >> 2)] = key.w;
        barrier(CLK_LOCAL_MEM_FENCE);

        key.x = sMem[localId];
        key.y = sMem[localId +     localSize];
        key.z = sMem[localId + 2 * localSize];
        key.w = sMem[localId + 3 * localSize];
		barrier(CLK_LOCAL_MEM_FENCE);
		
		sMem[(r.x & 3) * localSize + (r.x >> 2)] = value.x;
        sMem[(r.y & 3) * localSize + (r.y >> 2)] = value.y;
        sMem[(r.z & 3) * localSize + (r.z >> 2)] = value.z;
        sMem[(r.w & 3) * localSize + (r.w >> 2)] = value.w;
        barrier(CLK_LOCAL_MEM_FENCE);

        value.x = sMem[localId];
        value.y = sMem[localId +     localSize];
        value.z = sMem[localId + 2 * localSize];
        value.w = sMem[localId + 3 * localSize];
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	
	keysOut[globalId] = key;
	valuesOut[globalId] = value;
}

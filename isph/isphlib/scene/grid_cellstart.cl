__kernel void FindCellStart
(
	__global const uint *sortedHash	: CELLS_HASH,
	__global uint *cellsStart		: CELLS_START,
	__local uint *localHash			: LOCAL_SIZE_UINT
)
{
    size_t i = get_global_id(0);
	size_t j = get_local_id(0);
	uint hash = sortedHash[i];
	localHash[j+1] = hash;
	
	if(i>0 && j==0)
		localHash[0] = sortedHash[i-1];

    barrier(CLK_LOCAL_MEM_FENCE);
	
	if((i==0 || hash!=localHash[j]) && hash!=UINT_MAX)
		cellsStart[hash] = i;
}

__kernel void ComputeCellIds
(
	__global uint *hash : CELLS_HASH,
	__global uint *particle : HASHES_PARTICLE,
	__global const vector *pos : POSITIONS,
	uint particleCount : PARTICLE_COUNT,
	uint2 cellCount : CELL_COUNT,
	vector gridStart : GRID_START,
	scalar cellSizeInv : CELL_SIZE_INV
)
{
    uint i = get_global_id(0);
	if(i < particleCount)
	{
		hash[i] = CellHash(CellPos(pos[i], gridStart, cellSizeInv), cellCount);
		particle[i] = i;
	}
	else hash[i] = UINT_MAX;
}

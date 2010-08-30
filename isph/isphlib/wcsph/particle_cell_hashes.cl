/*!
 *	\brief	Compute number of neigbours of each particle
 */
__kernel void ParticelCellHashes
(
	__global scalar *c  : PARTICLE_CELL_HASH,
	__global const vector *pos : POSITIONS,
	__global const uint *cellsStart : CELLS_START,
	__global const uint *hashes : CELLS_HASH,
	__global const uint *particles : HASHES_PARTICLE,
	uint particleCount : PARTICLE_COUNT,
	vector gridStart : GRID_START,
	uint2 cellCount : CELL_COUNT,
	scalar cellSizeInv : CELL_SIZE_INV
)
{
	size_t i = get_global_id(0);
	if(i >= particleCount) return;
	c[i] = hashes[i];
}

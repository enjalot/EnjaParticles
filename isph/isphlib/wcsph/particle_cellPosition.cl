/*!
 *	\brief	Compute xy in cell of each particle
 */
__kernel void CellPosition
(
	__global vector *c1  : CELL_POSITION,
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
	vector posI = pos[i];
	vector c= floor((posI - gridStart) * cellSizeInv);
	c1[i].x = (scalar)c.x;
	c1[i].y = (scalar)c.y;
}

/*!
 *	\brief	Shepard filter - pre step
 */
__kernel void ShepardPre
(
	__global scalar *weight : DENSITY_TMP,
	__global const scalar *density : DENSITIES,
	__global const scalar *mass : MASSES,
	__global const vector *pos : POSITIONS,
	__global const uint *cellsStart : CELLS_START,
	__global const uint *hashes : CELLS_HASH,
	__global const uint *particles : HASHES_PARTICLE,
	uint particleCount : PARTICLE_COUNT,
	scalar2 h : SMOOTHING_LENGTH_INV,
	vector gridStart : GRID_START,
	uint2 cellCount : CELL_COUNT,
	scalar cellSizeInv: CELL_SIZE_INV
)
{
	size_t i = get_global_id(0);
	if(i >= particleCount) return;
	vector posI = pos[i];
	scalar tmp = 0;
	
	ForEachSetup(posI,gridStart,cellSizeInv,cellCount)
	ForEachNeighbor(cellCount,hashes,particles,cellsStart)
		scalar W = SphKernel(posI-pos[j], h.x, h.y);
		tmp += W * mass[j] / density[j]; 
	ForEachEnd
	
    weight[i] = tmp + SphKernel((vector)0, h.x, h.y) * mass[i] / density[i];
	
}

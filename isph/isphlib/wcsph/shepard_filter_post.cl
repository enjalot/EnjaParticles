/*!
 *	\brief	Shepard filter - post step
 */
__kernel void ShepardPost
(
	__global scalar *density : DENSITIES,
	__global scalar *mass : MASSES,
	__global const scalar *weight : DENSITY_TMP,
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
	scalar densityI = 0;
	scalar weightInv = 1/weight[i];

	ForEachSetup(posI,gridStart,cellSizeInv,cellCount)
	ForEachNeighbor(cellCount,hashes,particles,cellsStart)
		scalar W = SphKernel(posI-pos[j], h.x, h.y);
		densityI += mass[j] * W * weightInv; 
	ForEachEnd 

	density[i] = densityI + SphKernel((vector)0, h.x, h.y) * mass[i] * weightInv;

}

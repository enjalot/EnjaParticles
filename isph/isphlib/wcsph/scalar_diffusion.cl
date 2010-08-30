/*!
 *	\brief	Compute diffusion of passive scalar
 */
__kernel void ScalarDiffusion
(
	__global scalar *c  : PASSIVE_SCALAR,
	__global const vector *pos : POSITIONS,
	__global const scalar *density : DENSITIES,
	__global const uint *cellsStart : CELLS_START,
	__global const uint *hashes : CELLS_HASH,
	__global const uint *particles : HASHES_PARTICLE,
	uint particleCount : PARTICLE_COUNT,
	vector gridStart : GRID_START,
	uint2 cellCount : CELL_COUNT,
	scalar cellSizeInv : CELL_SIZE_INV,
	scalar particleMass : MASS,
	scalar2 h : SMOOTHING_LENGTH_INV,
	scalar diffusivity : DIFFUSIVITY,
	scalar dt : TIME_STEP
)
{
	size_t i = get_global_id(0);
	if(i >= particleCount) return;
	vector posI = pos[i];
	scalar densityI = density[i];
	scalar cI = c[i];
	
	scalar dc =0;
	
	ForEachSetup(posI,gridStart,cellSizeInv,cellCount)
	ForEachNeighbor(cellCount,hashes,particles,cellsStart)
		vector posDif = posI-pos[j];
  	    vector gradW = SphKernelGrad(posDif, h.x, h.y);
		//  rate of change
		dc +=  (cI-c[j]) * dot(gradW, posDif) / (density[j] * length(posDif));           
	ForEachEnd

	c[i] += 2.0 * particleMass * dc * dt;
}

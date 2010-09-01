/*!
 *	\brief	XSPH velocity correction, Monaghan JCP 2000
 */
__kernel void Xsph
(
	__global vector *xsphVel : XSPH_VELOCITIES,
	__global const vector *vel : VELOCITIES,
	__global const vector *pos : POSITIONS,
	__global const scalar *density : DENSITIES,
	__global const scalar *mass : MASSES,
	__global const uint *cellsStart : CELLS_START,
	__global const uint *hashes : CELLS_HASH,
	__global const uint *particles : HASHES_PARTICLE,
	uint fluidParticleCount : FLUID_PARTICLE_COUNT,
	vector gridStart : GRID_START,
	uint2 cellCount : CELL_COUNT,
	scalar cellSizeInv : CELL_SIZE_INV,
	scalar2 h : SMOOTHING_LENGTH_INV,
	scalar xsphFactor : XSPH_FACTOR
)
{
	size_t i = get_global_id(0);

	if(i < fluidParticleCount) {

	scalar densityI = density[i];
	vector posI = pos[i];
	vector velI = vel[i];
	vector corr = (vector)0;
	
	ForEachSetup(posI,gridStart,cellSizeInv,cellCount)
	ForEachNeighbor(cellCount,hashes,particles,cellsStart)
		if(j < fluidParticleCount)
			corr += mass[j] * (vel[j] - velI) * SphKernel(posI - pos[j], h.x, h.y) / (densityI + density[j]);
	ForEachEnd
	
	xsphVel[i] = velI + 2.0f * xsphFactor * corr;

	}
	else xsphVel[i] = vel[i];
}

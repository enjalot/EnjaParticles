/*!
 *	\brief	Compute viscous accelerations for each particle - Laminar Formulation
 */
__kernel void Viscosity
(
	__global vector *acc: ACCELERATIONS,
	__global const vector *vel : VELOCITIES,
	__global const vector *pos : POSITIONS,
	__global const scalar *density : DENSITIES,
	__global const scalar *mass : MASSES,
	__global const uint *cellsStart : CELLS_START,
	__global const uint *hashes : CELLS_HASH,
	__global const uint *particles : HASHES_PARTICLE,
	uint particleCount : PARTICLE_COUNT,
	vector gridStart : GRID_START,
	uint2 cellCount : CELL_COUNT,
	scalar cellSizeInv : CELL_SIZE_INV,
	scalar2 h : SMOOTHING_LENGTH_INV,
	scalar visc : DYNAMIC_VISCOSITY,
	scalar distEpsilon : DIST_EPSILON
)
{
	size_t i = get_global_id(0);
	if(i >= particleCount) return;
	vector posI = pos[i];
	vector v = vel[i];
	scalar densityI = density[i];
	vector a = (vector)0;
	
	ForEachSetup(posI,gridStart,cellSizeInv,cellCount)
	ForEachNeighbor(cellCount,hashes,particles,cellsStart)
	
		vector posDif = posI-pos[j];
		vector gradW = SphKernelGrad(posDif, h.x, h.y);	
		// viscous acceleration
		a +=  (4.*mass[j]*visc*(v-vel[j]) * dot(gradW, posDif) / ((densityI + density[j]) * (dot(posDif,posDif) + distEpsilon)));           
	ForEachEnd
	acc[i] += a;
}

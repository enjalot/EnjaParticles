/*!
 *	\brief	Calculate density and correct the pressure
 */
__kernel void CorrectPressure
(
	__global scalar *density		: DENSITIES,
	__global scalar *pressure		: PRESSURES,
	__global scalar *pods			: PODS,
	__global const vector *vel		: VELOCITIES,
	__global const vector *pos		: POSITIONS,
	__global const scalar *mass		: MASSES,
	__global const uint *cellsStart	: CELLS_START,
	__global const uint *hashes		: CELLS_HASH,
	__global const uint *particles	: HASHES_PARTICLE,
	uint particleCount 				: PARTICLE_COUNT,
	vector gridStart 				: GRID_START,
	uint2 cellCount					: CELL_COUNT,
	scalar cellSizeInv				: CELL_SIZE_INV,
	scalar2 h						: SMOOTHING_LENGTH_INV,
	scalar restDensity				: DENSITY,
	scalar restDensityInv			: DENSITY_INV,
	scalar dt 						: TIME_STEP,
	scalar delta					: DELTA,
	scalar distEpsilon				: DIST_EPSILON
)
{
	size_t i = get_global_id(0);
	if(i >= particleCount) return;
	
	vector posI = pos[i];
	vector velI = vel[i];
	scalar massI = mass[i];
	scalar densityI = 0;
	
	ForEachSetup(posI,gridStart,cellSizeInv,cellCount)
	ForEachNeighbor(cellCount,hashes,particles,cellsStart)
	
		densityI += mass[j]*SphKernel(posI-pos[j], h.x, h.y);

	ForEachEnd
	densityI += massI*SphKernel((vector)(0), h.x, h.y);

	scalar densityError = densityI - restDensity;
	scalar correctedPressure = pressure[i] + densityError * delta / (massI*massI * dt*dt + distEpsilon);

	density[i] = densityI;
	pressure[i] = correctedPressure;
	pods[i] = correctedPressure / (densityI * densityI + distEpsilon);
}

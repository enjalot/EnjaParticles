/*!
 *	\brief	Calculate viscosity and gravity accelerations, reset some vars
 */
__kernel void Prepare
(
	__global vector *acc_rest		: ACCEL_REST,
	__global vector *acc_p			: ACCEL_PRESSURE,
	__global scalar *pressure		: PRESSURES,
	__global vector *vel_tmp		: VELOCITIES_TMP,
	__global vector *pos_tmp		: POSITIONS_TMP,
	__global const vector *vel		: VELOCITIES,
	__global vector *pos			: POSITIONS,
	__global const scalar *density	: DENSITIES,
	__global const scalar *mass		: MASSES,
	__global const uint *cellsStart	: CELLS_START,
	__global const uint *hashes		: CELLS_HASH,
	__global const uint *particles	: HASHES_PARTICLE,
	uint particleCount 				: PARTICLE_COUNT,
	uint fluidParticleCount 		: FLUID_PARTICLE_COUNT,
	vector gridStart 				: GRID_START,
	uint2 cellCount					: CELL_COUNT,
	scalar cellSizeInv				: CELL_SIZE_INV,
	scalar hh 						: SMOOTHING_LENGTH,
	scalar2 h						: SMOOTHING_LENGTH_INV,
	scalar distEpsilon				: DIST_EPSILON,
	scalar alpha					: ALPHA_VISCOSITY,
	scalar soundSpeed				: SOUND_SPEED,
	scalar restDensityInv			: DENSITY_INV,
	vector gravity					: GRAVITY,
	scalar dt 						: TIME_STEP
)
{
	size_t i = get_global_id(0);
	if(i >= particleCount) return;
	
	// if boundary, advance immediately so it's not needed in convergence loop
	if(i >= fluidParticleCount)
	{
		//pos[i] += vel[i] * dt;
		return;
	}
	
	vector posI = pos[i];
	vector velI = vel[i];
	scalar densityI = density[i];
	scalar csI = densityI * restDensityInv;
	csI *= soundSpeed * csI * csI;
	
	vector a = gravity;
	
	ForEachSetup(posI,gridStart,cellSizeInv,cellCount)
	ForEachNeighbor(cellCount,hashes,particles,cellsStart)

		scalar densityJ = density[j];
		vector posDif = posI - pos[j];
	    scalar csJ = densityJ * restDensityInv;
	    csJ *= soundSpeed * csJ * csJ;

		a -= (alpha * (csI + csJ) / (densityI + densityJ) * hh * min(dot(posDif,velI-vel[j]), (scalar)0) / (dot(posDif,posDif) + distEpsilon)) * mass[j] * SphKernelGrad(posDif, h.x, h.y);

	ForEachEnd

	acc_rest[i] = a;
	acc_p[i] = (vector)0;
	pressure[i] = (scalar)0;
	vel_tmp[i] = vel[i];
	pos_tmp[i] = pos[i];
}

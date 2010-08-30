// cos ATI has a weird bug with abs
scalar absolute(scalar x)
{
	return x>0 ? x : -x;
}

/*!
 *	\brief	Suggest next time step according to CFL
 */
__kernel void CflTimeStep
(
	__global const vector *vel : VELOCITIES,
	__global const vector *pos : POSITIONS,
	__global const scalar *density : DENSITIES,
	__global const vector *acc : ACCELERATIONS,
	__global const uint *cellsStart : CELLS_START,
	__global const uint *hashes : CELLS_HASH,
	__global const uint *particles : HASHES_PARTICLE,
	uint particleCount : PARTICLE_COUNT,
	vector gridStart : GRID_START,
	uint2 cellCount : CELL_COUNT,
	scalar cellSizeInv : CELL_SIZE_INV,
	scalar particleMass : MASS,
	scalar h : SMOOTHING_LENGTH,
	scalar distEpsilon : DIST_EPSILON,
	scalar soundSpeed : WC_SOUND_SPEED,
	scalar restDensityInv : DENSITY_INV,
	__global scalar *dt : NEXT_TIME_STEP
)
{
	size_t i = get_global_id(0);
	if(i >= particleCount) return;
	vector posI = pos[i];
	vector velI = vel[i];
	vector accI = acc[i];
	scalar densityI = density[i];
	scalar sigmaMax = 0.0;
	scalar localSoundSpeed = densityI * restDensityInv;
	localSoundSpeed *= soundSpeed * localSoundSpeed * localSoundSpeed;
	
	ForEachSetup(posI,gridStart,cellSizeInv,cellCount)
	ForEachNeighbor(cellCount,hashes,particles,cellsStart)
		// Find maximum of viscosity stability parameter Monaghan JWPCOE 1999
		vector posDif = pos[j]-posI;
		scalar sigma = absolute(dot(vel[j]-velI, posDif)) / (dot(posDif, posDif) + distEpsilon);
		if(sigma > sigmaMax)
			sigmaMax = sigma;
	ForEachEnd

	// Compare with maximum acceleration stability Monaghan ARAA 1992
    scalar dtMin = soundSpeed + h * sigmaMax;//(maxlocalSoundSpeed + h * sigmaMax, dot(accI,accI));
    dtMin =  0.3 * h / dtMin;
		
	if(dtMin < *dt)
		*dt = dtMin;
}

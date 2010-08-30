/*!
 *	\brief	Compute accelerations for each particle
 */
__kernel void Accelerations
(
	__global vector *acc			: ACCELERATIONS,
	__global const vector *vel		: VELOCITIES,
	__global const vector *pos		: POSITIONS,
	__global const scalar *density	: DENSITIES,
	__global const scalar *mass		: MASSES,
	__global const scalar *pods		: PODS,
	__global const uint *cellsStart	: CELLS_START,
	__global const uint *hashes		: CELLS_HASH,
	__global const uint *particles	: HASHES_PARTICLE,
	uint particleCount 				: FLUID_PARTICLE_COUNT,
	vector gridStart 				: GRID_START,
	uint2 cellCount					: CELL_COUNT,
	scalar cellSizeInv				: CELL_SIZE_INV,
	scalar2 h						: SMOOTHING_LENGTH_INV,
	scalar distEpsilon				: DIST_EPSILON,
	scalar tcEpsilon1				: TC_EPSILON1,
	scalar tcEpsilon2				: TC_EPSILON2,
	scalar deltaPKernelInv			: DELTA_P_INV,
	scalar visc						: DYNAMIC_VISCOSITY,
	vector gravity					: GRAVITY
)
{
	size_t i = get_global_id(0);
	if(i >= particleCount) return;
	vector posI = pos[i];
	vector velI = vel[i];
	scalar densityI = density[i];
	scalar podsI = pods[i];
	scalar tensileI = podsI * (podsI<0 ? tcEpsilon1 : tcEpsilon2);
	
	vector a = (vector)0;
	
	ForEachSetup(posI,gridStart,cellSizeInv,cellCount)
	ForEachNeighbor(cellCount,hashes,particles,cellsStart)

		scalar massJ = mass[j];
		scalar podsJ = pods[j];
		vector posDif = posI-pos[j];
		vector gradW = SphKernelGrad(posDif, h.x, h.y);

	 	// tensile correction, Monaghan JCP 2000
		scalar f = SphKernel(posDif, h.x, h.y) * deltaPKernelInv;
		f *= f; f *= f; // (Wij/Wdp)^4
		scalar tensileJ = podsJ * (podsJ<0 ? tcEpsilon1 : tcEpsilon2);
		// Additional condition positive pressure tensile control only if both pressures are positive
	    scalar tensile  = podsI<0 ? tensileI : 0;
		tensile += podsJ<0 ? tensileJ : 0;
	    tensile += (podsI>0 && podsJ>0) ? tensileI + tensileJ : 0;

	    // pressure acceleration
		a -= (podsI + podsJ + f * tensile) * massJ * gradW;
		
		// laminar viscosity acceleration
		a += (4 * massJ * visc * (velI-vel[j]) * dot(gradW, posDif) / ((densityI + density[j]) * (dot(posDif,posDif) + distEpsilon)));

	ForEachEnd

	acc[i] = a + gravity;
}

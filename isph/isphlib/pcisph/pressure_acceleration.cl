/*!
 *	\brief	Calculate acceleration of corrected pressure
 */
__kernel void PressureAcceleration
(
	__global vector *acc_p			: ACCEL_PRESSURE,
	__global const scalar *pods		: PODS,
	__global const vector *pos		: POSITIONS,
	__global const scalar *mass		: MASSES,
	__global const uint *cellsStart	: CELLS_START,
	__global const uint *hashes		: CELLS_HASH,
	__global const uint *particles	: HASHES_PARTICLE,
	uint fluidParticleCount 		: FLUID_PARTICLE_COUNT,
	vector gridStart 				: GRID_START,
	uint2 cellCount					: CELL_COUNT,
	scalar cellSizeInv				: CELL_SIZE_INV,
	scalar2 h						: SMOOTHING_LENGTH_INV,
	scalar deltaPKernelInv 			: DELTA_P_INV
)
{
	size_t i = get_global_id(0);
	if(i >= fluidParticleCount) return;
	
	vector posI = pos[i];
	scalar podsI = pods[i];
	//scalar tensileI = podsI * (podsI<0 ? -0.2 : 0.01);
	
	vector a = (vector)0;
	
	ForEachSetup(posI,gridStart,cellSizeInv,cellCount)
	ForEachNeighbor(cellCount,hashes,particles,cellsStart)

		scalar podsJ = pods[j];
		vector posDif = posI-pos[j];
		vector gradW = SphKernelGrad(posDif, h.x, h.y);
/*
	 	// tensile correction
		scalar f = SphKernel(posDif, h.x, h.y) * deltaPKernelInv;
		f *= f; f *= f;
		scalar tensileJ = podsJ * (podsJ<0 ? -0.2 : 0.01);
	    scalar tensile = podsI<0 ? tensileI : 0;
		tensile += podsJ<0 ? tensileJ : 0;
	    tensile += (podsI>0 && podsJ>0)? tensileI + tensileJ : 0;
*/
	    // pressure acceleration
		a -= (podsI + podsJ /*+ f * tensile*/) * mass[j] * gradW;

	ForEachEnd

	acc_p[i] = a;
}

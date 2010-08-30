/*!
 *	\brief	Compute density acceleration with continuity equation
 */
__kernel void Continuity
(
	__global scalar *densityRoC : DENSITY_ROC,
	__global const vector *pos : POSITIONS,
	__global const scalar *mass : MASSES,
	__global const vector *xsphVel : XSPH_VELOCITIES,
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
	vector xsphVelI = xsphVel[i];
	scalar densityI = 0;
	
	ForEachSetup(posI,gridStart,cellSizeInv,cellCount)
	ForEachNeighbor(cellCount,hashes,particles,cellsStart)
		vector gradW = SphKernelGrad(posI-pos[j], h.x, h.y);
		densityI += mass[j]*dot(gradW, (xsphVelI - xsphVel[j])); // Avoids pressure fluctuation at free surfaces - Monaghan 1992 ARAA  	         
		//scalar W = SphKernel(-posI+pos[j], h.x, h.y);
		//densityI += W;
	ForEachEnd

	densityRoC[i] = densityI;
	
}

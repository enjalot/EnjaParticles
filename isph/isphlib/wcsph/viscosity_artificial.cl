/*!
 *	\brief	Compute viscous accelerations for each particle - Artificial Formulation
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
	scalar hh : SMOOTHING_LENGTH,
	scalar alpha : ALPHA_VISCOSITY,
	scalar distEpsilon : DIST_EPSILON,
	scalar soundSpeed : WC_SOUND_SPEED,
	scalar restDensityInv : DENSITY_INV
)
{
	size_t i = get_global_id(0);
	if(i >= particleCount) return;
	vector posI = pos[i];
	vector v = vel[i];
	vector a = (vector)0;
	
	scalar densityI = density[i];
	
	scalar csI = densityI * restDensityInv;
	csI *= soundSpeed * csI * csI ;
	
	ForEachSetup(posI,gridStart,cellSizeInv,cellCount)
	ForEachNeighbor(cellCount,hashes,particles,cellsStart)
	
		vector posDif = posI-pos[j];
		vector velDif  = v-vel[j];
	    scalar densityJ = density[j];
	    vector gradW = SphKernelGrad(posDif, h.x, h.y);	
	    
	    scalar csJ = densityJ * restDensityInv;
	    csJ *= soundSpeed * csJ* csJ;
		
	    scalar tmp = dot(posDif,velDif);
		tmp = tmp > 0 ? 0 : tmp;
		
	    a -= alpha*mass[j]*(csI + csJ)/(densityI+densityJ) * hh*tmp*gradW/(dot(posDif,posDif) + distEpsilon);
	   
	ForEachEnd
	acc[i] -= a ;
}

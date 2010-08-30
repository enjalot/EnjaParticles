/*!
 *	\brief	MLS pre step calculates MLS matrix coefficients and store them to 3 global vectors 
 *            only 6 elements needed since matrix is symmetric
 */
__kernel void MLSPre
(
	__global vector *mls_1 : MLS_1,
	__global vector *mls_2 : MLS_2,
	__global vector *mls_3 : MLS_3,
	__global const scalar *density : DENSITIES,
	__global const scalar *mass : MASSES,
	__global const vector *pos : POSITIONS,
	__global const uint *cellsStart : CELLS_START,
	__global const uint *hashes : CELLS_HASH,
	__global const uint *particles : HASHES_PARTICLE,
	uint particleCount : PARTICLE_COUNT,
	scalar2 h : SMOOTHING_LENGTH_INV,
	vector gridStart : GRID_START,
	uint2 cellCount : CELL_COUNT,
	scalar cellSizeInv : CELL_SIZE_INV
)
{
	size_t i = get_global_id(0);
	if(i >= particleCount) return;
	
	vector posI = pos[i];
	
	// Zero temp array
	vector  tmp_1= (vector)0;
	vector  tmp_2= (vector)0;
	vector  tmp_3= (vector)0;
	
	// Add contribution to MLS matrix terms from neighboring particles
	ForEachSetup(posI,gridStart,cellSizeInv,cellCount)
	ForEachNeighbor(cellCount,hashes,particles,cellsStart)
		vector posDif = posI - pos[j]; 
	    scalar W = SphKernel(posDif, h.x, h.y) * mass[j] / density[j];
	    tmp_1.x += W;                             //A(1,1)
	    tmp_1.y += posDif.x * W;               //A(1,2)
	    tmp_2.x += posDif.y * W;               //A(1,3)
	    tmp_2.y += posDif.x * posDif.x * W; //A(2,2)
	    tmp_3.x += posDif.x * posDif.y * W; //A(2,3)
	    tmp_3.y += posDif.y * posDif.y * W; //A(3,3)
	ForEachEnd
	
	// Self contribution
	tmp_1.x += SphKernel((vector)0, h.x, h.y) * mass[i]/ density[i];
	// Write to global
	mls_1[i] = tmp_1;
	mls_2[i] = tmp_2;
	mls_3[i] = tmp_3;
}

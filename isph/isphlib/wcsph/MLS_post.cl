/*!
 *	\brief	Inverts MLS 3x3 matrices and calculates beta MLS kernel correction vector 
 *             then corrects density.
 */
__kernel void MLSPost
(
	__global const vector *mls_1 : MLS_1,
	__global const vector *mls_2 : MLS_2,
	__global const vector *mls_3 : MLS_3,
	__global scalar *density : DENSITIES,
	__global scalar *mass : MASSES,
	__global const vector *pos : POSITIONS,
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
	
	// Load global values
	vector  tmp_1= mls_1[i];
	vector  tmp_2= mls_2[i];
	vector  tmp_3= mls_3[i];
	
	// Calculate adjoint matrix terms (it is symmetric too)
	scalar   t11 =  (tmp_3.y * tmp_2.y  - tmp_3.x * tmp_3.x); //(a33*a22 - a32*a23) 
	scalar   t12 = -(tmp_3.y * tmp_1.y  - tmp_3.x * tmp_2.x); //(a33*a12 - a32*a13)  
	scalar   t13 =  (tmp_3.x * tmp_1.y  - tmp_2.y * tmp_2.x); //(a23*a12 - a22*a13)
	
	//scalar   t22 =  (tmp_3.y * tmp_1.x  - tmp_2.x * tmp_2.x); //(a33*a11 - a31*a13)  
	//scalar   t23 = -(tmp_3.x * tmp_1.x  - tmp_1.y * tmp_2.x); //(a23*a11 - a21*a13)  
	//scalar   t33 =  (tmp_2.y * tmp_1.x  - tmp_1.y * tmp_1.y); //(a22*a11 - a21*a12)  
	
	//Calculate determinant a11*(a33*a22 - a32*a23) -a21*(a33*a12 - a32*a13) + a31*(a23*a12 - a22*a13)
	scalar detA =    tmp_1.x * t11;
    detA         +=    tmp_1.y * t12; 
	detA         +=    tmp_2.x * t13; 
	
	//Calcualte MLS parameters
	scalar beta0 = t11/detA; 
	scalar beta1 = t12/detA; 
	scalar beta2 = t13/detA; 
   
	vector posI = pos[i];
	scalar densityI = 0;
	
	// Correct density as integral of neighboring masses with corrected kernel
	ForEachSetup(posI,gridStart,cellSizeInv,cellCount)
	ForEachNeighbor(cellCount,hashes,particles,cellsStart)
		vector posDif = posI - pos[j]; 
	    scalar W = (beta0  + beta1 * posDif.x + beta2 * posDif.y ) * SphKernel(posDif, h.x, h.y);
		densityI += W*mass[j] ; 
	ForEachEnd 
    
	// Correct density adding particle self contribution
	density[i] = (densityI + beta0*SphKernel((vector)0, h.x, h.y) *mass[i]) ;
}

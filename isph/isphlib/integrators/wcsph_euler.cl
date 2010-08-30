/*!
 *	\brief	Advance particle positions
 */
__kernel void IntegrateEuler
(
	__global vector *pos : POSITIONS,
	__global scalar *density : DENSITIES,
	__global vector *vel : VELOCITIES,
	__global const vector *xsphVel : XSPH_VELOCITIES,
	__global const vector *acc : ACCELERATIONS,
	__global const scalar *densityRoC : DENSITY_ROC,
	uint fpc : FLUID_PARTICLE_COUNT,
	scalar dt : TIME_STEP
)
{
	size_t i = get_global_id(0);
	density[i] += densityRoC[i] * dt;
	if(i >= fpc) return; // skip the boundary particles
	
	pos[i] += xsphVel[i] * dt;
	vel[i] += acc[i] * dt;
}
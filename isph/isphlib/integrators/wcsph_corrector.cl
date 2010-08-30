/*!
 *	\brief	Advance particle positions
 */
__kernel void IntegrateCorrector
(
	__global vector *pos : POSITIONS,
	__global const vector *pos_tmp : POSITIONS_TMP,
	__global vector *vel : VELOCITIES,
	__global scalar  *density: DENSITIES,
	__global const scalar  *density_tmp: DENSITY_TMP,
	__global const vector *vel_tmp : VELOCITIES_TMP,
	__global const vector *xsphVel : XSPH_VELOCITIES,
	__global const vector *acc : ACCELERATIONS,
	__global const scalar *densityRoC : DENSITY_ROC,
	uint fpc : FLUID_PARTICLE_COUNT,
	uint pc : PARTICLE_COUNT,
	scalar dt : TIME_STEP
)
{
	size_t i = get_global_id(0);

	density[i] = density_tmp[i] + densityRoC[i] * dt;

	if(i < fpc)
	{
		pos[i] = pos_tmp[i] + xsphVel[i] * dt;
		vel[i] = vel_tmp[i] + acc[i] * dt;
	}
	else if(i < pc)
	{
		pos[i] = pos_tmp[i] + vel[i] * dt;
	}
}

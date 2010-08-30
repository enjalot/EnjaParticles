/*!
 *	\brief	Advance particles PCISPH, simple Euler
 */
__kernel void Advance
(
	__global vector *vel			: VELOCITIES,
	__global vector *pos			: POSITIONS,
	__global const vector *vel_tmp	: VELOCITIES_TMP,
	__global const vector *pos_tmp	: POSITIONS_TMP,
	__global const vector *acc_rest	: ACCEL_REST,
	__global const vector *acc_p	: ACCEL_PRESSURE,
	uint fluidParticleCount 		: FLUID_PARTICLE_COUNT,
	scalar dt 						: TIME_STEP
)
{
	size_t i = get_global_id(0);
	if(i >= fluidParticleCount) return;

	vector v = vel_tmp[i] + (acc_p[i] + acc_rest[i]) * dt;
	vel[i] = v;
	pos[i] = pos_tmp[i] + v * dt;
}

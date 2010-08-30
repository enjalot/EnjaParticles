/*!
 *	\brief	Moves all the particle by rigid motion around center cor with angular velocity angularSpeed
 */
__kernel void PrescribedMotion
(
	__global vector *pos : POSITIONS,
	__global vector *vel : VELOCITIES,
	scalar dt : TIME_STEP,
	scalar4 angularSpeed : ANGULAR_SPEED,
    scalar4 cor: CENTER_OF_ROTATION
)
{
	size_t i = get_global_id(0);
	
	//vel[i]  = cross(angularSpeed, pos[i]-cor) ;
   	scalar4 pos4 = (scalar4)(pos[i].x, pos[i].y, 0, 0);
	scalar4 vel4= cross(angularSpeed, pos4 - cor) ;	
	//vel[i] = vel4.xy;
	vel[i].x = 1.0;
	vel[i].y = 0.0;
	pos[i] = pos[i] + vel[i] * dt;
}

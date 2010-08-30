/*!
 *	\brief	Compute new density and pressure
 */
__kernel void EoS
(
	__global const scalar *density : DENSITIES,
	__global scalar *pods : PODS,
	__global scalar *pressure : PRESSURES,
	uint particleCount : PARTICLE_COUNT,
	scalar wcConst : WC_CONST,
	scalar gamma : WC_GAMMA,
	scalar restDensityInv : DENSITY_INV

)
{
	size_t i = get_global_id(0);
	if(i >= particleCount) return;
	scalar densityI = density[i];
	scalar p = wcConst * (pow(densityI*restDensityInv, gamma) - 1);
	pods[i] = p / (densityI * densityI);
	pressure[i] = p;
}

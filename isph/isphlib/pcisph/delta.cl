/*!
 *	\brief	Calculate PCISPH delta constant
 */
__kernel void GetDeltaPCISPH
(
	__global scalar *output	: FOUND_DELTA,
	scalar l				: PARTICLE_SPACING,
	scalar h 				: SMOOTHING_LENGTH,
	scalar2 hInv			: SMOOTHING_LENGTH_INV
)
{
	scalar sum = (scalar)0;
	int n = (int)ceil(h / l);

	// TODO 3D version
	for(int x=-n; x<=n; x++)
	for(int y=-n; y<=n; y++)
	if(x!=0 && y!=0)
	{
		vector gradW = SphKernelGrad((vector)(x*l,y*l), hInv.x, hInv.y);
		sum += dot(gradW,gradW);
	}

	*output = sum;
}

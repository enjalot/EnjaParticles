__kernel void ComputeDeltaP
(
	__global scalar *output	: DELTA_P,
	scalar2 h 				: SMOOTHING_LENGTH_INV,
	scalar   l 				: PARTICLE_SPACING
)
{
	*output = SphKernelDeltaP(l, h.x, h.y);
}

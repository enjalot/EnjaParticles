#ifndef M_1_PI 
#define M_1_PI 0.31830988618379067154
#endif

/*!
 *	\brief	Wendland's quintic (5th order with good perfomance) smoothing kernel.
 */
scalar SphKernel(vector distVec, scalar smoothInv, scalar smoothInvSq)
{
	scalar c = (7 * M_1_PI / 4) * smoothInvSq;
	scalar dist = length(distVec);
	scalar Q = dist * smoothInv;
	if(Q < 2)
	{
		scalar dif = 1 - 0.5*Q;
		return c * pow(dif, 4) * (2*Q + 1);
	} 
	return 0;
}

/*!
 *	\brief	Wendland's quintic (5th order with good perfomance) smoothing kernel gradient.
 */
vector SphKernelGrad(vector distVec, scalar smoothInv, scalar smoothInvSq)
{
	scalar c = (-35 * M_1_PI / 4) * smoothInvSq * smoothInvSq;
	scalar dist = length(distVec);
	scalar Q = dist * smoothInv;
	if(Q < 2)
	{
		scalar dif = 1 - 0.5*Q;
		return distVec * (c * dif * dif * dif);
	}
	return (vector)0;
}

/*!
 *	\brief	Tensile correction specific distance function.
 */
scalar SphKernelDeltaP(scalar particleSpacing, scalar smoothInv, scalar smoothInvSq) 
{
	vector a = (vector)0;
	a.x = particleSpacing;
	return SphKernel( a, smoothInv, smoothInvSq);
}



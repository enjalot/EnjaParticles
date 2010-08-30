#ifndef M_1_PI 
#define M_1_PI 0.31830988618379067154
#endif
#ifndef CUBIC_KERNEL_SUPPORT
#define CUBIC_KERNEL_SUPPORT 2
#endif


/*!
 *	\brief	Cubic (3rd order) spline smoothing kernel.
 */
scalar SphKernel(vector distVec, scalar smoothInv, scalar smoothInvSq)
{
	scalar dist = length(distVec);
	scalar Q = dist * smoothInv;
	if(Q < 1)
	{
		scalar c = 10 * M_1_PI / 7 * smoothInvSq;
		return c * (1 - 1.5*Q*Q + 0.75*Q*Q*Q);
	}
	else if(Q < 2)
	{
		scalar c = 10 * M_1_PI / 28 * smoothInvSq;
		scalar dif = 2 - Q;
		return c * dif * dif * dif;
	}
	return 0;
}

/*!
 *	\brief	Cubic (3rd order) spline smoothing kernel gradient.
 */
vector SphKernelGrad(vector distVec, scalar smoothInv, scalar smoothInvSq)
{
	scalar dist = length(distVec);
	scalar Q = dist * smoothInv;
	if(Q < 1)
	{
		scalar c = 90 * M_1_PI / 28 * smoothInvSq * smoothInvSq;
		return distVec * (c * (Q - 4.0/3.0));
	}
	else if(Q < 2)
	{
		scalar c = -30 * M_1_PI / 28 * smoothInvSq * smoothInv;
		scalar dif = 2 - Q;
		return distVec * (c * dif * dif / dist);
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

/*!
 *	\brief	Kernel compact support .
 */
scalar SphKernelSupport(scalar smoothInv) 
{
	return  CUBIC_KERNEL_SUPPORT / smoothInv;
}

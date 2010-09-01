#ifndef M_1_PI 
#define M_1_PI 0.31830988618379067154
#endif
#ifndef GAUSS_KERNEL_SUPPORT
#define GAUSS_KERNEL_SUPPORT 3
#endif
/*!
 *	\brief	Modified Gauss' (exponential based) smoothing kernel.
 */
scalar SphKernel(vector distVec, scalar smoothInv, scalar smoothInvSq)
{
	scalar dist = length(distVec);
	scalar Q = dist * smoothInv;
	scalar d = exp(-9.0); 
	scalar c = M_1_PI * smoothInvSq;
	if(Q < GAUSS_KERNEL_SUPPORT)
		return c * (exp(-(Q*Q)) - d) / (1.0 - 10.0 * d);
	return 0;
}

/*!
 *	\brief	Modified Gauss' (exponential based) smoothing kernel gradient.
 */
vector SphKernelGrad(vector distVec, scalar smoothInv, scalar smoothInvSq)
{
	scalar c = -2 * M_1_PI * smoothInvSq * smoothInvSq;
	scalar d = exp(-9.0); 
	scalar dist = length(distVec);
	scalar Q = dist * smoothInv;
	if(Q < GAUSS_KERNEL_SUPPORT)
		return distVec * (c * exp(-(Q*Q)) / (1 - 10.0f*d));
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
	return  GAUSS_KERNEL_SUPPORT / smoothInv;
}	

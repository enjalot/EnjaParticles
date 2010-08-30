#ifndef ISPH_DIFFUSIONSIMULATION_H
#define ISPH_DIFFUSIONSIMULATION_H

#include "simulation.h"

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

namespace isph {


	/*!
	 *	\class	DiffusionSimulation 
	 *	\brief	Pure scalar diffusion simulation with specified external motion (rotation around point). Used to test spatial indexing.
	 *	\tparam	dim	Number of space dimensions: 2 or 3.
	 *	\tparam typ Type representing real numbers (affects accuracy and performance): float or double.
	 */
	template<int dim, typename typ>
	class DiffusionSimulation : public Simulation<dim, typ>
	{
	public:

		DiffusionSimulation();
		~DiffusionSimulation();

		/*!
		 *	\brief	Set diffusion parameters.
		 *	\param	diffusvity  Diffusivity to be used in diffusion equation [m2/s].
		 */
		void SetDiffusivity(typ diffusivity);

		/*!
		 *	\brief	Get Diffisivity.
		 */
		inline typ Diffusivity() { return diffusivity; }

		/*!
		 *	\brief	Set rotational motion parameters.
		 *	\param	centerOfRotation [m].
         *	\param	angular speed    [rad/s].
		 */
		void SetRotationParameters(const Vec<4,typ>& centerOfRotation, const Vec<4,typ>& angularSpeed);

	protected:

		virtual bool InitSph();
		virtual bool RunSph();

		cl_float4   centerOfRotation;
		cl_float4   angularSpeed;
		typ diffusivity;
	};
}

#endif

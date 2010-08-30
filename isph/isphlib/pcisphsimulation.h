#ifndef ISPH_PCISPHSIMULATION_H
#define ISPH_PCISPHSIMULATION_H

#include "simulation.h"

namespace isph
{

	/*!
	 *	\class	PcisphSimulation
	 *	\brief	Predictive-corrective incompressible SPH simulation class.
	 *	\tparam	dim	Number of space dimensions: 2 or 3.
	 *	\tparam typ Type representing real numbers (affects accuracy and performance): float or double.
	 */
	template<int dim, typename typ>
	class PcisphSimulation : public Simulation<dim,typ>
	{
	public:

		PcisphSimulation();
		~PcisphSimulation();

		/*!
		 *	\brief	Set speed of sound inside fluid, in meters per second [m/s].
		 */
		void SetSpeedOfSound(typ speed);

		/*!
		 *	\brief	Get speed of sound inside fluid, in meters per second [m/s].
		 */
		inline typ SpeedOfSound() { return speedOfSound; }

		/*!
		 *	\brief	Initialize the PCISPH simulation with all the parameters set.
		 *	\return	Success.
		 */
		virtual bool Initialize();

	protected:

		using Simulation<dim, typ>::program;

		virtual bool InitSph();
		virtual bool RunSph();

		typ speedOfSound;

	};

}

#endif

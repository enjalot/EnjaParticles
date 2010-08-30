#ifndef ISPH_WCSPHSIMULATION_H
#define ISPH_WCSPHSIMULATION_H

#include "simulation.h"

namespace isph {

     /*!
	 *	\enum DensityReinitMethod
	 *	\brief	Different density reinitialization methods.
	 */
	enum DensityReinitMethods
	{
		None,						//!< No density reinitialization.
		ShepardFilter,				//!< Shepard Filter.
		MovingLeastSquares			//!< Moving Least Square Dilts IJNME 99.
	};

	/*!
	 *	\class	WcsphSimulation
	 *	\brief	Weakly compressible SPH simulation class.
	 *	\tparam	dim	Number of space dimensions: 2 or 3.
	 *	\tparam typ Type representing real numbers (affects accuracy and performance): float or double.
	 */
	template<int dim, typename typ>
	class WcsphSimulation : public Simulation<dim,typ>
	{
	public:
		using Simulation<dim, typ>::program;
		
		WcsphSimulation();
		~WcsphSimulation();

		/*!
		 *	\brief	Set WCSPH parameters.
		 *	\param	speedOfSound	Speed of sound inside fluid, in meters per second [m/s].
		 *	\param	gamma			Gamma parameter, default value is 7.
		 */
		void SetWcsphParameters(typ speedOfSound, typ gamma);

		/*!
		 *	\brief	Converts pressure in density using equation on state and WCSPH params.
		 *	\param	pressure Pressure value to be converted [N/m2].
		 */
        typ  GetDensityFromPressure(typ pressure);

		/*!
		 *	\brief	Get WCSPH gamma parameter.
		 */
		inline typ WcsphGamma() { return wcGamma; }

		/*!
		 *	\brief	Get WCSPH speed-of-sound parameter, in meters per second [m/s].
		 */
		inline typ WcsphSpeedOfSound() { return wcSpeedOfSound; }

		/*!
		 *	\brief	Set XSPH velocity correction parameters.
		 *	\param	factor	The factor of XSPH, default is 0.5
		 */
		inline void SetXsphFactor(typ factor) { xsphFactor = factor; }

		/*!
		 *	\brief	Get the XSPH factor.
		 */
		inline typ XsphFactor() { return xsphFactor; }

		/*!
		 *	\brief	Set density reinitialization method.
		 *	\param	method Selected method for density reinitialization.
		 */
		inline void SetDensityReinitMethod(DensityReinitMethods method) { densityReinitMethod = method; }

		/*!
		 *	\brief	Get selected method for density reinitialization.
		 */
		inline DensityReinitMethods DensityReinitMethod() { return densityReinitMethod ; }

		/*!
		 *	\brief	Set frequency of density reinitialization.
		 *	\param	reinitFreq	    Number of timesteps between reinitializations use -1 to prevent reinitialization.
		 */
		inline void SetDensityReinitFrequency(int reinitFreq) { densityReinitFrequency = reinitFreq; }

		/*!
		 *	\brief	Get frequency of density reinitialization in terms of timesteps.
		 */
		inline int DensityReinitFrequency() { return densityReinitFrequency; }

		/*!
		 *	\brief	Suggest next time step based on Courant Friedrichs Levy type condition, Monaghan & Kos (1999).
		 *	\return	Time in seconds.
		 */
		virtual typ SuggestTimeStep();

	protected:

		virtual bool UploadParticleData();

		virtual bool InitSph();
		virtual bool RunSph();

		typ wcGamma;
		typ wcSpeedOfSound;
		typ xsphFactor;

        // Density reinitialization
		DensityReinitMethods densityReinitMethod;
		int densityReinitFrequency;
	};


}

#include "wcsphsimulation.cpp"

#endif

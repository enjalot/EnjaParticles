#ifndef ISPH_UTILS_H
#define ISPH_UTILS_H

#include <cmath>
#include <string>
#include "vec.h"

namespace isph
{

	/*!
	 *	\class	Consts
	 *	\brief	Static class holding some useful constants.
	 */
	class Consts
	{
	public:
		Consts(){}
		~Consts(){}

		static const double Pi; /*!< Ludolph's number PI */
		static const double PiInv; /*!< Reciprocal Ludolph's number PI */
		static const double e; /*!< Euler's number e */
		static const double eInv; /*!< Reciprocal Euler's number e */
		static const double g; /*!< Standard gravity (acceleration due to gravity) [m/s^2] */
		
		class Water
		{
		public:
			static const double StdDensity; /*!< Standard density of water [kg/m^3], at 20°C */
			static double Density(int temperature); /*!< Density of water [kg/m^3] based on temperature [°C] */
			static const double StdViscosity; /*!< Standard dynamic viscosity of water [Pa*s=kg/ms] */
			static double Viscosity(int temperature); /*!< Viscosity of water [kg/m^3] based on temperature [°C] */
		};

		class Sea
		{
		public:
			static const double StdDensity; /*!< Standard density of sea water [kg/m^3] */
			static double Density(int temperature); /*!< Density of sea water [kg/m^3] based on temperature [°C] */
			static const double StdViscosity; /*!< Standard dynamic viscosity of sea water [Pa*s=kg/ms] */
			static double Viscosity(int temperature); /*!< Viscosity of sea water [kg/m^3] based on temperature [°C] */
		};
	};

	class Utils
	{
	public:
		Utils(){}
		~Utils(){}

		static bool IsPowerOf2(unsigned int num);
		static unsigned int NearestPowerOf2(unsigned int num);
		static unsigned int NearestMultiple(unsigned int num, unsigned int divisor, bool snapUp = true);
		static unsigned int FactorRadix2(unsigned int num);
		static std::string LoadCLSource(const std::string& filename);
		static Vec<2,unsigned int>* CreateRandomSequence(unsigned int sequenceLength);

	};


} // namespace isph

#endif

#ifndef ISPH_PARTICLE_H
#define ISPH_PARTICLE_H

#include "vec.h"
#include "clvariable.h"
#include <cstring>
#include <string>

namespace isph {

	template<int dim, typename typ>	class Simulation;

	
	/*!
	 *	\enum	ParticleType
	 *	\brief	Enumeration of every type of particle.
	 */
	enum ParticleType
	{
		FluidParticle,
		BoundaryParticle,

		ParticleTypeCount
	};


	/*!
	 *	\class	Particle
	 *	\brief	Base particle class.
	 *	\tparam	dim	Number of space dimensions: 2 or 3.
	 *	\tparam typ Type representing real numbers (affects accuracy and performance): float or double.
	 */
	template<int dim, typename typ>
	class Particle
	{
	public:

		Particle(Simulation<dim,typ>* simulation, unsigned int index);
		~Particle();

		/*!
		 *	\brief	Get the classification/type of the particle.
		 */
		ParticleType Type();

		/*!
		 *	\brief	Get the particle mass [kg].
		 */
		//const typ& Mass() { return sim->particleMass[Type()]; }
		inline const typ& Mass() { return *(typ*)sim->massesBuffer->Get(id); }

		/*!
		 *	\brief	Get the particle density [kg/m^3].
		 */
		inline const typ& Density() { return *(typ*)sim->densitiesBuffer->Get(id); }

		/*!
		 *	\brief	Get the particle pressure [Pa=N/m^2].
		 */
		inline const typ& Pressure() { return *(typ*)sim->pressuresBuffer->Get(id); }

		/*!
		 *	\brief	Get the particle current position vector.
		 */
		inline const Vec<dim,typ>& Position() { return *(Vec<dim,typ>*)sim->positionsBuffer->Get(id); }

		/*!
		 *	\brief	Get the particle current velocity vector.
		 */
		inline const Vec<dim,typ>& Velocity() { return *(Vec<dim,typ>*)sim->velocitiesBuffer->Get(id); }

		/*!
		 *	\brief	Set the particle mass [kg].
		 */
		void SetMass(const typ& mass) { sim->massesBuffer->Set(id, &mass); }

		/*!
		 *	\brief	Set the particle density [kg/m^3].
		 */
		void SetDensity(const typ& density) { sim->densitiesBuffer->Set(id, &density); }

		/*!
		 *	\brief	Set the particle pressure [Pa=N/m^2].
		 */
		void SetPressure(const typ& pressure) { sim->pressuresBuffer->Set(id, &pressure); }

		/*!
		 *	\brief	Set the particle position.
		 */
		void SetPosition(const Vec<dim,typ>& position) { sim->positionsBuffer->Set(id, &position); }

		/*!
		 *	\brief	Set the particle velocity.
		 */
		void SetVelocity(const Vec<dim,typ>& velocity) { sim->velocitiesBuffer->Set(id, &velocity); }

	protected:

		unsigned int id;
		Simulation<dim,typ> *sim;

	};


	/*!
	 *	\class	ParticleAttributeBuffer
	 *	\brief	Class handling device and host buffers for particle attributes.
	 */
	class ParticleAttributeBuffer
	{
	public:

		ParticleAttributeBuffer(CLVariable* buffer);
		~ParticleAttributeBuffer();

		inline const std::string& Name() { return deviceData->Semantic(); }

		inline CLVariable* DeviceData() { return deviceData; }

		inline char* HostData() { return hostData; }

		inline VariableDataType DataType() { return deviceData->DataType(); }

		inline size_t DataTypeSize() { return deviceData->DataTypeSize(); }

		inline bool HostHasData() { return hostHasData; }

		inline bool HostDataChanged() { return hostDataChanged; }

		inline void* Get(unsigned int id) { return (void*)(hostData + DataTypeSize()*id); }

		inline void Get(unsigned int id, void* memoryPos) { std::memcpy(memoryPos, Get(id), DataTypeSize()); }

		inline void Set(unsigned int id, const void* memoryPos) { std::memcpy(Get(id), memoryPos, DataTypeSize()); hostDataChanged = true; }

		bool Download();

		bool Upload();

	private:

		CLVariable* deviceData;
		char* hostData;
		bool hostHasData;
		bool hostDataChanged;
	};

} // namespace isph

#endif

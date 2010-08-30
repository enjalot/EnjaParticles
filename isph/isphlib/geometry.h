#ifndef ISPH_GEOMETRY_H
#define ISPH_GEOMETRY_H

#include <string>
#include "particle.h"

namespace isph 
{

	/*!
	 *	\class	Geometry
	 *	\brief	2D/3D model finally converted to particles.
	 */
	template<int dim, typename typ>
	class Geometry
	{		
	public:
		Geometry(Simulation<dim,typ>* parentSimulation, ParticleType particleType);
		Geometry(Simulation<dim,typ>* parentSimulation, ParticleType particleType, std::string name);
		virtual ~Geometry();

		/*!
		 *	\brief	Get the number of particles that represent the geometry.
		 */
		unsigned int ParticleCount();

		/*!
		 *	\brief	Get the ID of the first particle.
		 */
		inline unsigned int ParticleStartId() { return startId; }

		/*!
		 *	\brief	Get the type of particles that represent the geometry.
		 */
		inline ParticleType Type() { return type; }

		/*!
		 *	\brief	Set object's velocity.
		 */
		void SetVelocity(const Vec<dim,typ>& velocity);

	protected:

		template<int _dim, typename _typ> friend class Simulation;

		/*!
		 *	\brief	Get the number of particle needed to represent the geometry.
		 */
		virtual unsigned int CountNeededParticles() = 0;

		/*!
		 *	\brief	Build geometry with particles.
		 */
		virtual void Build() = 0;

		/*!
		 *	\brief	Init particle with position, fluid density, and zero velocity and pressure.
		 */
		void InitParticle(unsigned int id, Vec<dim,typ> pos);

		unsigned int startId;
		unsigned int particleCount;
		Simulation<dim,typ>* sim;
		ParticleType type;

	};


	/*!
	 *	\namespace	geo
	 *	\brief		Library of geometric shapes.
	 */
	namespace geo
	{
		
		/*!
		 *	\class	Point
		 *	\brief	Point i.e. a particle. Lowest level shape.
		 */
		template<int dim, typename typ>
		class Point : public Geometry<dim,typ>
		{
		public:
			using Geometry<dim, typ>::sim;
			using Geometry<dim, typ>::particleCount;
			Point(Simulation<dim,typ>* parentSimulation, ParticleType particleType) : Geometry<dim,typ>::Geometry(parentSimulation,particleType) {}
			Point(Simulation<dim,typ>* parentSimulation, ParticleType particleType, std::string name) : Geometry<dim,typ>::Geometry(parentSimulation,particleType, name) {}
			virtual ~Point() {}
			inline void Define(const Vec<dim,typ>& pos) { position = pos; }
		protected:
			virtual unsigned int CountNeededParticles();
			virtual void Build();
			Vec<dim,typ> position;
		};

		/*!
		 *	\class	Line
		 *	\brief	Line of particles between two points.
		 */
		template<int dim, typename typ>
		class Line : public Geometry<dim,typ>
		{
		public:
			using Geometry<dim, typ>::sim;
			using Geometry<dim, typ>::particleCount;
			Line(Simulation<dim,typ>* parentSimulation, ParticleType particleType) : Geometry<dim,typ>::Geometry(parentSimulation,particleType), width(1) {}
			Line(Simulation<dim,typ>* parentSimulation, ParticleType particleType, std::string name) : Geometry<dim,typ>::Geometry(parentSimulation,particleType, name), width(1) {}
			virtual ~Line() {}
			inline void Define(Vec<dim,typ> start, Vec<dim,typ> end) { startPoint=start; endPoint=end; width = 1; invertLayerOrientation = false;}
			inline void Define(Vec<dim,typ> start, Vec<dim,typ> end, unsigned int rowsOfParticles) { startPoint=start; endPoint=end; width = rowsOfParticles; invertLayerOrientation = false;}
			inline void Define(Vec<dim,typ> start, Vec<dim,typ> end, unsigned int rowsOfParticles, typ rowsSpacing) { startPoint=start; endPoint=end; width = rowsOfParticles; layerSpacing = rowsSpacing; invertLayerOrientation = false;}
			inline void SetWidth(unsigned int rowsOfParticles) { width = rowsOfParticles; }
			inline void SetLayerSpacing(typ rowsSpacing) { layerSpacing = rowsSpacing; }
			inline void InvertLayerOrientation(bool orientation) { invertLayerOrientation = orientation; }
		protected:
			virtual unsigned int CountNeededParticles();
			virtual void Build();
			Vec<dim,typ> startPoint, endPoint;
			unsigned int width;
			typ layerSpacing;
			bool invertLayerOrientation;
		};

		/*!
		 *	\class	Box
		 *	\brief	Box (3D) i.e. rectangle (2D).
		 */
		template<int dim, typename typ>
		class Box : public Geometry<dim,typ>
		{
		public:
			using Geometry<dim, typ>::sim;
			using Geometry<dim, typ>::particleCount;
			Box(Simulation<dim,typ>* parentSimulation, ParticleType particleType) : Geometry<dim,typ>::Geometry(parentSimulation,particleType), filled(false) {}
			Box(Simulation<dim,typ>* parentSimulation, ParticleType particleType, std::string name) : Geometry<dim,typ>::Geometry(parentSimulation,particleType, name), filled(false) {}
			virtual ~Box() {}
			inline void Define(const Vec<dim,typ>& start, const Vec<dim,typ>& end) { startPoint=start; endPoint=end; }
			inline void Fill() { filled = true; }
		protected:
			virtual unsigned int CountNeededParticles();
			virtual void Build();
			Vec<3,typ> startPoint, endPoint;
			Vec<3,unsigned int> particleCounts;
			Vec<3,typ> realSpacings;
			bool filled;
		};

		/*!
		 *	\class	Sphere
		 *	\brief	Sphere (3D) i.e. circle (2D).
		 */
		template<int dim, typename typ>
		class Sphere : public Geometry<dim,typ>
		{
		public:
			using Geometry<dim, typ>::sim;
			using Geometry<dim, typ>::particleCount;
			Sphere(Simulation<dim,typ>* parentSimulation, ParticleType particleType) : Geometry<dim,typ>::Geometry(parentSimulation,particleType), filled(false) {}
			Sphere(Simulation<dim,typ>* parentSimulation, ParticleType particleType, std::string name) : Geometry<dim,typ>::Geometry(parentSimulation,particleType, name), filled(false) {}
			virtual ~Sphere() {}
			inline void Define(const Vec<dim,typ>& centerPoint, typ radiusDist) { center=centerPoint; radius=radiusDist; }
			inline void Fill() { filled = true; }
		protected:
			virtual unsigned int CountNeededParticles();
			virtual void Build();
			Vec<3,typ> center;
			typ radius, realSpacing;
			unsigned int slices;
			bool filled;
		};

	}
}

#endif

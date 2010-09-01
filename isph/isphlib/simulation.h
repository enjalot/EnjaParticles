#ifndef ISPH_SIMULATION_H
#define ISPH_SIMULATION_H

#include <string>
#include <map>
#include "particle.h"
#include "geometry.h"
#include "gpubitonicsort.h"
#include "probemanager.h"
#include "abstractintegrator.h"


namespace isph {

	class CLLink;

	/*!
	 *	\enum SmoothingKernelType
	 *	\brief	Different smoothing kernel types.
	 */
	enum SmoothingKernelType
	{
		CubicSplineKernel,	//!< Cubic (3rd order) spline smoothing kernel.
		GaussKernel,		//!< Gauss' (exponential based) smoothing kernel.
		ModifiedGaussKernel,//!< Modified Gauss' (exponential based) smoothing kernel takes into account compact support.
		WendlandKernel		//!< Wendland's quintic (5th order with good perfomance) smoothing kernel.
	};

	/*!
	 *	\enum ViscosityFormulationType
	 *	\brief	Different viscosity formulations types.
	 */
	enum ViscosityFormulationType
	{
		ArtificialViscosity,		//!< Artificial Viscosity.
		LaminarViscosity,			//!< Laminar Viscosity.
		SubParticleScaleViscosity	//!< Sub-particle-scale (SPS) viscosity.
	};

	/*!
	 *	\enum IntegratorType
	 *	\brief	Different time advancing schemes.
	 */
	enum IntegratorType
	{
		PredictorCorrector,			//!< Modified Euler or Predictor Corrector.
		RungeKutta4					//!< Runge-Kutta fourth order.
	};

	/*!
	 *	\class	Simulation
	 *	\brief	Abstract class for different SPH kinds of simulation.
	 *	\tparam	dim	Number of space dimensions: 2 or 3.
	 *	\tparam typ Type representing real numbers (affects accuracy and performance): float or double.
	 */
	template<int dim, typename typ>
	class Simulation
	{
	public:

		Simulation();
		virtual ~Simulation();

		/*!
		 *	\brief	Set the initial spacing between particles [m].
		 */
		void SetParticleSpacing(typ spacing);

		/*!
		 *	\brief	Get the initial spacing between particles [m].
		 */
		inline typ ParticleSpacing() { return particleSpacing; }

		/*!
		 *	\brief	Get the number of particles of specific type.
		 */
		inline unsigned int ParticleCount(ParticleType type) { return particleCountByType[type]; }

		/*!
		 *	\brief	Get the number of particles.
		 */
		inline unsigned int ParticleCount() { return particleCount; }

		/*!
		 *	\brief	Set the default mass for particle type.
		 */
		void SetParticleMass(ParticleType type, typ mass);

		/*!
		 *	\brief	Retirns the default mass for particle type.
		 */
		typ ParticleMass(ParticleType type);

		/*!
		 *	\brief	Get the particle from type and "local" id.
		 */
		Particle<dim,typ> GetParticle(ParticleType type, unsigned int id);

		/*!
		 *	\brief	Get the particle from "global" id only.
		 */
		Particle<dim,typ> GetParticle(unsigned int id);

		/*!
		 *	\brief	Set the smoothing kernel type and its length.
		 */
		void SetSmoothingKernel(SmoothingKernelType type, typ length);

		/*!
		 *	\brief	Get the smoothing kernel type.
		 */
		inline SmoothingKernelType SmoothingKernel() { return smoothingKernel; }

		/*!
		 *	\brief	Get the smoothing length.
		 */
		inline typ SmoothingLength() { return smoothingLength; }

		/*!
		 *	\brief	Set the formulation for viscosity.
		 */
		void SetViscosityFormulationType(ViscosityFormulationType type);

		/*!
		 *	\brief	Get the viscosity formulation type.
		 */
		inline ViscosityFormulationType  ViscosityFormulation() { return viscosityFormulation; }

		/*!
		 *	\brief	Set the fluid density.
		 */
		void SetDensity(typ density);

		/*!
		 *	\brief	Get the fluid density.
		 */
		inline typ Density() { return density; }

		/*!
		 *	\brief	Set the external global acceleration in [m/s^2].
		 */
		void SetGravity(const Vec<dim,typ>& acceleration);

		/*!
		 *	\brief	Get the external global acceleration in [m/s^2].
		 */
		inline Vec<dim,typ> Gravity() { return gravity; }

		/*!
		 *	\brief	Set the fluid (dynamic) viscosity.
		 *	\param	viscosity	Value of the viscosity, in pascal-seconds [Pa*s = kg/(s*m)]).
		 */
		void SetDynamicViscosity(typ viscosity);

		/*!
		 *	\brief	Get the fluid dynamic viscosity, in pascal-seconds [Pa*s = kg/(s*m)].
		 */
		inline typ DynamicViscosity() { return dynamicViscosity; }

		/*!
		 *	\brief	Get the fluid kinematic viscosity, in meters squared per seconds [m^2/s].
		 */
		inline typ KinematicViscosity() { return kinematicViscosity; }

		/*!
		 *	\brief	Set the fluid alpha parameter for artificial viscosity.
		 *	\param	alpha	Value of the alpha parameter for artificial viscosity, in  ???.
		 */
		void SetAlphaViscosity(typ alpha);

		/*!
		 *	\brief	Get the fluid alpha parameter for artificial viscosity, in  ???..
		 */
		inline typ AlphaViscosity() { return alphaViscosity; }

		/*!
		 *	\brief	Sets the simulation container boundaries.
		 */
		void SetBoundaries(const Vec<dim,typ>& boundsMin, const Vec<dim,typ>& boundsMax);

		/*!
		 *	\brief	Get the simulation container minimum point.
		 */
		inline Vec<dim,typ> BoundaryMin() { return gridMin; }

		/*!
		 *	\brief	Get the simulation container maximum point.
		 */
		inline Vec<dim,typ> BoundaryMax() { return gridMax; }

		/*!
		 *	\brief	Get the bit count of a any number in simulation.
		 *	\return	32 for single, or 64 for double precision floating point number.
		 */
		inline unsigned int ScalarPrecision() { return sizeof(typ)*8; }

		/*!
		 *	\brief	Get the number of dimensions.
		 *	\return	Dimension count: 2 or 3
		 */
		inline unsigned int Dimensions() { return dim; }

		/*!
		 *	\brief	Set link to devices that will run the simulation.
		 */
		void SetDevices(CLLink* linkToDevices);

		/*!
		 *	\brief	Get link to the probe manager.
		 */
		virtual inline ProbeManager<dim,typ>* GetProbeManager(){ return probeManager; };

		/*!
		 *	\brief	Initialize the simulation with all the parameters set.
		 *	\return	Success.
		 */
		virtual bool Initialize();

		/*!
		 *	\brief	Advance simulation by specified time step, in seconds.
		 *	\param	advanceTimeStep	Time step to advance simulation by, in seconds.
		 *	\return	Success.
		 */
		virtual bool Advance(float advanceTimeStep);

		/*!
		 *	\brief	Suggest next time step based on Courant Friedrichs Levy type condition.
		 *	\return	Time in seconds.
		 */
		virtual typ SuggestTimeStep();

		/*!
		 *	\brief	Current time of simulaton.
		 *	\return	Time in seconds.
		 */
		inline double Time() { return timeOverall; }

		/*!
		 *	\brief	Get the data from devices back to host.
		 */
		bool DownloadParticleData(const std::string& attribute);

		/*!
		 *	\brief	Send the data from host to devices.
		 */
		virtual bool UploadParticleData();

		/*!
		 *	\brief	Get the amount of used memory in bytes simulation has allocated on devices.
		 */
		size_t UsedMemorySize();

		/*!
		 *	\brief	Get geometry model by name.
		 *	\return	If found, pointer to the geometry object, else a NULL pointer.
		 */
		Geometry<dim,typ>* GetGeometry(const std::string& name);

		/*!
		 *	\brief	Get the particle attribute buffer by its name.
		 */
		ParticleAttributeBuffer* ParticleAttribute(const std::string& name);

		inline ParticleAttributeBuffer* ParticleMasses()     { return massesBuffer; }
		inline ParticleAttributeBuffer* ParticlePositions()  { return positionsBuffer; }
		inline ParticleAttributeBuffer* ParticleDensities()  { return densitiesBuffer; }
		inline ParticleAttributeBuffer* ParticlePressures()  { return pressuresBuffer; }
		inline ParticleAttributeBuffer* ParticleVelocities() { return velocitiesBuffer; }

		VariableDataType ScalarDataType();
		VariableDataType Scalar2DataType();
		VariableDataType Scalar4DataType();
		VariableDataType VectorDataType(bool realNumbers = true);

		/*!
		 *	\brief	Load a subprogram from .CL file, and set a name for it.
		 */
		bool LoadSubprogram(const std::string& name, const std::string& filename);

		/*!
		 *	\brief	Execute subprogram.
		 */
		bool EnqueueSubprogram(const std::string& name, size_t globalSize=0, size_t localSize=0);        

		/*!
		 *	\brief	Copies buffer contents via EnqueueCopyBuffer.
		 */
	    bool WriteVariableFrom(const std::string& semanticTo, const std::string& semanticFrom);

		/*!
		 *	\brief	Insert all particles to grid.
		 */
		virtual bool RunGrid();


	protected: 
	//public:

		/*!
		 *	\brief	Set the number of particles.
		 */
		void SetParticleCount(ParticleType type, unsigned int count);


		/*!
		 *	\brief	Init particles attribute needed for simulation.
		 */
		bool InitParticleAttribute(const std::string& semantic, VariableDataType dataType);

		/*!
		 *	\brief	Init simulation attribute.
		 */
		bool InitSimulationConstant(const std::string& semantic, VariableDataType dataType, void* data = NULL);

		/*!
		 *	\brief	Init simulation attribute.
		 */
		bool InitSimulationBuffer(const std::string& semantic, VariableDataType dataType, unsigned int elementCount);

		/*!
		 *	\brief	Init general simulation subprograms, variables and build options.
		 */
		virtual bool InitGeneral();

		/*!
		 *	\brief	Init simulation grid subprograms, variables and build options.
		 */
		virtual bool InitGrid();

		/*!
		 *	\brief	Init radix sort subprograms and variables.
		 */
		bool InitRadixSort();

		/*!
		 *	\brief	Init subprograms, variables and build options, specific to the inherited SPH method.
		 */
		virtual bool InitSph() = 0;


		/*!
		 *	\brief	Insert all particles to grid.
		 */
		bool RunRadixSort();

		/*!
		 *	\brief	Run actual SPH simulation. To be inherited by different SPH method classes.
		 */
		virtual bool RunSph() = 0;


		// variable members:

		template<int dim_, typename typ_>	friend class Particle;
		template<int dim__, typename typ__>	friend class Geometry;
        template<int dim___, typename typ___>	friend class ProbeManager;

		// solver
		CLProgram *program;

		// subprograms
		std::map<std::string,CLSubProgram*> subprograms;

		// scene data
		std::multimap<std::string,Geometry<dim,typ>*> models;

		// probe manager
		ProbeManager<dim,typ> *probeManager;

        // integrator
        AbstractIntegrator<dim,typ> *integrator;
		IntegratorType integratorType;

		// particle data
		typ particleSpacing;
		unsigned int particleCount;
		unsigned int particleCountByType[ParticleTypeCount];
		unsigned int allocatedParticleCount;
		typ particleMass[ParticleTypeCount];
		ParticleAttributeBuffer *massesBuffer, *positionsBuffer, *velocitiesBuffer, *pressuresBuffer, *densitiesBuffer;
		
		std::map<std::string, ParticleAttributeBuffer*> particleAttributes;


		// kernel
		SmoothingKernelType smoothingKernel;
		typ smoothingLength;
	
		// density
		typ density;

		// viscosity
		ViscosityFormulationType viscosityFormulation;

		typ dynamicViscosity;
		typ kinematicViscosity;
		typ alphaViscosity;

		// misc
		Vec<dim,typ> gravity;

		// scene grid
		typ gridCellSize;
		Vec<dim,typ> gridMin;
		Vec<dim,typ> gridMax;
		Vec<dim,typ> gridSize;
		Vec<dim,unsigned int> gridCellCount;
		GpuBitonicSort* sorter;
		unsigned int scanCta;
		unsigned int radixSortCta;

		// time
		typ timeOverall;
		typ timeStep;
		unsigned int timeStepCount;

	};
}

#endif

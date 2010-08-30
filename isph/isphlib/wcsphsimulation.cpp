

#include "isph.h"

using namespace isph;


template<int dim, typename typ>
WcsphSimulation<dim, typ>::WcsphSimulation()
	: Simulation<dim, typ>()
	, wcSpeedOfSound(0)
	, wcGamma(7)
	, xsphFactor(0.5)
	, densityReinitMethod(None)
	, densityReinitFrequency(-1)
{
    //Simulation<dim,typ>::integratorType = PredictorCorrector;
    integratorType = PredictorCorrector;
}


template<int dim, typename typ>
WcsphSimulation<dim, typ>::~WcsphSimulation()
{
}


template<int dim, typename typ>
void WcsphSimulation<dim, typ>::SetWcsphParameters( typ speedOfSound, typ gamma )
{
	wcSpeedOfSound = speedOfSound;
	wcGamma = gamma;
}


template<int dim, typename typ>
typ WcsphSimulation<dim, typ>::GetDensityFromPressure( typ pressure )
{
	typ wcConst = this->density * wcSpeedOfSound * wcSpeedOfSound / wcGamma;
	typ value = pow((( pressure /  wcConst ) + 1), 1/wcGamma) * this->density;
	return value;
}


template<int dim, typename typ>
bool WcsphSimulation<dim, typ>::InitSph()
{
	LogDebug("Initializing WCSPH stuff");
	
	if(!wcSpeedOfSound)
	{
		Log::Send(Log::Error, "You need to set WCSPH numerical speed of sound");
		return false;
	}

	if(!wcGamma)
	{
		Log::Send(Log::Error, "You need to set WCSPH gamma parameter");
		return false;
	}

	// WCSPH specific variables
	typ wcConst = Simulation<dim,typ>::density * wcSpeedOfSound * wcSpeedOfSound / wcGamma;
	typ tcEpsilon1 = (typ)-0.2; // Tensile correction factor for negative pressure
	typ tcEpsilon2 = (typ)0.01; // Tensile correction factor for positive pressure

	this->InitSimulationConstant("WC_SOUND_SPEED", this->ScalarDataType(), &wcSpeedOfSound);
	this->InitSimulationConstant("WC_CONST", this->ScalarDataType(), &wcConst);
	this->InitSimulationConstant("WC_GAMMA", this->ScalarDataType(), &wcGamma);
	this->InitSimulationConstant("XSPH_FACTOR", this->ScalarDataType(), &xsphFactor);
	this->InitSimulationConstant("TC_EPSILON1", this->ScalarDataType(), &tcEpsilon1);
	this->InitSimulationConstant("TC_EPSILON2", this->ScalarDataType(), &tcEpsilon2);

	this->InitParticleAttribute("XSPH_VELOCITIES", this->VectorDataType()); // xsph corrected velocities
	this->InitParticleAttribute("PODS", this->ScalarDataType()); // aux variable - pressure over density squared
    this->InitParticleAttribute("ACCELERATIONS", this->VectorDataType()); // aux variable accelerations
	this->InitParticleAttribute("DENSITY_ROC", this->ScalarDataType()); // aux variable
    this->InitParticleAttribute("VELOCITIES_TMP", this->VectorDataType()); // aux variable old velocities
    this->InitParticleAttribute("POSITIONS_TMP", this->VectorDataType()); // aux variable old positions
	this->InitParticleAttribute("DENSITY_TMP", this->ScalarDataType()); // aux variable old densities


   	// program build options
	program->AddBuildOption("-D WCSPH");

	switch(this->viscosityFormulation)
	{
	case ArtificialViscosity: 
         this->LoadSubprogram("acceleration", "wcsph/acceleration_artificial.cl");
		 break;
	case LaminarViscosity: 
         this->LoadSubprogram("acceleration", "wcsph/acceleration_laminar.cl");
		 break;
	default: Log::Send(Log::Error, "Viscosity formulation choice is not correct.");
	}

	switch(densityReinitMethod)
	{
  	case None:
         break;
	case ShepardFilter: 
         this->LoadSubprogram("densityReinitPre", "wcsph/shepard_filter_pre.cl");
         this->LoadSubprogram("densityReinitPost", "wcsph/shepard_filter_post.cl");
		 break;
	case MovingLeastSquares: 
		 program->ConnectSemantic("MLS_1", program->Variable("VELOCITIES_TMP"));
         program->ConnectSemantic("MLS_2", program->Variable("POSITIONS_TMP"));
         program->ConnectSemantic("MLS_3", program->Variable("ACCELERATIONS"));
         this->LoadSubprogram("densityReinitPre", "wcsph/MLS_pre.cl");
         this->LoadSubprogram("densityReinitPost", "wcsph/MLS_post.cl");
		 break;
	default: Log::Send(Log::Error, "Density reinitialization method choice is not correct.");
	}

	this->LoadSubprogram("xsph", "wcsph/xsph.cl");
    this->LoadSubprogram("continuity", "wcsph/continuity.cl");
	this->LoadSubprogram("cfl", "wcsph/cfl.cl");
	this->LoadSubprogram("eos", "wcsph/tait_eos.cl");


	switch(this->integratorType)
	{
	case PredictorCorrector: 
		 this->LoadSubprogram("predictor", "integrators/wcsph_predictor.cl");
    	 this->LoadSubprogram("corrector", "integrators/wcsph_corrector.cl");
         //want to instantiate the integrator
         integrator = new PCIntegrator<dim, typ>(this);
         printf("instantiated PCIntegrator\n");
         break;
	default: Log::Send(Log::Error, "Integrator type choice is not correct.");
	}

	return true;
}


template<int dim, typename typ>
bool WcsphSimulation<dim, typ>::RunSph()
{
    printf("Run SPH\n");
	//Simulation<dim,typ>::integrator->Integrate();
	integrator->Integrate();
    printf("Done with integrator->Integrate()\n");

	// Shepard filter every densityReinitFrequency steps
	if (((this->timeStepCount + 1) % densityReinitFrequency ) == 0) 
	{
		this->EnqueueSubprogram("densityReinitPre");
		this->EnqueueSubprogram("densityReinitPost");
		this->EnqueueSubprogram("eos");
	}

	return true;
}


template<int dim, typename typ>
typ WcsphSimulation<dim, typ>::SuggestTimeStep()
{
	// init it to extra high value
	typ nextStep = 1000;
	program->Variable("NEXT_TIME_STEP")->WriteFrom(&nextStep);
	
	// find minimum required next time step
	this->EnqueueSubprogram("cfl");
	program->Finish();
	
	// read the result
	program->Variable("NEXT_TIME_STEP")->ReadTo(&nextStep);
	return nextStep;
}
 

template<int dim, typename typ>
bool WcsphSimulation<dim, typ>::UploadParticleData()
{
	if(this->ParticleDensities()->HostDataChanged() || this->ParticlePressures()->HostDataChanged())
	{
		ParticleAttributeBuffer* pods = this->ParticleAttribute("PODS");

		for (unsigned int i=0; i < this->particleCount; i++)
		{
			typ density = *(typ*)this->ParticleDensities()->Get(i);
			typ pod = *(typ*)this->ParticlePressures()->Get(i) / (density * density);
			pods->Set(i, &pod);
		}
	}

	if(this->ParticleVelocities()->HostDataChanged())
	{
		ParticleAttributeBuffer* xvel = this->ParticleAttribute("XSPH_VELOCITIES");

		for (unsigned int i=0; i < this->particleCount; i++)
			xvel->Set(i, this->ParticleVelocities()->Get(i));
	}

	return Simulation<dim, typ>::UploadParticleData();
}

// explicit specializations
template class WcsphSimulation<2,float>;



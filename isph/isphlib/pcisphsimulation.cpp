#include "isph.h"
using namespace isph;


template<int dim, typename typ>
PcisphSimulation<dim, typ>::PcisphSimulation()
	: Simulation<dim, typ>()
{
}


template<int dim, typename typ>
PcisphSimulation<dim, typ>::~PcisphSimulation()
{
}


template<int dim, typename typ>
void PcisphSimulation<dim, typ>::SetSpeedOfSound( typ speed )
{
	speedOfSound = speed;
}


template<int dim, typename typ>
bool PcisphSimulation<dim, typ>::InitSph()
{
	LogDebug("Initializing PCISPH stuff");

	if(!speedOfSound)
	{
		Log::Send(Log::Error, "You need to specify speed of sound inside fluid");
		return false;
	}

	// PCSIPH specific particle attributes
	this->InitParticleAttribute("ACCEL_REST", this->VectorDataType());
	this->InitParticleAttribute("ACCEL_PRESSURE", this->VectorDataType());
	this->InitParticleAttribute("VELOCITIES_TMP", this->VectorDataType());
	this->InitParticleAttribute("POSITIONS_TMP", this->VectorDataType());
	this->InitParticleAttribute("PODS", this->ScalarDataType());

	// PCISPH simulation variables
	this->InitSimulationConstant("SOUND_SPEED", this->ScalarDataType(), &speedOfSound);
	this->InitSimulationBuffer("FOUND_DELTA", this->ScalarDataType(), 1);
	this->InitSimulationConstant("DELTA", this->ScalarDataType());

	// program build options
	program->AddBuildOption("-D PCISPH");

	// subprograms
	this->LoadSubprogram("find delta", "pcisph/delta.cl");
	this->LoadSubprogram("prepare", "pcisph/prepare.cl");
	this->LoadSubprogram("advance", "pcisph/advance.cl");
	this->LoadSubprogram("correct", "pcisph/correct.cl");
	this->LoadSubprogram("pressure acceleration", "pcisph/pressure_acceleration.cl");

	return true;
}


template<int dim, typename typ>
bool PcisphSimulation<dim, typ>::Initialize()
{
	bool success = Simulation<dim, typ>::Initialize();

	if(success)
	{
		// precompute PCISPH delta constant
		typ delta;
		this->EnqueueSubprogram("find delta", 1, 1);
		program->Finish();
		success &= program->Variable("FOUND_DELTA")->ReadTo(&delta); // TODO use copy from clvariable to another
		delta = this->density * this->density / (2 * delta);
		success &= program->Variable("DELTA")->WriteFrom(&delta);
	}

	return success;
}


template<int dim, typename typ>
bool PcisphSimulation<dim, typ>::RunSph()
{
	// refresh grid
	this->RunGrid();

	// calculate viscosity and gravity accelerations, reset some vars
	this->EnqueueSubprogram("prepare");

	// TODO now it's fixed to 3 steps, it should loop until error < eta
	for(unsigned int i=0; i<3; i++)
	{
		// advance particles with current pressure
		this->EnqueueSubprogram("advance", this->ParticleCount(FluidParticle));

		// refresh grid
		//this->RunGrid(); // TODO test if needed, in paper they dont do it

		// calculate density and correct the pressure
		this->EnqueueSubprogram("correct");

		// calculate acceleration of corrected pressure
		this->EnqueueSubprogram("pressure acceleration", this->ParticleCount(FluidParticle));
	}

	// advance particles with corrected pressure
	this->EnqueueSubprogram("advance", this->ParticleCount(FluidParticle));

	return true;
}


// explicit specializations
template class PcisphSimulation<2,float>;

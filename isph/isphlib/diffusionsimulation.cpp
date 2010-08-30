#include "diffusionsimulation.h"
#include "isph.h"
#include <cstdlib>
#include <cstring>
using namespace isph;

template<int dim, typename typ>
DiffusionSimulation<dim, typ>::DiffusionSimulation() : 
	Simulation<dim, typ>(),
	diffusivity(1)
{
}


template<int dim, typename typ>
DiffusionSimulation<dim, typ>::~DiffusionSimulation()
{
}


template<int dim, typename typ>
void DiffusionSimulation<dim, typ>::SetDiffusivity(typ diffusivity)
{
	this->diffusivity = diffusivity;
}


template<int dim, typename typ>
void DiffusionSimulation<dim, typ>::SetRotationParameters(const Vec<4,typ>& centerOfRotation, const Vec<4,typ>& angularSpeed)
{
	std::memcpy(&this->centerOfRotation, &centerOfRotation, sizeof(cl_float4));
	std::memcpy(&this->angularSpeed, &angularSpeed, sizeof(cl_float4));
}


template<int dim, typename typ>
bool DiffusionSimulation<dim, typ>::InitSph()
{
	LogDebug("Initializing Diffusion Simulation stuff");

	if(!diffusivity)
	{
		Log::Send(Log::Error, "You need to set diffusivity value");
		return false;
	}

	// Diffusion specific variables

	this->InitSimulationConstant("DIFFUSIVITY", this->ScalarDataType(), &diffusivity);
	this->InitSimulationConstant("ANGULAR_SPEED", this->Scalar4DataType(), &angularSpeed);
    this->InitSimulationConstant("CENTER_OF_ROTATION", this->Scalar4DataType(), &centerOfRotation);

	this->InitParticleAttribute("PASSIVE_SCALAR", this->ScalarDataType()); // Passive scalar
	
	// subprograms
	this->LoadSubprogram("scalarDiffusion", "wcsph/scalar_diffusion.cl");
    this->LoadSubprogram("prescribedMotion", "wcsph/prescribed_motion.cl");

	return true;
}


template<int dim, typename typ>
bool DiffusionSimulation<dim, typ>::RunSph()
{
	// solve diffusion eequation
	Simulation<dim, typ>::EnqueueSubprogram("scalarDiffusion");

	// move particles 
	//EnqueueSubprogram("prescribedMotion");

	// reinit grid with new particle positions
	Simulation<dim, typ>::RunGrid(); 

	return true;
}


// explicit specializations
template class DiffusionSimulation<2,float>;

#include "particle.h"
#include "simulation.h"
#include <cstring>
using namespace std;
using namespace isph;

template<int dim, typename typ>
Particle<dim,typ>::Particle(Simulation<dim,typ>* simulation, unsigned int index)
	: sim(simulation)
	, id(index)
{
}

template<int dim, typename typ>
Particle<dim,typ>::~Particle()
{
}


template<int dim, typename typ>
ParticleType Particle<dim, typ>::Type()
{
	if(id < sim->particleCountByType[FluidParticle])
		return FluidParticle;
	return BoundaryParticle;
}


//////////////////////////////////////////////////////////////////////////


ParticleAttributeBuffer::ParticleAttributeBuffer( CLVariable* buffer )
	: deviceData(buffer)
	, hostData(NULL)
	, hostHasData(true)
	, hostDataChanged(false)
{
	hostData = new char[buffer->MemorySize()];
}


ParticleAttributeBuffer::~ParticleAttributeBuffer()
{
	delete [] hostData;
}

bool ParticleAttributeBuffer::Download()
{
	if(!deviceData)
		return false;
	if(!hostData)
		return false;
	if(hostHasData)
		return true;

	if(deviceData->ReadTo(hostData))
	{
		hostHasData = true;
		hostDataChanged = false;
		return true;
	}

	return false;
}


bool ParticleAttributeBuffer::Upload()
{
	if(!deviceData)
		return false;
	if(!hostData)
		return false;
	if(!hostDataChanged)
	{
		hostHasData = false;
		return true;
	}

	if(deviceData->WriteFrom(hostData))
	{
		hostDataChanged = false;
		hostHasData = false;
		return true;
	}

	return false;
}


// explicit specializations
template class Particle<2,float>;

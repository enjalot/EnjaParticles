#include "geometry.h"
#include "isph.h"

using namespace isph;


template<int dim, typename typ>
Geometry<dim, typ>::Geometry( Simulation<dim,typ>* parentSimulation, ParticleType particleType )
	: sim(parentSimulation)
	, type(particleType)
	, particleCount(0)
	, startId(0)
{
	sim->models.insert(std::pair<std::string,Geometry<dim,typ>*>("nameless",this));
}


template<int dim, typename typ>
Geometry<dim, typ>::Geometry( Simulation<dim,typ>* parentSimulation, ParticleType particleType, std::string name )
	: sim(parentSimulation)
	, type(particleType)
	, particleCount(0)
	, startId(0)
{
	sim->models.insert(std::pair<std::string,Geometry<dim,typ>*>(name,this));
}


template<int dim, typename typ>
Geometry<dim, typ>::~Geometry()
{

}


template<int dim, typename typ>
unsigned int Geometry<dim, typ>::ParticleCount()
{
	if(!particleCount)
		particleCount = CountNeededParticles();
	return particleCount;
}


template<int dim, typename typ>
void Geometry<dim, typ>::InitParticle( unsigned int id, Vec<dim,typ> pos )
{
	Particle<dim,typ> p = sim->GetParticle(type, startId + id);
	p.SetPosition(pos);
	p.SetVelocity(Vec<dim,typ>());
	p.SetDensity(sim->Density());
	p.SetMass(sim->ParticleMass(type));
	p.SetPressure(0);
}


template<int dim, typename typ>
void Geometry<dim, typ>::SetVelocity( const Vec<dim,typ>& velocity )
{
	if(particleCount)
	{
		sim->program->Variable("VECTOR_VALUE")->WriteFrom(&velocity);
		sim->program->Variable("OBJECT_START")->WriteFrom(&startId);
		sim->program->Variable("OBJECT_PARTICLE_COUNT")->WriteFrom(&particleCount);
		sim->EnqueueSubprogram("upload attribute", Utils::NearestMultiple(particleCount, 512));
	}
}


// Point


template<int dim, typename typ>
unsigned int geo::Point<dim, typ>::CountNeededParticles()
{
	return 1;
}

template<int dim, typename typ>
void geo::Point<dim, typ>::Build()
{
	InitParticle(0, position);
}


// Line


template<int dim, typename typ>
unsigned int geo::Line<dim, typ>::CountNeededParticles()
{
	Vec<dim,typ> diff = endPoint - startPoint;
	unsigned int count = 0;
	unsigned int lineCnt = 1 + (unsigned int)(diff.length() / sim->ParticleSpacing());
	for (unsigned int i=0; i<width ;i++) 
		count += lineCnt + (i % 2);
	return count;
}

template<int dim, typename typ>
void geo::Line<dim, typ>::Build()
{
	Vec<dim,typ> diff = endPoint - startPoint;
	unsigned int lineCnt = 1 + (unsigned int)(diff.length() / sim->ParticleSpacing());
	typ spacing = sim->ParticleSpacing();//diff.length() / (typ)(lineCnt - 1);
	typ spacing2 = 0;

	if (!layerSpacing) 
		spacing2 = spacing;
	else
	    spacing2 = layerSpacing;

	Vec<dim,typ> unitDiff = diff / diff.length();
	Vec<dim,typ> unitNorm = unitDiff.normalDir();
	if (invertLayerOrientation) unitNorm = -1.0 * unitNorm;
	
	unsigned int cnt = 0;
	for (unsigned int j=0; j<width; j++) {
		for (unsigned int i=0; i<(lineCnt + (j % 2)); i++)  {
          Vec<dim,typ> displacement  = unitDiff * (i - (typ)0.5*(j%2))*spacing - unitNorm * (j * spacing2);
		  InitParticle(cnt++, startPoint + displacement );
		}
	}
}


// Box


template<int dim, typename typ>
unsigned int geo::Box<dim, typ>::CountNeededParticles()
{
	Vec<3,typ> dif = endPoint - startPoint;
	unsigned int count;
	particleCounts.z = 1;
	if(filled)
	{
		count = 1;
		for(unsigned int i=0; i<dim; i++)
		{
			particleCounts[i] = std::max((int)(std::abs(dif[i]) / sim->ParticleSpacing()), 1);
			realSpacings[i] = sim->ParticleSpacing();//std::abs(dif[i]) / particleCounts[i];
			count *= particleCounts[i];
		}
	}
	else
	{
		for(unsigned int i=0; i<dim; i++)
		{
			particleCounts[i] = std::max((int)(std::abs(dif[i]) / sim->ParticleSpacing()), 1);
			realSpacings[i] = std::abs(dif[i]) / (particleCounts[i] - 1);
		}
		if(dim == 2)
			count = 2 * (particleCounts.x + particleCounts.y) - 4;
		else
			count = particleCounts.x*particleCounts.y*particleCounts.z - (particleCounts.x-1)*(particleCounts.y-1)*(particleCounts.z-1);
	}
	return count;
}

template<int dim, typename typ>
void geo::Box<dim, typ>::Build()
{
	if(filled)
	{
		startPoint += realSpacings / 2;
		for(unsigned int xi=0; xi<particleCounts.x; xi++)
			for(unsigned int yi=0; yi<particleCounts.y; yi++)
				for(unsigned int zi=0; zi<particleCounts.z; zi++)
				{
                    unsigned int idx = xi*particleCounts.y*particleCounts.z + particleCounts.z*yi + zi;
                   	InitParticle(idx, Vec<dim,typ>(startPoint.x + realSpacings.x * xi, startPoint.y + realSpacings.y * yi, startPoint.z + realSpacings.z * zi));
				}
		startPoint -= realSpacings / 2;
	}
	else
	{
		Vec<dim,typ> pos;
		if(dim == 2)
		{
			unsigned int i = 0;
			for(unsigned int xi=0; xi<particleCounts.x; xi++)
			{
				pos.x = startPoint.x + realSpacings.x * xi;
				pos.y = startPoint.y;
				InitParticle(i++, pos);
				pos.y = endPoint.y;
				InitParticle(i++, pos);
			}
			for(unsigned int yi=1; yi<particleCounts.y-1; yi++)
			{
				pos.y = startPoint.y + realSpacings.y * yi;
				pos.x = startPoint.x;
				InitParticle(i++, pos);
				pos.x = endPoint.x;
				InitParticle(i++, pos);
			}
		}
		else
		{
			// TODO
		}
	}
}


// Sphere


template<int dim, typename typ>
unsigned int geo::Sphere<dim, typ>::CountNeededParticles()
{
	unsigned int count = 0;

	if(filled)
	{
		typ radiusSq = radius * radius;
		slices = (unsigned int)(2 * radius / sim->ParticleSpacing());
		realSpacing = 2 * radius / slices;
		Vec<3,typ> localStartPoint = -Vec<3,typ>(radius-realSpacing/2, radius-realSpacing/2, dim==2 ? 0 : radius-realSpacing/2);
		unsigned int zslices = dim==2 ? 1 : slices;
		for(unsigned int k=0; k<zslices; k++)
			for(unsigned int i=0; i<slices; i++)
				for(unsigned int j=0; j<slices; j++) 
				{
					Vec<3,typ> dist = localStartPoint + Vec<3,typ>(realSpacing*i, realSpacing*j, realSpacing*k);
					if(dist.lengthSq() <= radiusSq)
						count++;
				}
	}
	else
	{
		if(dim == 2)
		{
			count = (unsigned int)(2 * radius * (typ)M_PI / sim->ParticleSpacing());
			realSpacing = 2 * (typ)M_PI / count;
		}
		else
		{
			// TODO
		}
	}

	return count;
}

template<int dim, typename typ>
void geo::Sphere<dim, typ>::Build()
{
	unsigned int p = 0;
	if(filled)
	{
		typ radiusSq = radius * radius;
		Vec<3,typ> localStartPoint = -Vec<3,typ>(radius-realSpacing/2, radius-realSpacing/2, dim==2 ? 0 : radius-realSpacing/2);
		unsigned int zslices = dim==2 ? 1 : slices;
		for(unsigned int k=0; k<zslices; k++)
			for(unsigned int i=0; i<slices; i++)
				for(unsigned int j=0; j<slices; j++) 
				{
					Vec<3,typ> dist = localStartPoint + Vec<3,typ>(realSpacing*i, realSpacing*j, realSpacing*k);
					if(dist.lengthSq() <= radiusSq)
					{
						Vec<3,typ> pos = center + dist;
						InitParticle(p, Vec<dim,typ>(&pos.x));
						p++;
					}
				}
	}
	else
	{
		if(dim == 2)
		{
			for (unsigned int i=0; i<particleCount; i++)
			{
				typ angle = i * realSpacing;
				InitParticle(i, Vec<dim,typ>(cos(angle)*radius, sin(angle)*radius));
			}
		}
		else
		{
			// TODO
		}	
	}
}


// explicit specializations
template class Geometry<2,float>;
template class geo::Point<2,float>;
template class geo::Line<2,float>;
template class geo::Box<2,float>;
template class geo::Sphere<2,float>;

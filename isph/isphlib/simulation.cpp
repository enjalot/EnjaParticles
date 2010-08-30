#include "simulation.h"
#include "isph.h"

using namespace isph;
using namespace std;

template<int dim, typename typ>
Simulation<dim,typ>::Simulation()
	: particleCount(0)
	, smoothingKernel(CubicSplineKernel)
	, smoothingLength(0)
	, density(0)
	, dynamicViscosity(0)
	, kinematicViscosity(0)
	, gridCellSize(0)
	, sorter(NULL)
	, timeOverall(0)
	, timeStep(0)
	, timeStepCount(0)
{
	LogDebug("Creating new simulation object");

	program = new CLProgram();

	for (unsigned int i=0; i<(unsigned int)ParticleTypeCount; i++)
	{
		particleMass[i] = 0;
        particleCountByType[i] = 0;
	}   
 
    probeManager = new ProbeManager<dim,typ>(this);

	particleCount  = 0;
}


template<int dim, typename typ>
Simulation<dim,typ>::~Simulation()
{
	LogDebug("Destroying simulation object");
	delete program;

	LogDebug("Destroying host data");
	for (std::map<std::string, ParticleAttributeBuffer*>::iterator i = particleAttributes.begin(); i != particleAttributes.end(); i++)
		delete i->second;
}


template<int dim, typename typ>
void Simulation<dim,typ>::SetSmoothingKernel(SmoothingKernelType type, typ length)
{
	smoothingKernel = type;
	smoothingLength = length;
}

template<int dim, typename typ>
void Simulation<dim,typ>::SetViscosityFormulationType(ViscosityFormulationType type)
{
	viscosityFormulation = type;
}


template<int dim, typename typ>
void Simulation<dim,typ>::SetDensity( typ density )
{
	if(density <= 0.0)
	{
		Log::Send(Log::Error, "Fluid density must be greater than zero");
		return;
	}

	this->density = density;
	kinematicViscosity = dynamicViscosity / density;
}

template<int dim, typename typ>
void Simulation<dim,typ>::SetGravity(const Vec<dim,typ>& acceleration)
{
	gravity = acceleration;
}

template<int dim, typename typ>
void Simulation<dim,typ>::SetDynamicViscosity( typ viscosity )
{
	dynamicViscosity = viscosity;
	if(density != 0.0)
		kinematicViscosity = dynamicViscosity / density;
}

template<int dim, typename typ>
void Simulation<dim,typ>::SetAlphaViscosity( typ alpha)
{
	alphaViscosity = alpha;
}


template<int dim, typename typ>
void Simulation<dim,typ>::SetBoundaries(const Vec<dim,typ>& boundsMin, const Vec<dim,typ>& boundsMax)
{
	gridMin = boundsMin;
	gridMax = boundsMax;
	gridSize = gridMax - gridMin;
}


template<int dim, typename typ>
bool Simulation<dim,typ>::LoadSubprogram(const std::string& name, const std::string& filename)
{
	CLSubProgram *sp;
	std::map<std::string,CLSubProgram*>::iterator found = subprograms.find(name);

	if(found == subprograms.end())
	{
		sp = new CLSubProgram();
		subprograms[name] = sp;
	}
	else
		sp = found->second;

	bool success = sp->Load(filename);
	program->AddSubprogram(sp);
	return success;
}


template<int dim, typename typ>
bool isph::Simulation<dim, typ>::EnqueueSubprogram(const std::string& name, size_t globalSize, size_t localSize)
{
	std::map<std::string,CLSubProgram*>::iterator found = subprograms.find(name);

	if(found != subprograms.end())
	{
		CLSubProgram *sp = found->second;

		if(!sp->Enqueue(localSize>1 ? Utils::NearestMultiple(globalSize, 256) : globalSize, localSize))
		{
			Log::Send(Log::Error, "Error while executing subprogram: " + sp->KernelName());
			return false;
		}
		return true;

		/*if(sp->Enqueue(globalSize, localSize)) // for debugging
		{
			if(!program->Finish())
				Log::Send(Log::Error, "Error while waiting for subprogram: " + sp->KernelName());

			return true;
		}
		else
		{
			Log::Send(Log::Error, "Error while executing subprogram: " + sp->KernelName());
			return false;
		}*/

	}
	
	Log::Send(Log::Error, "Subprogram you want to enqueue doesn't exist");
	return false;
}

template<int dim, typename typ>
bool Simulation<dim, typ>::WriteVariableFrom(const std::string& semanticTo, const std::string& semanticFrom)
{
   return program->Variable(semanticTo)->WriteFrom(program->Variable(semanticFrom));
}


template<int dim, typename typ>
bool Simulation<dim, typ>::InitParticleAttribute(const std::string& semantic, VariableDataType dataType)
{
	CLVariable *var = program->Variable(semantic);
	if(!var)
	{
		var = new CLVariable(program, semantic, GlobalBuffer);
		var->SetSpace(dataType, Utils::NearestMultiple(particleCount, 512)); // TODO allocatedParticleCount

		ParticleAttributeBuffer* buf = new ParticleAttributeBuffer(var);
		particleAttributes.insert(std::make_pair(semantic, buf));
	}
	// TODO else, simulation reiniting ?

	return true;
}


template<int dim, typename typ>
bool Simulation<dim, typ>::InitSimulationConstant(const std::string& semantic, VariableDataType dataType, void* data)
{
	CLVariable *var = program->Variable(semantic);
	if(!var)
		var = new CLVariable(program, semantic, KernelArgument); // TODO differ between constants and kernel args
	var->SetSpace(dataType, 1);
	if(data)
		return var->WriteFrom(data);
	else
		return true;
}


template<int dim, typename typ>
bool Simulation<dim, typ>::InitSimulationBuffer(const std::string& semantic, VariableDataType dataType, unsigned int elementCount)
{
	CLVariable *var = program->Variable(semantic);
	if(!var)
		var = new CLVariable(program, semantic, GlobalBuffer);
	var->SetSpace(dataType, elementCount);
	return true;
}


template<int dim, typename typ>
void isph::Simulation<dim, typ>::SetDevices( CLLink* linkToDevices )
{
	program->SetLink(linkToDevices);
}


template<int dim, typename typ>
bool Simulation<dim,typ>::Initialize()
{
	LogDebug("Initializing the simulation");

	// release
	program->ClearBuildOptions();
	program->ClearSubprograms();

	// init probes
	if(!probeManager->Prepare())
	{
		Log::Send(Log::Error, "Inizializing probe manager failed.");
		return false;
	}

	// create

	if(!InitGeneral())
	{
		Log::Send(Log::Error, "Setting up general simulation stuff failed");
		return false;
	}

	if(!InitGrid())
	{
		Log::Send(Log::Error, "Initializing simulation grid failed");
		return false;
	}

	// general particle variables
	InitParticleAttribute("MASSES", ScalarDataType()); massesBuffer = ParticleAttribute("MASSES");
	InitParticleAttribute("DENSITIES", ScalarDataType()); densitiesBuffer = ParticleAttribute("DENSITIES");
	InitParticleAttribute("PRESSURES", ScalarDataType()); pressuresBuffer = ParticleAttribute("PRESSURES");
	InitParticleAttribute("POSITIONS", VectorDataType()); positionsBuffer = ParticleAttribute("POSITIONS");
	InitParticleAttribute("VELOCITIES", VectorDataType()); velocitiesBuffer = ParticleAttribute("VELOCITIES");

	// build geometry
	typename std::multimap<std::string,Geometry<dim,typ>*>::iterator it;
	for(it=models.begin(); it != models.end(); it++)
		it->second->Build();

	if(!InitSph())
	{
		Log::Send(Log::Error, "Initializing SPH stuff failed");
		return false;
	}

	// init probes
    probeManager->InitKernels();

	// build

	if(!program->Build())
	{
		Log::Send(Log::Error, "Building the simulation program failed");
		return false;
	}

	// precompute tensile correction kernel dP constant
	typ dp;
	EnqueueSubprogram("deltaP", 1, 1);
    //enjalot
    printf("ij: about to call program finish from Initialize()\n");
	program->Finish();
	program->Variable("DELTA_P")->ReadTo(&dp); // TODO copy directly with clvariable
	dp = 1 / dp;
	program->Variable("DELTA_P_INV")->WriteFrom(&dp);

	return true;
}


template<int dim, typename typ>
bool Simulation<dim, typ>::InitGrid()
{
	LogDebug("Initializing simulation grid");

	// to be sure extend boundaries by particle spacing
	SetBoundaries(gridMin - Vec<dim,typ>(particleSpacing), gridMax + Vec<dim,typ>(particleSpacing));


	// setup grid constants

    // Set kernel support radius
	switch(smoothingKernel)
	{
	case CubicSplineKernel:
    	gridCellSize = 2 * smoothingLength;
		break;
	case GaussKernel:
		gridCellSize = 3 * smoothingLength;
    	break;
	case ModifiedGaussKernel:
    	gridCellSize = 3 * smoothingLength;
		break;
	case WendlandKernel:
    	gridCellSize = 2 * smoothingLength;
		break;
	}

	gridCellCount.x = static_cast<unsigned int>(ceil(gridSize.x / gridCellSize));
	gridCellCount.y = static_cast<unsigned int>(ceil(gridSize.y / gridCellSize));
	unsigned int cells = gridCellCount.x * gridCellCount.y;
	if(dim == 3)
	{
		gridCellCount[2] = static_cast<unsigned int>(ceil(gridSize[2] / gridCellSize));
		cells *= gridCellCount[2];
	}

	// test if all needed parameters are set
	if(!gridCellCount.x || !gridCellCount.y)
	{
		Log::Send(Log::Error, "You need to correctly set simulation boundaries and smoothing length");
		return false;
	}

	// variables
	typ gridCellSizeInv = 1 / gridCellSize;
	InitSimulationBuffer("CELLS_START", UintType, Utils::NearestMultiple(cells, 512));
	InitSimulationConstant("GRID_START", VectorDataType(), &gridMin);
	InitSimulationConstant("CELL_SIZE", ScalarDataType(), &gridCellSize);
	InitSimulationConstant("CELL_SIZE_INV", ScalarDataType(), &gridCellSizeInv);
	InitSimulationConstant("CELL_COUNT", VectorDataType(false), &gridCellCount);

	// subprograms
	LoadSubprogram("grid utils", "scene/grid_utils.cl");
	LoadSubprogram("clear grid", "scene/grid_clear.cl");
	LoadSubprogram("set cell ids", "scene/grid_cellids.cl");
	LoadSubprogram("set cell start", "scene/grid_cellstart.cl");
	
	// TODO
	if(CLSystem::Instance()->FirstPlatform()->Name().find("ATI") != std::string::npos)
	{
		sorter = new GpuBitonicSort(program->Link()->Context(), program->Link()->Queue(0), ParticleCount());
		allocatedParticleCount = Utils::NearestMultiple(particleCount, 1024);
	}
	else
	{
		if(!InitRadixSort())
		{
			Log::Send(Log::Error, "Error when loading radix sort.");
			return false; 
		}
	}

	InitSimulationConstant("ALLOCATED_PARTICLE_COUNT", UintType, &allocatedParticleCount);
	InitSimulationBuffer("CELLS_HASH", UintType, allocatedParticleCount);
	InitSimulationBuffer("HASHES_PARTICLE", UintType, allocatedParticleCount);

	return true;
}

template<int dim, typename typ>
bool Simulation<dim, typ>::InitRadixSort()
{
	radixSortCta = 256;
	scanCta = 0;

	for(allocatedParticleCount = Utils::NearestMultiple(particleCount, 512); !scanCta; allocatedParticleCount += 512)
	{
		for (unsigned int s = 2; s <= std::min(program->Link()->Device(0)->MaxWorkGroupSize(), 512u); s *= 2u)
			if((allocatedParticleCount >= radixSortCta * s)
				&& (allocatedParticleCount <= radixSortCta*s*s/2u)
				&& (allocatedParticleCount % (radixSortCta*4) == 0)
				&& Utils::IsPowerOf2(8u * allocatedParticleCount / radixSortCta)
				&& ((2u * allocatedParticleCount) % (radixSortCta*s) == 0u)
				&& (2u * allocatedParticleCount / (radixSortCta*s) >= s))
			{
				scanCta = s;
			}

		if(scanCta)
			break;
	}

	unsigned int numBlocks = allocatedParticleCount / (radixSortCta * 4);
	unsigned int numBlocks2 = allocatedParticleCount / (radixSortCta * 2);
	unsigned int scanSize = 2 * allocatedParticleCount / (radixSortCta * scanCta);

	InitSimulationConstant("RADIX_STARTBIT", UintType, &numBlocks2);
	InitSimulationConstant("RADIX_BLOCK_COUNT", UintType, &numBlocks2);
	InitSimulationConstant("SCAN_SIZE", UintType, &scanSize);
	
	InitSimulationBuffer("CELLS_HASH_TEMP", UintType, allocatedParticleCount);
	InitSimulationBuffer("HASHES_PARTICLE_TEMP", UintType, allocatedParticleCount);
	InitSimulationBuffer("RADIX_COUNTERS", UintType, 32 * numBlocks);
	InitSimulationBuffer("RADIX_COUNTERS_SUM", UintType, 32 * numBlocks);
	InitSimulationBuffer("RADIX_BLOCK_OFFSETS", UintType, 32 * numBlocks);
	InitSimulationBuffer("SCAN_BUFFER", UintType, scanSize);

	LoadSubprogram("radix sort blocks", "scene/radix_sort_blocks.cl");
	LoadSubprogram("radix find offsets", "scene/radix_offsets.cl");
	LoadSubprogram("radix reorder", "scene/radix_reorder.cl");

	LoadSubprogram("scan 1", "scene/scan_1.cl");
	LoadSubprogram("scan 2", "scene/scan_2.cl");
	LoadSubprogram("scan uniform update", "scene/scan_uniform.cl");

	return true;
}


template<int dim, typename typ>
bool Simulation<dim, typ>::InitGeneral()
{
	if(!smoothingLength)
	{
		Log::Send(Log::Error, "You need to set smoothing length");
		return false;
	}

	if(!density)
	{
		Log::Send(Log::Error, "You need to set density");
		return false;
	}

	if(!particleSpacing)
	{
		Log::Send(Log::Error, "You need to set particle spacing");
		return false;
	}

	// count the particles
	for (unsigned int i=0; i<(unsigned int)ParticleTypeCount; i++)
		particleCountByType[i] = 0;

	typename std::multimap<std::string,Geometry<dim,typ>*>::iterator it;
	for(it=models.begin(); it != models.end(); it++)
	{
		it->second->startId = particleCountByType[(unsigned int)it->second->Type()];
		particleCountByType[(unsigned int)it->second->Type()] += it->second->ParticleCount();
	}

	particleCount = 0;
	for (unsigned int i=0; i<(unsigned int)ParticleTypeCount; i++)
		particleCount += particleCountByType[i];

	if(!particleCount)
	{
		Log::Send(Log::Error, "You need to make some geometry for the simulation");
		return false;
	}

	// general simulation variables
	timeStepCount = 0;
	timeOverall = 0;
	timeStep = 0;

	Vec<2,typ> lengthsInv(1/smoothingLength, 1/(smoothingLength*smoothingLength));
	typ densityInv = 1 / density;
	typ distEpsilon = ( smoothingLength * smoothingLength ) / 1000;
	particleMass[FluidParticle] = pow(particleSpacing, dim) * density;
	particleMass[BoundaryParticle] = particleMass[FluidParticle] / 2;

	InitSimulationConstant("PARTICLE_COUNT", UintType, &particleCount);
	InitSimulationConstant("FLUID_PARTICLE_COUNT", UintType, &particleCountByType[FluidParticle]);
	InitSimulationConstant("BOUNDARY_PARTICLE_COUNT", UintType, &particleCountByType[BoundaryParticle]);
	InitSimulationConstant("PARTICLE_SPACING", ScalarDataType(), &particleSpacing);

	InitSimulationConstant("MASS", ScalarDataType(), &particleMass[FluidParticle]);

	InitSimulationConstant("GRAVITY", VectorDataType(), &gravity);

	InitSimulationConstant("SMOOTHING_LENGTH", ScalarDataType(), &smoothingLength);
	InitSimulationConstant("SMOOTHING_LENGTH_INV", Scalar2DataType(), &lengthsInv);

	InitSimulationConstant("DENSITY", ScalarDataType(), &density);
	InitSimulationConstant("DENSITY_INV", ScalarDataType(), &densityInv);
    
	switch(viscosityFormulation)
	{
	case LaminarViscosity: 
	     InitSimulationConstant("DYNAMIC_VISCOSITY", ScalarDataType(), &dynamicViscosity);
   	     InitSimulationConstant("KINEMATIC_VISCOSITY", ScalarDataType(), &kinematicViscosity);
		 break;
	case ArtificialViscosity: 
	     InitSimulationConstant("ALPHA_VISCOSITY", ScalarDataType(), &alphaViscosity);
		 break;
	default: Log::Send(Log::Error, "Viscosity formulation choice is not correct.");
	}

	InitSimulationConstant("DIST_EPSILON", ScalarDataType(), &distEpsilon);

	InitSimulationConstant("TIME_STEP", ScalarDataType(), &timeStep);
	InitSimulationConstant("HALF_TIME_STEP", ScalarDataType(), &timeStep);
	InitSimulationBuffer("NEXT_TIME_STEP", ScalarDataType(), 1);

	InitSimulationConstant("VECTOR_VALUE", VectorDataType());
	InitSimulationConstant("OBJECT_START", UintType);
	InitSimulationConstant("OBJECT_PARTICLE_COUNT", UintType);

	// for tensile correction
	InitSimulationBuffer("DELTA_P", this->ScalarDataType(), 1);
	InitSimulationConstant("DELTA_P_INV", this->ScalarDataType());

	// program build options
	if(dim == 2)
		program->AddBuildOption("-D DIM=2");
	else
		program->AddBuildOption("-D DIM=3");

	if(ScalarPrecision() == 32)
		program->AddBuildOption("-D FP=32");
	else
		program->AddBuildOption("-D FP=64");
	
	// subprograms
	LoadSubprogram("types", "general/types.cl");
	LoadSubprogram("upload attribute", "general/attribute_upload.cl");

	switch(smoothingKernel)
	{
	case CubicSplineKernel:
		LoadSubprogram("kernel", "kernels/cubic.cl"); 
		program->AddBuildOption("-D CUBIC");
		break;
	case GaussKernel:
		LoadSubprogram("kernel", "kernels/gauss.cl"); 
		program->AddBuildOption("-D GAUSS");
		break;
	case ModifiedGaussKernel:
		LoadSubprogram("kernel", "kernels/gaussmod.cl"); 
		program->AddBuildOption("-D GAUSS");
		break;
	case WendlandKernel:
		LoadSubprogram("kernel", "kernels/wendland.cl");
		program->AddBuildOption("-D WENDLAND");
		break;
	}

	LoadSubprogram("deltaP", "kernels/delta_p.cl"); 
	
	return true;
}


template<int dim, typename typ>
void Simulation<dim, typ>::SetParticleSpacing( typ spacing )
{
	particleSpacing = spacing;
}


template<int dim, typename typ>
void Simulation<dim, typ>::SetParticleCount(ParticleType type, unsigned int count)
{
	if (particleCountByType[type] > 0) particleCount -= particleCountByType[type]; 
	particleCountByType[type] = count;
	particleCount += count;
}


template<int dim, typename typ>
void Simulation<dim, typ>::SetParticleMass( ParticleType type, typ mass )
{
	particleMass[type] = mass;
}

template<int dim, typename typ>
typ Simulation<dim, typ>::ParticleMass( ParticleType type)
{
	return particleMass[type];
}


template<int dim, typename typ>
Particle<dim,typ> Simulation<dim, typ>::GetParticle( ParticleType type, unsigned int id )
{
	if(type == FluidParticle)
		return Particle<dim,typ>(this, id);
	return Particle<dim,typ>(this, particleCountByType[FluidParticle] + id);
}



template<int dim, typename typ>
Particle<dim,typ> Simulation<dim, typ>::GetParticle( unsigned int id )
{
	return Particle<dim,typ>(this, id);
}


template<int dim, typename typ>
bool Simulation<dim, typ>::RunRadixSort()
{
	const unsigned int bitStep = 4;
	for (unsigned int i=0; i*bitStep < 32; i++)
	{
		unsigned int startBit = i*bitStep;
		program->Variable("RADIX_STARTBIT")->WriteFrom(&startBit);

		EnqueueSubprogram("radix sort blocks", allocatedParticleCount / 4, radixSortCta);
		EnqueueSubprogram("radix find offsets", allocatedParticleCount / 2, radixSortCta);
		EnqueueSubprogram("scan 1", 2 * allocatedParticleCount / radixSortCta, scanCta);
		EnqueueSubprogram("scan 2", 2 * allocatedParticleCount / (radixSortCta * scanCta), scanCta);
		EnqueueSubprogram("scan uniform update", 2 * allocatedParticleCount / radixSortCta, scanCta);
		EnqueueSubprogram("radix reorder", allocatedParticleCount / 2, radixSortCta);
	}
	
	return true;
}


template<int dim, typename typ>
bool Simulation<dim, typ>::RunGrid()
{
	// clear grid
	EnqueueSubprogram("clear grid");
	EnqueueSubprogram("set cell ids");

	// sort
	if(sorter)
		sorter->sort(program->Variable("CELLS_HASH")->Buffer(0), program->Variable("HASHES_PARTICLE")->Buffer(0));//, radixSortParticleCount, 32 /*size_of(cl_uint)*/);
	else
		RunRadixSort();

	// set cell start
	EnqueueSubprogram("set cell start");
    //enjalot
    printf("ij: done with rungrid\n");
    program->Finish();

	return true;
}


template<int dim, typename typ>
bool Simulation<dim, typ>::Advance( float advanceTimeStep )
{
    //enjalot
    printf("ij: in Simulation<dim, typ>::Advance\n");
	LogDebug("Advancing simulation");

	if(!program->IsBuilt())
	{
		Log::Send(Log::Error, "Cannot run unbuilt simulation");
		return false;
	}

	// upload particle data if it changed
	if(!UploadParticleData())
	{
		Log::Send(Log::Error, "Failed to send particle data to devices.");
		return false;
	}

	// update time step vars
	timeStep = advanceTimeStep;
	typ halfTimeStep = timeStep / 2;
	program->Variable("TIME_STEP")->WriteFrom(&timeStep);
	program->Variable("HALF_TIME_STEP")->WriteFrom(&halfTimeStep);

	// run the SPH simulation on devices
	RunSph();
    printf("ij: Advance:: RunSph();\n");

	// TODO dont wait to finish
	program->Finish();
    printf("ij: Advance:: finished\n");

	// advance sim time
	timeOverall += advanceTimeStep; 
	timeStepCount++;

	// Store probes data 
	probeManager->ReadProbes(timeStepCount,timeOverall);
	
	return true;
}


template<int dim, typename typ>
bool Simulation<dim, typ>::UploadParticleData()
{
	bool positionsHaveChanged = false;
	bool success = true;
	for (std::map<std::string, ParticleAttributeBuffer*>::iterator i = particleAttributes.begin(); i != particleAttributes.end(); i++)
	{
		// TODO put out of loop
		if(i->first == "POSITIONS" && i->second->HostDataChanged()) 
			positionsHaveChanged = true;

		success &= i->second->Upload();
	}

	// since we changed particles, refresh the grid
	if(success && positionsHaveChanged)
		RunGrid();

	return success;
}


template<int dim, typename typ>
bool Simulation<dim, typ>::DownloadParticleData( const std::string& attribute )
{
	std::map<std::string, ParticleAttributeBuffer*>::iterator it = particleAttributes.find(attribute);

	if(it == particleAttributes.end())
	{
		Log::Send(Log::Error, "Particle attribute you want to read doesn't exist");
		return false;
	}

	if(!it->second->Download())
	{
		Log::Send(Log::Error, "Cannot download particle data");
		return false;
	}

	return true;
}


template<int dim, typename typ>
size_t Simulation<dim, typ>::UsedMemorySize()
{
	return program->UsedMemorySize();
}


template<int dim, typename typ>
typ Simulation<dim, typ>::SuggestTimeStep()
{
	return timeStep;
}


template<int dim, typename typ>
Geometry<dim,typ>* Simulation<dim, typ>::GetGeometry( const std::string& name )
{
	typename std::multimap<std::string,Geometry<dim,typ>*>::iterator it = models.find(name);
	if(it == models.end())
	{
		Log::Send(Log::Warning, "Couldn't find geometry: " + name);
		return NULL;
	}
	return it->second;
}


template<int dim, typename typ>
ParticleAttributeBuffer* Simulation<dim, typ>::ParticleAttribute( const std::string& name )
{
	std::map<std::string, ParticleAttributeBuffer*>::iterator it = particleAttributes.find(name);
	return (it == particleAttributes.end()) ? NULL : it->second;
}


template<int dim, typename typ>
VariableDataType Simulation<dim, typ>::ScalarDataType()
{
	if(sizeof(typ) == sizeof(cl_float))
		return FloatType;
	else //if(sizeof(typ) == sizeof(cl_double))
		return DoubleType;
}


template<int dim, typename typ>
VariableDataType Simulation<dim, typ>::Scalar2DataType()
{
	if(sizeof(typ) == sizeof(cl_float))
		return Float2Type;
	else //if(sizeof(typ) == sizeof(cl_double))
		return Double2Type;
}


template<int dim, typename typ>
VariableDataType Simulation<dim, typ>::Scalar4DataType()
{
	if(sizeof(typ) == sizeof(cl_float))
		return Float4Type;
	else //if(sizeof(typ) == sizeof(cl_double))
		return Double4Type;
}


template<int dim, typename typ>
VariableDataType Simulation<dim, typ>::VectorDataType(bool realNumbers)
{
	if(realNumbers)
	{
		if(dim == 2 && sizeof(typ) == sizeof(cl_float))
			return Float2Type;
		else if(dim == 2 && sizeof(typ) == sizeof(cl_double))
			return Double2Type;
		else if(dim == 3 && sizeof(typ) == sizeof(cl_float))
			return Float4Type;
		else //if(dim == 3 && sizeof(typ) == sizeof(cl_double))
			return Double4Type;
	}
	else
	{
		if(dim == 2)
			return Int2Type;
		else //if(dim == 3)
			return Int4Type;
	}
}


// explicit specializations
template class Simulation<2,float>;

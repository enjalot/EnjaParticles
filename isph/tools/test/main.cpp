#include <isph.h>
#include <iostream>
#include <stdlib.h>
#include <vec.h>
#include <time.h>

// for templates
#include <wcsphsimulation.h>

using namespace std;
using namespace isph;

//////////////////////////////////////////////////////////////////////////

void BuildWaterDrop(WcsphSimulation<2,float>* sim, int resFactor)
{
	float L = 1.0f;
	sim->SetBoundaries(Vec2f(-1*L, -2*L), Vec2f(1*L, 2*L));
	sim->SetGravity(Vec2f(0,0));
	sim->SetViscosityFormulationType(ArtificialViscosity);
	sim->SetAlphaViscosity(0.01f);
	sim->SetDynamicViscosity(0.01f);
	sim->SetWcsphParameters(1400.f,7.0f);
	sim->SetDensityReinitMethod(MovingLeastSquares);
    sim->SetDensityReinitFrequency(20);

	
	int minRes = 40;
	sim->SetParticleSpacing(L/(minRes*resFactor));
	geo::Sphere<2,float>* dropSphere = new geo::Sphere<2,float>(sim, FluidParticle, "WaterDrop");
	dropSphere->Define(Vec<2,float>(0,0), L);
	dropSphere->Fill();
}

void SetWaterDrop(WcsphSimulation<2,float>* sim)
{
    printf("setting water drop\n");
	Geometry<2,float>* dropSphere = sim->GetGeometry("WaterDrop");
	float spacing = sim->ParticleSpacing();
	for (unsigned int i=0; i<dropSphere->ParticleCount(); i++)
	{
		float a0 = 100.0f;
		Particle<2,float> p = sim->GetParticle(dropSphere->Type(), dropSphere->ParticleStartId() + i);

		Vec<2,float> pos = p.Position();
		p.SetVelocity(Vec2f(-a0*pos.x, a0*pos.y));
		float pressure = 0.5f*sim->Density()*(a0 * a0)* (1 - (pos.x*pos.x + pos.y*pos.y));
		float density = sim->GetDensityFromPressure(pressure); 
		p.SetDensity(density);
		p.SetPressure(pressure);
		p.SetMass(density * spacing * spacing);
	}
    printf("dropSphere->ParticleCount(): %d\n", dropSphere->ParticleCount());
}

//////////////////////////////////////////////////////////////////////////

void BuildDryDamBreak1(WcsphSimulation<2,float>* sim, int resFactor)
{
    int minRes = 38;
	float dx = 0.38f/(minRes*resFactor);
	
	sim->SetBoundaries(Vec2f(0,0), Vec2f(1.38f,0.2f));
	sim->SetParticleSpacing(dx);
	sim->SetGravity(Vec2f(0,-Consts::g));
	sim->SetViscosityFormulationType(ArtificialViscosity);
	sim->SetAlphaViscosity(0.01f);
	sim->SetWcsphParameters(45, 7); // water speed of sound and wcsph parameter

	geo::Box<2,float>* waterRect = new geo::Box<2,float>(sim, FluidParticle, "WaterColumn");
	waterRect->Define(Vec2f(0.5f*dx,0.5f*dx), Vec2f(0.38f,0.15f));
	waterRect->Fill();

	geo::Line<2,float>* bottomWall = new geo::Line<2,float>(sim, BoundaryParticle,"BottomWall");
	bottomWall->Define(Vec2f(0,0),Vec2f(1.38f, 0), 2, 0.5f*dx);

	geo::Line<2,float>* leftWall = new geo::Line<2,float>(sim, BoundaryParticle,"LeftWall");
	leftWall->Define(Vec2f(0, dx), Vec2f(0, 0.2f), 2, 0.5f*dx);
	leftWall->InvertLayerOrientation(true);

	geo::Line<2,float>* rightWall = new geo::Line<2,float>(sim, BoundaryParticle,"RightWall");
	rightWall->Define(Vec2f(1.38f, dx), Vec2f(1.38f, 0.2f), 2, 0.5f*dx);
	rightWall->InvertLayerOrientation(true);

}

//////////////////////////////////////////////////////////////////////////

void BuildDryDamBreak2(WcsphSimulation<2,float>* sim, int resFactor)
{
    int minRes = 32;   
    float H = 2.0;
	float dx = (0.5f*H)/(minRes*resFactor);

	sim->SetParticleSpacing(dx);
	sim->SetGravity(Vec2f(0,-Consts::g));
	sim->SetViscosityFormulationType(ArtificialViscosity);
	sim->SetAlphaViscosity(0.03f);
	sim->SetWcsphParameters(45, 7); // water speed of sound and wcsph parameter
	sim->SetDensityReinitMethod(MovingLeastSquares);
    sim->SetDensityReinitFrequency(20);
	sim->SetBoundaries(Vec2f(-4 * dx, -4 * dx), Vec2f(2*H + 4*dx, 2*H + 4*dx)); // set appropriate grid boundary rectangle

	geo::Box<2,float>* fluid = new geo::Box<2,float>(sim,FluidParticle,"WaterColumn");
	fluid->Define(Vec2f(0.5f*dx,0.5f*dx), Vec2f(0.5f*H,H));
	fluid ->Fill(); 

	geo::Line<2,float>* bottomWall = new geo::Line<2,float>(sim,BoundaryParticle,"BottomWall");
	bottomWall->Define(Vec2f(0,0), Vec2f(2.0f*H,0), 2, 0.5f*dx);

	geo::Line<2,float>* leftWall = new geo::Line<2,float>(sim, BoundaryParticle,"LeftWall");
	leftWall->Define(Vec2f(0, dx), Vec2f(0, 1.5f*H), 2, 0.5f*dx);
	leftWall->InvertLayerOrientation(true);

	geo::Line<2,float>* rightWall = new geo::Line<2,float>(sim, BoundaryParticle,"RightWall");
	rightWall->Define(Vec2f(2.0f*H, dx), Vec2f(2.0f*H, 2.0f*H), 2, 0.5f*dx);

}

void SetDryDamBreak2(WcsphSimulation<2,float>* sim)
{
    float H = 2.0;
	Geometry<2,float>* water = sim->GetGeometry("WaterColumn");
	Geometry<2,float>* lwall = sim->GetGeometry("LeftWall");
	Geometry<2,float>* rwall = sim->GetGeometry("RightWall");
	Geometry<2,float>* bwall = sim->GetGeometry("BottomWall");
	float spacing = sim->ParticleSpacing();
	for (unsigned int i=0; i<water ->ParticleCount(); i++)
	{
		Particle<2,float> p = sim->GetParticle(water->Type(), water->ParticleStartId() + i);
		Vec<2,float> pos = p.Position();
		float density = sim->Density();
		float pressure = (pos.y - H)*sim->Density()*sim->Gravity().y;
		density = sim->GetDensityFromPressure(pressure);
		p.SetDensity(density);
		p.SetPressure(pressure);
		p.SetMass(density * spacing * spacing);
	}
	for (unsigned int i=0; i<lwall->ParticleCount(); i++)
	{
		Particle<2,float> p = sim->GetParticle(lwall->Type(), lwall->ParticleStartId() + i);
		Vec<2,float> pos = p.Position();
		float density = sim->Density();
		if (pos.y < H) 
		{
		  float pressure = (pos.y - H)*sim->Density()*sim->Gravity().y;
		  density = sim->GetDensityFromPressure(pressure);
		  p.SetDensity(density);
		  p.SetPressure(pressure);
		}
		p.SetMass(0.5f * spacing * spacing * sim->Density());
	}
	for (unsigned int i=0; i<rwall->ParticleCount(); i++)
	{
		Particle<2,float> p = sim->GetParticle(rwall->Type(), rwall->ParticleStartId() + i);
		Vec<2,float> pos = p.Position();
		float density = sim->Density();
		p.SetMass(0.5f * spacing * spacing * sim->Density());
	}
	for (unsigned int i=0; i<bwall->ParticleCount(); i++)
	{
		Particle<2,float> p = sim->GetParticle(bwall->Type(), bwall->ParticleStartId() + i);
		Vec<2,float> pos = p.Position();
		float pressure = (pos.y - H)*sim->Density()*sim->Gravity().y;
		float density = sim->Density();
		if (pos.x < 0.5*H) 
		{
		  float pressure = (pos.y - H)*sim->Density()*sim->Gravity().y;
		  density = sim->GetDensityFromPressure(pressure);
		  p.SetDensity(density);
		  p.SetPressure(pressure);
		}
		p.SetMass(0.5f * spacing * spacing * sim->Density());
	}

}

//////////////////////////////////////////////////////////////////////////

void BuildDropFall(WcsphSimulation<2,float>* sim, int resFactor)
{

	sim->SetBoundaries(Vec2f(0,0), Vec2f(2.0f,2.0f));

	geo::Sphere<2,float>* dropSphere = new geo::Sphere<2,float>(sim, FluidParticle,"WaterSphere");
	dropSphere->Define(Vec2f(1.0f,1.5f), 0.3f);
	dropSphere->Fill();

	geo::Box<2,float>* fluidRect = new geo::Box<2,float>(sim, FluidParticle, "WaterTank" );
	fluidRect->Define(Vec2f(0,0), Vec2f(2.0f,0.5f));
	fluidRect->Fill();

	geo::Box<2,float>* wallRect = new geo::Box<2,float>(sim, BoundaryParticle, "Box");
	wallRect->Define(Vec2f(0,0), Vec2f(2.0f,2.0f));
}

//////////////////////////////////////////////////////////////////////////

void BuildHydrostatic(WcsphSimulation<2,float>* sim, int resFactor)
{
	int minRes = 20;
	sim->SetViscosityFormulationType(LaminarViscosity);
	sim->SetDynamicViscosity(0.01f);
	sim->SetWcsphParameters(100.0f,7.0f);

	float dx = 1.0f/(minRes*resFactor);
    
	sim->SetParticleSpacing(dx);
	sim->SetBoundaries(Vec2f(0 - 2.0f*dx,0 - 2.0f*dx), Vec2f(1.0f + 2.0f*dx,1.5f + 2.0f*dx));

	geo::Box<2,float>* fluidRect = new geo::Box<2,float>(sim, FluidParticle, "WaterTank");
	fluidRect->Define(Vec2f(0.5f*dx,0.5f*dx), Vec2f(1.0f,1.0f));
	fluidRect->Fill();

	geo::Line<2,float>* bWall = new geo::Line<2,float>(sim,BoundaryParticle, "BottomWall");
	bWall->Define(Vec2f(0,0), Vec2f(1.0f+0.5f*dx,0), 2, 0.5f*dx);

	geo::Line<2,float>* lWall = new geo::Line<2,float>(sim,BoundaryParticle, "LeftWall");
	lWall->Define(Vec2f(0,dx), Vec2f(0,1.5f-dx), 2, 0.5f*dx);
	lWall->InvertLayerOrientation(true);

	geo::Line<2,float>* rWall = new geo::Line<2,float>(sim,BoundaryParticle, "RightWall");
	rWall->Define(Vec2f(1.0f, dx), Vec2f(1.0f,1.5f-dx), 2, 0.5f*dx);
}

void SetHydrostatic(WcsphSimulation<2,float>* sim)
{
    float H = 1;
	Geometry<2,float>* tank = sim->GetGeometry("WaterTank");
	Geometry<2,float>* lwall = sim->GetGeometry("LeftWall");
	Geometry<2,float>* rwall = sim->GetGeometry("RightWall");
	for (unsigned int i=0; i<tank->ParticleCount(); i++)
	{
		Particle<2,float> p = sim->GetParticle(tank->Type(), tank->ParticleStartId() + i);
		Vec<2,float> pos = p.Position();
		float pressure = (pos.y - H)*sim->Density()*sim->Gravity().y;
		p.SetDensity(sim->GetDensityFromPressure(pressure));
		p.SetPressure(pressure);
	}
	/*for (unsigned int i=0; i<lwall->ParticleCount(); i++)
	{
		Particle<2,float> p = sim->GetParticle(lwall->Type(), lwall->ParticleStartId() + i);
		Vec<2,float> pos = p.Position();
		float pressure = (pos.y - H)*sim->Density()*sim->Gravity().y;
		p.SetDensity(sim->GetDensityFromPressure(pressure));
		p.SetPressure(pressure);
	}
	for (unsigned int i=0; i<rwall->ParticleCount(); i++)
	{
		Particle<2,float> p = sim->GetParticle(rwall->Type(), rwall->ParticleStartId() + i);
		Vec<2,float> pos = p.Position();
		float pressure = (pos.y - H)*sim->Density()*sim->Gravity().y;
		p.SetDensity(sim->GetDensityFromPressure(pressure));
		p.SetPressure(pressure);
	}*/

}
//////////////////////////////////////////////////////////////////////////

void BuildSquarePatch(WcsphSimulation<2,float>* sim, int resFactor)
{
	sim->SetGravity(Vec2f(0,0));
	sim->SetViscosityFormulationType(ArtificialViscosity);
	sim->SetAlphaViscosity(0.01f);
	sim->SetWcsphParameters(140.0f,7.0f);
	sim->SetDensityReinitFrequency(30);

	int minRes = 50;
	float L = 1.0f;
    float dx = L/( minRes*resFactor);

	sim->SetParticleSpacing(dx);    
	sim->SetBoundaries(Vec2f(-2.0f * L,-2.0f * L), Vec2f( 2.0f * L,2.0f * L)); // set appropriate grid boundary rectangle

	geo::Box<2,float>* fluidRect = new geo::Box<2,float>(sim, FluidParticle,"FluidPatch");
	fluidRect->Define(Vec2f(-0.5f*L,-0.5f*L), Vec2f(0.5f*L,0.5f*L));
	fluidRect->Fill();
}

void SetSquarePatch(WcsphSimulation<2,float>* sim)
{
	float omega = 1.0f;
    float L = 1.0f;
	unsigned int mmax = 10;
    unsigned int nmax = 10;

	Geometry<2,float>* lid = sim->GetGeometry("FluidPatch");
	for (unsigned int i=0; i<lid ->ParticleCount(); i++)
	{
		Particle<2,float> p = sim->GetParticle(lid ->Type(), lid ->ParticleStartId() + i);
		Vec<2,float> pos = p.Position();
		p.SetVelocity(Vec2f(omega * pos.y, - omega * pos.x));
	    // Calculate pressure from series expansion Colagrassi Phd Thesis
		float pressure = 0;
		for ( unsigned int m = 1; m<mmax; m+=2  )
			for ( unsigned int n = 1; n<nmax; n+=2  )
			{
				float tmp = -32.0f * ( omega * omega ) / ( m * n * (float)Consts::Pi * (float)Consts::Pi );
				    tmp /=  (( m * (float)Consts::Pi /L ) * ( m * (float)Consts::Pi /L ) + ( n * (float)Consts::Pi /L ) * ( n * (float)Consts::Pi /L )); 
					tmp *=  sinf(( m * (float)Consts::Pi * ( pos.x + 0.5f * L ) ) /L );
 	                tmp *=  sinf(( n * (float)Consts::Pi * ( pos.y + 0.5f * L) ) /L );

				pressure += tmp;
			}
		pressure *= sim->Density();
		p.SetPressure(pressure);
		p.SetDensity(sim->GetDensityFromPressure(pressure));
	}
}

//////////////////////////////////////////////////////////////////////////

bool running = true;
void RecieveLogMessage(const Log::Message& m)
{
	if(m.type != Log::DebugInfo)
		cout << m.text << endl;
	if(m.type == Log::Error)
		running = false;
}

int main(int argc, char *argv[])
{
	Log::SetOutput("test_log.txt"); // output everything to log file
	Log::SetLevel(Log::Info);
	Log::SetUserReceiver(&RecieveLogMessage);
	CLSystem* sys = CLSystem::Instance();

	if (argc < 3)
	{
		cout << "Usage: test [test index] [particle density] [optional export step]" << endl << endl;

		cout << "** Test indices **" << endl;
		cout << "  0: evolution of elliptic water drop" << endl;
		cout << "  1: dry bed dam break #1" << endl;
		cout << "  2: dry bed dam break #2" << endl;
		cout << "  3: water drop free fall" << endl;
		cout << "  4: hydrostatic pressure" << endl;
		cout << "  5: evolution of square patch" << endl;
		cout << endl;

		cout << "** Particle density **" << endl;
		cout << "  1: sparse" << endl;
		cout << "  ..." << endl;
		cout << "  10: pretty dense" << endl;
		cout << endl;

		cout << "** Export step **" << endl;
		cout << "  in seconds, default 0.01" << endl;
		cout << endl;

		cout << "** OpenCL information **" << endl;

		for(unsigned int i=0; i<sys->PlatformCount(); i++)
		{
			CLPlatform* pl = sys->Platform(i);
			cout << endl << "Platform " << i << ": " << pl->Name() << endl;

			for (unsigned int j=0; j<pl->DeviceCount(); j++)
			{
				CLDevice* dev = pl->Device(j);
				cout << endl << "  Device " << j << ": " << dev->Name() << endl;
				cout << "  Cores: " << dev->Cores() << endl;
				cout << "  Max frequency: " << dev->MaxFrequency() << " MHz" << endl;
				cout << "  Global memory: " << dev->GlobalMemorySize()/1024/1024 << " Mb" << endl;
				cout << "  Local memory: " << dev->LocalMemorySize()/1024 << " kb" << endl;
				cout << "  Double floating precision: " << dev->DoublePrecision() << endl;
				cout << "  Half floating precision: " << dev->HalfPrecision() << endl;
				cout << "  Atomic functions: " << dev->Atomics() << endl;
			}
		}
		
		return 0;
	}

	Log::Send(Log::Info, "Building the test.");

	// create simulation and set its parameters
    WcsphSimulation<2,float> sim;
	sim.SetDensity((float)Consts::Water::StdDensity);
	sim.SetGravity(Vec2f(0,-Consts::g));
	sim.SetViscosityFormulationType(ArtificialViscosity);
	sim.SetDynamicViscosity(0.02f); 	
	sim.SetAlphaViscosity(0.01f); 	
	sim.SetWcsphParameters(45, 7); // water speed of sound and wcsph parameter
	sim.SetXsphFactor(0.5f); 
	sim.SetDevices(new CLLink(sys->FirstPlatform()->Device(0)));

	// create geometry

	int testCaseNumber = atoi(argv[1]);
	int resFactor = atoi(argv[2]);
	switch(testCaseNumber)
	{
	case 0: BuildWaterDrop(&sim,resFactor); break;
	case 1: BuildDryDamBreak1(&sim,resFactor); break;
	case 2: BuildDryDamBreak2(&sim,resFactor); break;
	case 3: BuildDropFall(&sim,resFactor); break;
	case 4: BuildHydrostatic(&sim,resFactor); break;
	case 5: BuildSquarePatch(&sim,resFactor); break;
	default: cout << "Test case " << testCaseNumber << "doesn't exist." << endl; return 0;
	}

	// other parameters
	Vec<2,float> simSize = sim.BoundaryMax() - sim.BoundaryMin();
	sim.SetSmoothingKernel(CubicSplineKernel, 1.4f * sim.ParticleSpacing());

	// prepare VTK exporter
	VtkWriter<2,float> exporter(&sim);
	exporter.SetOutput("test_results");
	exporter.AddAttribute("PRESSURES");
    exporter.AddAttribute("DENSITIES");
    exporter.AddAttribute("VELOCITIES");

    // prepare Probe Manager
	ProbeManager<2,float>* probeManager = sim.GetProbeManager(); 
    probeManager->AddAttribute("PRESSURES");
	probeManager->AddAttribute("VELOCITIES");
    probeManager->AddProbe(Vec2f(0,0));
    probeManager->AddProbe(Vec2f(0,0.5f));
    probeManager->AddProbe(Vec2f(0.5f,0));

	// initialize simulation
	cout << "Initializing simulation solver." << endl;
	if(!sim.Initialize())
		return 0;
	
	cout << "Simulation allocated " << sim.UsedMemorySize()/1024/1024 << "Mb, for " << sim.ParticleCount() << " particles." << endl;

	// some additional initialization
	switch(testCaseNumber)
	{
	case 0: SetWaterDrop(&sim); break;
	case 2: SetDryDamBreak2(&sim); break;
	default: break;
	}

    printf("Made it past testCaseNumber\n");

	const double simMaxTime = 4.000;   // sec
	const double exportingStep = (argc>3) ? atof(argv[3]) : 0.01; // sec
	double nextTimeStep = 1e-6;        // sec
	double lastExportTime = -1;
	double minTimeStep = nextTimeStep;

    printf("about to do exporter.Prepare()\n");

	exporter.Prepare();
    exporter.WriteData(); // write scene setup

	clock_t startTime = clock();
	clock_t endTime;
	
	unsigned int stepCounts = 0;

    printf("about to start simulation loop\n");

	while(sim.Time() < simMaxTime && running)
	{
        printf("sim.time() < simMaxTime && running\n");
		// advance with CFL
		if(sim.Advance((float)nextTimeStep))
		{
			stepCounts++;
            printf("stepCounts: %d\n", stepCounts);
			// write results every something time step
			nextTimeStep = sim.SuggestTimeStep();
            printf("next time step %f\n" , (float)nextTimeStep);
			minTimeStep = std::min(nextTimeStep,minTimeStep);
			if(sim.Time() >= (exporter.FileIndex() * exportingStep))
			{
				endTime = clock();
				exporter.WriteData();
				lastExportTime = sim.Time();
				cout << "Exported at: " <<  lastExportTime << "s, min time-step was: " << minTimeStep << "s, it took: " << (endTime-startTime)*1000/CLOCKS_PER_SEC << "ms and: " << stepCounts << " advancing steps." << endl;
			    minTimeStep  = nextTimeStep;
				stepCounts = 0;
				startTime = clock();
			}
		}
		else
		{
			Log::Send(Log::Error, "Simulation failed to advance.");
			break;
		}
	}

	Log::Send(Log::Info, "Finished simulating.");   
	return 0;
}

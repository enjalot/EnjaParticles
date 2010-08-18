

#include <cuda_runtime_api.h>
#include <iostream>
#include <iomanip>
#include <numeric>
#include <limits>
#include <conio.h> 

#undef SPHSIMLIB_3D_SUPPORT
#include "SimulationSystem.h"

using namespace std;
using namespace SimLib;
using namespace ocu;

void pause()
{
	_getch(); 

// 	cin.clear();
// 	cin.ignore(std::numeric_limits<streamsize>::max());   //
// 	cout << "Press Enter to continue . . .\n";
// 	cin.ignore(std::numeric_limits<streamsize>::max(),'\n');
}
void testFluidSimLive()
{
	SimulationSystem *system = new SimulationSystem(true);
	system->Init();

	system->GetSettings()->SetValue("Timestep", 0.002);
	system->GetSettings()->SetValue("Particles Number", 128*1024);
	system->GetSettings()->SetValue("Grid World Size",  1024);
	system->GetSettings()->SetValue("Simulation Scale",  0.0005);
	system->GetSettings()->SetValue("Rest Density",  1000);
	system->GetSettings()->SetValue("Rest Pressure", 0);
	system->GetSettings()->SetValue("Ideal Gas Constant",  1.5);
	system->GetSettings()->SetValue("Viscosity",  1);
	system->GetSettings()->SetValue("Boundary Stiffness", 30);
	system->GetSettings()->SetValue("Boundary Dampening", 30);

	system->GetSettings()->Print();
	cout << "\n";
	//system->SetPrintTiming(true);
	system->SetScene(6);

	cout << "\nFPS:\n";
	cout << setw(15) << "Current";
	cout << setw(15) << "Avg (10)";
	cout << setw(15) << "Avg Total";
	cout << "\n";

	double totalavg;
	double fpshistory[100] = {0};
	GPUTimer *timer = new GPUTimer();

	 //CPUTimer *totalTimer = new CPUTimer();
	 //totalTimer->start();

	int ITERATIONS = 200;
	for(int i = 0; i < ITERATIONS; i++)
	{
		timer->start();
		system->Simulate(true, true);
		timer->stop();

		// calc fps
		double fps = 1000.0/timer->elapsed_ms();

		// calc running average of full history
		totalavg = (fps+i*totalavg)/(i+1);

		// store fps in history
		fpshistory[i%100] = fps;

		// get average of fps history
		double avg=0; for(int j=0;j<100;j++) fpshistory[j]==0? avg += fps : avg += fpshistory[j]; avg /= 100.0;

		cout << fixed;
		cout << setw(15) << setprecision(1) << fps ;
		cout << setw(15) << setprecision(1) << avg ;
		cout << setw(15) << setprecision(1) <<  totalavg;
		cout << "\r";
	}
	CUDA_SAFE_CALL(cudaThreadSynchronize());	

	// 	totalTimer->stop();
	// 	double totalTime = totalTimer->elapsed_ms();
	// 	cout << "Total ms: " << totalTime << "\n";
	// 	cout << "Avg ms/frame: " << totalTime/ITERATIONS << "\n";
	// 	cout << "Avg fps: " << 1000.0/(totalTime/ITERATIONS) << "\n";

	pause();
}

void testFluidSim()
{
	SimulationSystem *system = new SimulationSystem(false);
	system->Init();

	system->GetSettings()->SetValue("Timestep", 0.0005);
	system->GetSettings()->SetValue("Particles Number", 512*1024);
	system->GetSettings()->SetValue("Grid World Size",  1024);
	system->GetSettings()->SetValue("Simulation Scale",  0.0005);
	system->GetSettings()->SetValue("Rest Density",  1000);
	system->GetSettings()->SetValue("Rest Pressure", 0);
	system->GetSettings()->SetValue("Ideal Gas Constant",  1.5);
	system->GetSettings()->SetValue("Viscosity",  1);
 	system->GetSettings()->SetValue("Boundary Stiffness", 20000);
 	system->GetSettings()->SetValue("Boundary Dampening", 256);

	system->GetSettings()->Print();
	cout << "\n";
	//system->SetPrintTiming(true);
	system->SetScene(6);

	//GPUTimer *totalTimer = new GPUTimer();
	//totalTimer->start();

	system->PrintMemoryUse();

	int ITERATIONS = 1000;
	for(int i = 0; i < ITERATIONS; i++)
	{
		system->Simulate(true, true);
	}
	//CUDA_SAFE_CALL(cudaThreadSynchronize());

	
	//totalTimer->stop();
	//double totalTime = totalTimer->elapsed_ms();
	//cout << "Total ms: " << totalTime << "\n";
	//cout << "Avg ms/frame: " << totalTime/ITERATIONS << "\n";
	//cout << "Avg fps: " << 1000.0/(totalTime/ITERATIONS) << "\n";

	//pause();
}

void testPerformanceScaling()
{
	SimulationSystem *system = new SimulationSystem(true);
	system->Init();

	system->GetSettings()->SetValue("Timestep", 0.0005);
	system->GetSettings()->SetValue("Grid World Size",  1024);
	system->GetSettings()->SetValue("Simulation Scale",  0.0005);
	system->GetSettings()->SetValue("Rest Density",  1000);
	system->GetSettings()->SetValue("Rest Pressure", 0);
	system->GetSettings()->SetValue("Ideal Gas Constant",  1.5);
	system->GetSettings()->SetValue("Viscosity",  1);
	system->GetSettings()->SetValue("Boundary Stiffness",  20000);
	system->GetSettings()->SetValue("Boundary Dampening", 256);

	system->GetSettings()->Print();
	cout << "\n";

	uint startParticles = 256*1024;
	uint endParticles = 512*1024;

	int ITERATIONS = 1000;

	float psizes[100] = {0};
	float vals[100] = {0};

	uint DOUBLINGS = 0;
	for(uint numParticles = startParticles; numParticles<=endParticles; DOUBLINGS++,numParticles *= 2) 
	{
		psizes[DOUBLINGS] = numParticles;
		system->GetSettings()->SetValue("Particles Number", numParticles);

		//system->SetPrintTiming(true);
		system->SetScene(6);

	 	CPUTimer *totalTimer = new CPUTimer();
 		totalTimer->start();

		cout << "\n";
		for(int i = 0; i < ITERATIONS; i++)
		{
			system->Simulate(true, true);
			//cout << "\r" << i << "/" << ITERATIONS;
		}
		CUDA_SAFE_CALL(cudaThreadSynchronize());	

		totalTimer->stop();
		double totalTime = totalTimer->elapsed_ms();
		vals[DOUBLINGS] = totalTime;
		cout << "\nRESULTS: Total ms: " << totalTime << "\t";
		cout << "Avg ms/frame: " << totalTime/ITERATIONS << "\t";
		cout << "Avg fps: " << 1000.0/(totalTime/ITERATIONS) << "\n\n";

	}
	cout << "SUMMARY OF TIMES:\n";
	for(int i = 0; i<DOUBLINGS;i++)
		cout << setw(10) << psizes[i] << " ";
	cout << "\n";
	for(int i = 0; i<DOUBLINGS;i++)
		cout << setw(10) << 1000.0/(vals[i]/ITERATIONS) << ", ";

	pause();
}

void testKernel();

int main(int argc, char *argv[])
{
	SimLib::SimCudaHelper* simCudaHelper = new SimLib::SimCudaHelper();
	simCudaHelper->Initialize(1);

	//force the GPU to wake up
	cudaEvent_t wakeGPU;
	cutilSafeCall( cudaEventCreate( &wakeGPU) );
	//Sleep(1000);


//	testPerformanceScaling();
	//testFluidSimLive();
	testFluidSim();
	//testKernel();
	
}



#include "pcintegrator.h"
#include "simulation.h"


using namespace isph;

template<int dim, typename typ>
PCIntegrator<dim, typ>::PCIntegrator(Simulation<dim,typ>* simulation) :
	AbstractIntegrator<dim, typ>::AbstractIntegrator(simulation)
{

}

template<int dim, typename typ>
PCIntegrator<dim, typ>::PCIntegrator(Simulation<dim,typ>* simulation, bool regrid) :
	AbstractIntegrator<dim, typ>::AbstractIntegrator(simulation, regrid)
{


}


template<int dim, typename typ>
PCIntegrator<dim, typ>::~PCIntegrator()
{

}

template<int dim, typename typ>
bool PCIntegrator<dim, typ>::Integrate()
{
	
    printf("starting integrate\n");
  	// calculate particle accelerations
	sim->EnqueueSubprogram("acceleration");
    printf("enqueued acceleration\n");

	// calculate particle viscous acceleration
	//sim->EnqueueSubprogram("viscosity");
    //printf("enqueued viscosity\n");

	// correct velocities 
	sim->EnqueueSubprogram("xsph");
    printf("enqueued xsph\n");

	// calculate density rate of change with continuity eq.
	sim->EnqueueSubprogram("continuity");
    printf("enqueued continuity\n");

	// copy buffers since we used XSPH_VELOCITIES as a temp buffer
	sim->WriteVariableFrom("VELOCITIES_TMP","VELOCITIES");
    sim->WriteVariableFrom("POSITIONS_TMP","POSITIONS");
    sim->WriteVariableFrom("DENSITY_TMP","DENSITIES");

	// Update particle position, velocities and densities at predictor step
    sim->EnqueueSubprogram("predictor");
    printf("enqueued predictor\n");

	// reinit grid with new particle positions
	if (doRegrid)
	  sim->RunGrid(); 

	// Update particle pressure by Equation of State
    sim->EnqueueSubprogram("eos");
    printf("enqueued eos\n");

	// calculate particle accelerations
	sim->EnqueueSubprogram("acceleration");
    printf("enqueued acceleration\n");

	// calculate particle viscous acceleration
	//sim->EnqueueSubprogram("viscosity");
    //printf("enqueued viscosity\n");

	// correct velocities 
	sim->EnqueueSubprogram("xsph");
    printf("enqueued xsph\n");

	// calculate density rate of change with continuity eq.
	sim->EnqueueSubprogram("continuity");
    printf("enqueued continuity\n");

	// Update particle position, velocities and densities at corrector step
    sim->EnqueueSubprogram("corrector");
    printf("enqueued corrector\n");

	// Update particle pressure by Equation of State
    sim->EnqueueSubprogram("eos");
    printf("enqueued eos\n");

	// reinit grid with new particle positions
	sim->RunGrid(); 

	return true;

}




//////////////////////////////////////////////////////////////////////////

template class PCIntegrator<2,float>;

//#endif

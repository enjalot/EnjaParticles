
#include <GL/glew.h>
#include <math.h>

#include "GE_SPH.h"
#include "../particle/UniformGrid.h"

#include "utils/wpoly6_cpu.cpp"

// GE: need it have access to my datastructure (GE). Will remove 
// eventually. 
//#include "datastructures.h"

namespace rtps {


//----------------------------------------------------------------------
//void GE_SPH::setGEDataStructures(DataStructures* ds)
//{
	//this->ds = ds;
//}

//----------------------------------------------------------------------

#include "new_parameters.cpp"

//----------------------------------------------------------------------
GE_SPH::GE_SPH(RTPS *psfr, int n)
{
    num = n;

    //for reading back different values from the kernel

    //store the particle system framework
    ps = psfr;

	radixSort = 0;

	// density, force, pos, vel, surf tension, color
	nb_vars = 10;  // for array structure in OpenCL
	nb_el = n;
	printf("1 nb_el= %d\n", nb_el);

	//gordon_parameters(); // code does not work
	ian_parameters(); // Ian's code does work (in branch rtps)

	setupArrays(); // From GE structures

	GE_SPHParams& params = *(cl_params->getHostPtr());

	printf("=========================================\n");
	printf("Spacing =  particle_radius\n");
	printf("Smoothing distance = 2 * particle_radius to make sure we have particle-particle influence\n");
	printf("\n");
	printf("=========================================\n");
	params.print();
	sph_settings.print();
	cl_GridParams->getHostPtr()->print();
	cl_GridParamsScaled->getHostPtr()->print();
	cl_FluidParams->getHostPtr()->print();
	printf("=========================================\n");

	//exit(0);

	setupTimers();
	initializeData();
}

//----------------------------------------------------------------------
GE_SPH::~GE_SPH()
{
	//printf("**** inside GE_SPH destructor ****\n");

    if(pos_vbo && managed)
    {
        glBindBuffer(1, pos_vbo);
        glDeleteBuffers(1, (GLuint*)&pos_vbo);
        pos_vbo = 0;
    }
    if(col_vbo && managed)
    {
        glBindBuffer(1, col_vbo);
        glDeleteBuffers(1, (GLuint*)&col_vbo);
        col_vbo = 0;
    }

	#if 1
	delete 	cl_vars_sorted;
	delete 	cl_vars_unsorted;
	delete 	cl_cells; // positions in Ian code
	delete 	cl_cell_indices_start;
	delete 	cl_cell_indices_end;
	delete 	cl_vars_sort_indices;
	delete 	cl_sort_hashes;
	delete 	cl_sort_indices;
	delete 	cl_unsort;
	delete 	cl_sort;
	delete  cl_GridParams;
	delete  cl_FluidParams;
	delete  cl_params;
	delete	clf_debug;  //just for debugging cl files
	delete	cli_debug;  //just for debugging cl files

    delete cl_velocity;
    delete cl_density;
	delete cl_position;
	delete cl_color;
	delete cl_force;
	#endif

	if (radixSort) delete radixSort;

	//printf("***** exit GE_SPH destructor **** \n");
}

//----------------------------------------------------------------------
void GE_SPH::update()
{
	static int count=0;
	printf("==== count= %d\n", count);

	ts_cl[TI_UPDATE]->start(); // OK

	//printf("update: nb_el= %d\n", nb_el);

    //call kernels
    //TODO: add timings
#ifdef CPU
	computeOnCPU();
	#if 0
	for (int i=0; i < 10; i++) {
		float4 p = positions[i];
		printf("pos[%d] = %f, %f, %f\n", p.x, p.y, p.z);
	}
	exit(0);
	#endif
    glBindBuffer(GL_ARRAY_BUFFER, pos_vbo);
    glBufferData(GL_ARRAY_BUFFER, num * sizeof(float4), &positions[0], GL_DYNAMIC_DRAW);
#endif

	if (count == 0) {
		;//printGPUDiagnostics(count);
	}

#ifdef GPU
	int nb_sub_iter = 5;
	computeOnGPU(nb_sub_iter);
	if (count % 10 == 0) computeTimeStep();
#endif

// density is not uniform!

    /*
    std::vector<float4> ftest = cl_force->copyToHost(100);
    for(int i = 0; i < 100; i++)
    {
        if(ftest[i].z != 0.0)
            printf("force: %f %f %f  \n", ftest[i].x, ftest[i].y, ftest[i].z);
    }
    printf("execute!\n");
    */

	ts_cl[TI_UPDATE]->end(); // OK

	count++;
	//printGPUDiagnostics(count);
	//printf("count= %d\n", count);

#ifdef GPU
	if (count%20 == 0) {
		//count = 0;
		printf("ITERATION: %d\n", count*nb_sub_iter);
		printf("count= %d, nb_sub_iter= %d\n", count, nb_sub_iter);
		GE::Time::printAll();
	}
#endif

	//exit(0);
}
//----------------------------------------------------------------------
void GE_SPH::setupArrays()
{
	GE_SPHParams& params = *(cl_params->getHostPtr());
	printf("params: scale: %f\n", params.simulation_scale);

	// only for my test routines: sort, hash, datastructures
	//printf("setupArrays, nb_el= %d\n", nb_el); exit(0);

	cl_cells = new BufferGE<float4>(ps->cli, nb_el);
	for (int i=0; i < nb_el; i++) {
		(*cl_cells)[i] = positions[i];
	}
	cl_cells->copyToDevice();

// Need an assign operator (no memory allocation)

	printf("allocate BufferGE<GridParams>\n");
	printf("sizeof(GridParams): %d\n", sizeof(GridParams));
	cl_GridParams = new BufferGE<GridParams>(ps->cli, 1); // destroys ...
	cl_GridParamsScaled = new BufferGE<GridParamsScaled>(ps->cli, 1); // destroys ...

	GridParams& gp = *(cl_GridParams->getHostPtr());
	printf("gp(host)= %ld\n", (long) &gp);

	GridParamsScaled& gps = *(cl_GridParamsScaled->getHostPtr());

	gp.grid_min = grid.getMin();
	gp.grid_max = grid.getMax();
	gp.bnd_min  = grid.getBndMin();
	gp.bnd_max  = grid.getBndMax();
	gp.grid_res = grid.getRes();
	gp.grid_size = grid.getSize();
	printf("*** grid_size= %d\n", grid_size);
	gp.grid_delta = grid.getDelta();
	gp.numParticles = nb_el;

	gp.grid_inv_delta.x = 1. / gp.grid_delta.x;
	gp.grid_inv_delta.y = 1. / gp.grid_delta.y;
	gp.grid_inv_delta.z = 1. / gp.grid_delta.z;
	gp.grid_inv_delta.w = 1.;
	gp.grid_inv_delta.print("inv delta");
	gp.nb_vars = nb_vars;

	cl_GridParams->copyToDevice();

	gp.grid_size.print("grid size (domain dimensions)"); // domain dimensions
	gp.grid_delta.print("grid delta (cell size)"); // cell size
	gp.grid_min.print("grid min");
	gp.grid_max.print("grid max");
	gp.grid_res.print("grid res (nb points)"); // number of points
	gp.grid_delta.print("grid delta");
	gp.grid_inv_delta.print("grid inv delta");

    float ss = params.simulation_scale;
	//ss = 1.0; // TEMPORARY UNTIL I KNOW WHAT I AM DOING
	printf("ss= %f\n", ss);
//	exit(0);

	gps.bnd_min  = gp.bnd_min * ss;
	gps.bnd_max  = gp.bnd_max * ss;
	gps.grid_size = gp.grid_size * ss;
	gps.grid_delta = gp.grid_delta * ss;
	gps.grid_min = gp.grid_min * ss;
	gps.grid_max = gp.grid_max * ss;
	gps.grid_res = gp.grid_res;
	gps.grid_inv_delta = gp.grid_inv_delta / ss;
	gps.grid_inv_delta.w = 1.0;
	gps.nb_vars = nb_vars;
	gps.numParticles = nb_el;

	gp.print();
	gps.print();
	//printf("gps.grid_size.x = %f\n", gps.grid_size.x);
	//exit(0);

	cl_GridParamsScaled->copyToDevice();

	cl_vars_unsorted = new BufferGE<float4>(ps->cli, nb_el*nb_vars);
	cl_vars_sorted   = new BufferGE<float4>(ps->cli, nb_el*nb_vars);
	cl_sort_indices  = new BufferGE<int>(ps->cli, nb_el);
	cl_sort_hashes   = new BufferGE<int>(ps->cli, nb_el);

	// ERROR
	int grid_size = nb_cells.x * nb_cells.y * nb_cells.z;
	printf("grid_size= %d\n", grid_size);
	grid_size = 10000;
	if (grid_size > 10000000) {
		printf("nb cells: %d, %d, %d\n", nb_cells.x, nb_cells.y, nb_cells.z);
		printf("grid_size too large (> 10000000)\n");
		exit(0);
	}

	// Size is the grid size. That is a problem since the number of
	// occupied cells could be much less than the number of grid elements. 
	cl_cell_indices_start = new BufferGE<int>(ps->cli, grid_size);
	cl_cell_indices_end   = new BufferGE<int>(ps->cli, grid_size);

	// For bitonic sort. Remove when bitonic sort no longer used
	// Currently, there is an error in the Radix Sort (just run both
	// sorts and compare outputs visually
	cl_sort_output_hashes = new BufferGE<int>(ps->cli, nb_el);
	cl_sort_output_indices = new BufferGE<int>(ps->cli, nb_el);


	clf_debug = new BufferGE<float4>(ps->cli, nb_el);
	cli_debug = new BufferGE<int4>(ps->cli, nb_el);


	int nb_floats = nb_vars*nb_el;
	// WHY NOT cl_float4 (in CL/cl_platform.h)
	float4 f;
	float4 zero;

	zero.x = zero.y = zero.z = 0.0;
	zero.w = 1.;


	// SETUP FLUID PARAMETERS
	// cell width is one diameter of particle, which imlies 27 neighbor searches
	#if 1
	cl_FluidParams = new BufferGE<FluidParams>(ps->cli, 1);
	FluidParams& fp = *cl_FluidParams->getHostPtr();;
	float radius = sph_settings.particle_radius;
	fp.smoothing_length = sph_settings.smoothing_distance; // SPH radius
	fp.scale_to_simulation = params.simulation_scale; // overall scaling factor
	//fp.mass = 1.0; // mass of single particle (MIGHT HAVE TO BE CHANGED)
	fp.friction_coef = 0.1;
	fp.restitution_coef = 0.9;
	fp.damping = 0.9;
	fp.shear = 0.9;
	fp.attraction = 0.9;
	fp.spring = 0.5;
	fp.gravity = -9.8; // -9.8 m/sec^2
	fp.choice = 0; // compute density
	cl_FluidParams->copyToDevice();
	#endif

    printf("done with setup arrays\n");
}
//----------------------------------------------------------------------
void GE_SPH::checkDensity()
{
#if 0
        //test density
		// Density checks should be in Density.cpp I believe (perhaps not)
        std::vector<float> dens = cl_density->copyToHost(num);
        float dens_sum = 0.0f;
        for(int j = 0; j < num; j++)
        {
            dens_sum += dens[j];
        }
        printf("summed density: %f\n", dens_sum);
#endif
}
//----------------------------------------------------------------------
void GE_SPH::computeOnGPU(int nb_sub_iter)
{
    glFinish();
    cl_position->acquire();
    cl_color->acquire();
    
    for(int i=0; i < nb_sub_iter; i++)
    {
		// ***** Create HASH ****
		hash();
		//exit(0);

		// **** Sort arrays ****
		// only power of 2 number of particles
		// RADIX DOES NOT WORK. Bitonic works. 
		//radix_sort();
		bitonic_sort();

		// **** Reorder pos, vel
		buildDataStructures(); 

		#if 1
		// ***** DENSITY UPDATE *****
		neighborSearch(0); //density
		//exit(0);


		// ***** DENSITY DENOMINATOR *****
		//   *** DENSITY NORMALIZATION ***
		//neighborSearch(3); 

		// ***** COLOR GRADIENT *****
		//neighborSearch(2); 

		// ***** PRESSURE UPDATE *****
		neighborSearch(1); //pressure
		#endif

		// ***** WALL COLLISIONS *****
		computeCollisionWall();

        // ***** TIME UPDATE *****
		//computeEuler();
		computeLeapfrog();
	}

	// *** OUTPUT PHYSICAL VARIABLES FROM THE GPU
	//exit(0);

    cl_position->release();
    cl_color->release();
}
//----------------------------------------------------------------------
void GE_SPH::computeOnCPU()
{
    //cpuDensity();
    //cpuPressure();
    //cpuEuler();

	FluidParams* fp = cl_FluidParams->getHostPtr();;
	GridParamsScaled* gp = (cl_GridParamsScaled->getHostPtr());
	GE_SPHParams* sphp = (cl_params->getHostPtr());

	float4* density = cl_vars_unsorted->getHostPtr() + 0*nb_el;
	float4* pos     = cl_vars_unsorted->getHostPtr() + 1*nb_el;
	float4* vel     = cl_vars_unsorted->getHostPtr() + 2*nb_el;
	float4* force   = cl_vars_unsorted->getHostPtr() + 3*nb_el;

	//fp->print();
	//gp->print();
	//sphp->print();

	float4 stress;

	float h = sphp->smoothing_distance;
	float drest = sphp->rest_distance;
	float scale = sphp->simulation_scale;
	double pi = acos(-1.);
	printf("h= %f, drest= %f\n", h, drest);
	//printf("particle_spacing= %f\n", particle_spacing);
	//printf("nb_part= %d\n", (int) (4.*3.14*pow(h/drest,3.)));
	//printf("nb_part= %d\n", (int) (4.*pi/3. * pow(h/drest,3.)));
	//printf("nb_part= %d\n", (int) (hvol/particle_cell_vol));
	//sphp->print();
	//exit(0);

	for (int i=0; i < nb_el; i++) {
		float rho = 0.; 
		float4 xi = pos[i];

		int cnt = 0;

		#if 0
		//float sd = setSmoothingDist(50, drest);
		//for (int n=1; n < 20; n++) {
		for (int n=2; n < 5; n++) {
			fixedSphere(n);
		}
		exit(0);
		#endif

		for (int j=0; j < nb_el; j++) {
			float4 xj = pos[j];
	
			float4 r = xj-xi;
			float4 rs = r*scale;
			rs.w = 0.0;
			float rlen = rs.length();

			if (rlen > h) continue;
			//printf("rlen= %f\n", rlen);
			cnt++;
	
			// DENSITY
    		float Wij = Wpoly6(rs, sphp->smoothing_distance, sphp);

			#if 0
			float4 drs = float4(h/29., 0., 0., 0.);
			float vol = drest*drest*drest;
			for (int i=0; i < 30; i++) {
				float w = Wpoly6(i*drs, sphp->smoothing_distance, sphp);
				printf("w= %f, w*vol= %e\n", w, w*vol);
				printf("vol= %e\n", vol);
			}
			//----------------
			exit(0);
			#endif

			#if 0
			float4 drs = float4(h/29., 0., 0., 0.);
			float vol = drest*drest*drest;
			for (int i=0; i < 30; i++) {
				float w = Wpoly6(i*drs, sphp->smoothing_distance, sphp);
				printf("w= %f, w*vol= %e\n", w, w*vol);
				printf("vol= %e\n", vol);
			}
			//----------------
			exit(0);
			#endif

			//printf("Wij*mass= %f, h= %f, mass= %f\n", Wij*sphp->mass, sphp->smoothing_distance, sphp->mass);
			rho += sphp->mass*Wij;
		}
		//printf("dens(%d) cnt= %d, rho= %f\n", i, cnt, rho);
		density[i].x = rho;
		//printf("=================================\n");
	}

	//exit(0);

	//--------------
	for (int i=0; i < nb_el; i++) {
		int cnt = 0;
		float4 xi = pos[i];
		stress = float4(0.,0.,0.,0.);
		float di = density[i].x;  // should not repeat di=

		for (int j=0; j < nb_el; j++) {
			if (j == i) continue;
			float4 xj = pos[j];
			float4 r = xj-xi;
			float4 rs = r*scale;
			rs.w = 0.0;
			float rlen = rs.length();


			if (rlen > h) continue;
			//printf("rlen,h= %f, %f\n", rlen,h);
			cnt++;


			float dWijdr = Wspiky_dr(rlen, h, sphp);

			float dj = density[j].x;
			//printf("di,dj= %f, %f\n", di, dj);
			//printf("rlen= %f, h= %f\n", rlen, h);
			//printf("fact= %f\n", (h-rlen)*(h-rlen)/rlen);

			//form simple SPH in Krog's thesis

			float rest_density = 1000.f;
			float Pi = sphp->K*(di - rest_density);
			float Pj = sphp->K*(dj - rest_density);

			//printf("dWijdr= %f\n", dWijdr);

			float kern = -dWijdr * (Pi + Pj)*0.5 / (di*dj);
			//float kern = - (Pi + Pj)*0.5 / (di*dj);

			//printf("Pi,Pj= %f, %f\n", Pi, Pj);
			//rs.print("rs");
			//printf("kern= %f\n", kern);

			float4 ss = kern*rs;
			ss.w = 0.0;
			//ss.print("ss");

			stress = stress + ss;
			//stress.print("stress");
			//printf("-----\n");
		}
		force[i] = stress*sphp->mass;
		//force[i].print("force");
		//printf("visc(%d), cnt= %d\n", i, cnt);
		//printf("======\n");
	}

	int pt = 200;
	printf("pos[pt] = %f, %f, %f\n", pos[pt].x, pos[pt].y, pos[pt].z);
	collisionWallCPU();
	eulerOnCPU();


}
//----------------------------------------------------------------------
float4 GE_SPH::eulerOnCPU()
{
#if 0
void ge_euler(
		int* sort_indices,  
		float4* vars_unsorted, 
		float4* vars_sorted, 
		// should not be required since part of vars_unsorted
		float4* positions,  // for VBO 
		struct SPHParams* params, 
		float dt)
{
#endif

	FluidParams* fp = cl_FluidParams->getHostPtr();;
	GridParams* gp = (cl_GridParams->getHostPtr());
	GE_SPHParams* sphp = (cl_params->getHostPtr());

	float scale = sphp->simulation_scale;
	float h = sphp->smoothing_distance;
	float scale_inv = 1. / scale;
	printf("scal_inv= %f\nn", scale_inv);

	float4* density = cl_vars_unsorted->getHostPtr() + 0*nb_el;
	float4* pos     = cl_vars_unsorted->getHostPtr() + 1*nb_el;
	float4* vel     = cl_vars_unsorted->getHostPtr() + 2*nb_el;
	float4* force   = cl_vars_unsorted->getHostPtr() + 3*nb_el;

	int pt = 200;

	for (int i=0; i < nb_el; i++) {
    	float4 p = pos[i]*scale;
    	float4 v = vel[i];
    	float4 f = force[i];

		if (i == pt) {
			p.print("p");
			v.print("v");
			f.print("f");
		}

    	//external force is gravity
    	//f.z += -9.8f * 0.707;
		//f.x += -9.8f * 0.707;

    	f.z += -9.8f;

		// REMOVE FOR DEBUGGING
		// THIS IS REALLY A FORCE, NO?
		f.w = 0.0f;
    	float speed = f.length();
    	if(speed > 600.0f) //velocity limit, need to pass in as struct
    	//if(speed > 4.f) //velocity limit, need to pass in as struct
    	{
        	f = f * (600.0f/speed);
    	}

		//float dtt = dt / params->simulation_scale;
		float dtt = sphp->dt;

		float4 dtf= dtt*f;
    	v = v + dtf;  //    / params->simulation_scale;
		float4 dtv = dtt*v;
    	p = p + dtv; // params->simulation_scale;
    	p.w = 1.0f; //just in case

		pos[i] = p * scale_inv;
		vel[i] = v;
		positions[i] = pos[i];
	}
	printf("pos[pt] = %f, %f, %f\n", pos[pt].x, pos[pt].y, pos[pt].z);
	//exit(0);
}
//----------------------------------------------------------------------
float4 GE_SPH::calculateRepulsionForce(
      float4 normal, 
	  float4 vel, 
	  float boundary_stiffness, 
	  float boundary_dampening, 
	  float boundary_distance)
{
    vel.w = 0.0f;  // Removed influence of 4th component of velocity (does not exist)
	float dot_prod = normal.x*vel.x + normal.y*vel.y + normal.z*vel.z;
    float4 repulsion_force = (boundary_stiffness * boundary_distance 
	     - boundary_dampening * dot_prod)*normal;
	repulsion_force.w = 0.f;
    return repulsion_force;
}

//----------------------------------------------------------------------
void GE_SPH::collisionWallCPU()
{
	FluidParams* fp = cl_FluidParams->getHostPtr();;
	GridParamsScaled* gp = (cl_GridParamsScaled->getHostPtr());
	GE_SPHParams* sphp = (cl_params->getHostPtr());

	float4* density = cl_vars_unsorted->getHostPtr() + 0*nb_el;
	float4* pos     = cl_vars_unsorted->getHostPtr() + 1*nb_el;
	float4* vel     = cl_vars_unsorted->getHostPtr() + 2*nb_el;
	float4* force   = cl_vars_unsorted->getHostPtr() + 3*nb_el;

	float h = sphp->smoothing_distance;
	float scale = sphp->simulation_scale;

    //unsigned int i = get_global_id(0);
	//int num = get_global_size(0);
	//int nb_vars = gp->nb_vars;

	for (int i=0; i < nb_el; i++) {

    float4 p = scale*pos[i]; //  pos[i];
    float4 v = vel[i]; //  vel[i];
    float4 r_f = float4(0.f, 0.f, 0.f, 0.f);

    //bottom wall
    float diff = sphp->boundary_distance - (p.z - gp->grid_min.z);
    if (diff > sphp->EPSILON)
    {
		// normal points into the domain
        float4 normal = float4(0.0f, 0.0f, 1.0f, 0.0f);
		//if (dot(normal,v) < 0) {
        r_f += calculateRepulsionForce(normal, v, sphp->boundary_stiffness, sphp->boundary_dampening, diff);
		//}
    }

    //Y walls
    diff = sphp->boundary_distance - (p.y - gp->grid_min.y);
    if (diff > sphp->EPSILON)
    {
        float4 normal = float4(0.0f, 1.0f, 0.0f, 0.0f);
        r_f += calculateRepulsionForce(normal, v, sphp->boundary_stiffness, sphp->boundary_dampening, diff);
    }
    diff = sphp->boundary_distance - (gp->grid_max.y - p.y);
    if (diff > sphp->EPSILON)
    {
        float4 normal = float4(0.0f, -1.0f, 0.0f, 0.0f);
        r_f += calculateRepulsionForce(normal, v, sphp->boundary_stiffness, sphp->boundary_dampening, diff);
    }

    //X walls
    diff = sphp->boundary_distance - (p.x - gp->grid_min.x);
    if (diff > sphp->EPSILON)
    {
        float4 normal = float4(1.0f, 0.0f, 0.0f, 0.0f);
        r_f += calculateRepulsionForce(normal, v, sphp->boundary_stiffness, sphp->boundary_dampening, diff);
    }
    diff = sphp->boundary_distance - (gp->grid_max.x - p.x);
    if (diff > sphp->EPSILON)
    {
        float4 normal = float4(-1.0f, 0.0f, 0.0f, 0.0f);
        r_f += calculateRepulsionForce(normal, v, sphp->boundary_stiffness, sphp->boundary_dampening, diff);
    }

    //TODO add friction forces

	force[i] += r_f;   //sorted force
	}
}
//----------------------------------------------------------------------
float GE_SPH::computeTimeStep()
{
	cl_vars_unsorted->copyToHost();
	float4* vel = cl_vars_unsorted->getHostPtr() + 2*nb_el;
	GridParams* gp = cl_GridParams->getHostPtr();
	GE_SPHParams* params = cl_params->getHostPtr();

	float velmax = 0.;
	float vel2;
	for (int i=0; i < nb_el; i++) {
		vel2 = vel[i].x*vel[i].x+vel[i].y*vel[i].y+vel[i].z*vel[i].z;
		velmax = vel2 > velmax ? vel2 : velmax;
	}
	velmax = sqrt(velmax);
	float soundSpeed = sqrt(params->K);
	float dt = 0.5 * gp->grid_delta.x / (velmax+soundSpeed);

	printf("velmax = %f\n", velmax);
	printf("time step limit: %f\n", dt);
	return dt;
}
//----------------------------------------------------------------------
void GE_SPH::printGPUDiagnostics(int count)
{
		float4* pos;

		#if 0
		//Update position array (should be done on GPU)
		cl_vars_unsorted->copyToHost();
		pos  = cl_vars_unsorted->getHostPtr() + 1*nb_el;
		for (int i=0; i < nb_el; i++) {
			//printf("i= %d\n", i);
			positions[i] = pos[i];
		}
		#endif

		#if 1
		//  print out density
		cl_vars_unsorted->copyToHost();
		cl_vars_sorted->copyToHost();

		float4* density = cl_vars_unsorted->getHostPtr() + 0*nb_el;
		pos     = cl_vars_unsorted->getHostPtr() + 1*nb_el;
		float4* vel     = cl_vars_unsorted->getHostPtr() + 2*nb_el;
		float4* force   = cl_vars_unsorted->getHostPtr() + 3*nb_el;

		float4* density1 = cl_vars_sorted->getHostPtr() + 0*nb_el;
		float4* pos1     = cl_vars_sorted->getHostPtr() + 1*nb_el;
		float4* vel1     = cl_vars_sorted->getHostPtr() + 2*nb_el;
		float4* force1   = cl_vars_sorted->getHostPtr() + 3*nb_el;

		// identify particles that exited the domain
	    GridParams& gp = *(cl_GridParams->getHostPtr());
		float4 mn = gp.grid_min;
		float4 mx = gp.grid_max;
		mn.print("**** grid min");
		mx.print("**** grid max");
	
		//float fmin = computeMax(force, int nb_el);
		//float fmax = computeMin(force, int nb_el);
		//printf("force min/max = %f, %f\n", fmin, fmax);

    	float bd = sph_settings.boundary_distance;

		for (int i=0; i < nb_el; i++) {
			float4 p = pos[i];
			float4 f = force[i];
			if (p.z > 6. && force[i].z > 0) {
				printf("----------------\n");
				printf("(%d) pos: %f, %f, %f, rho= %f\n", count, p.x, p.y, p.z, p.w);
				printf("(%d) force: %f, %f, %f\n", count, f.x, f.y, f.z);
				//printf("p.z > 5\n");
			//	exit(0);
			}
			#if 0
			if ((p.x-bd) < mn.x || (p.x+bd) > mx.x) {
				printf("particle i= %d, ", i);
				printf("outside x: %f, force: %f\n", p.x, f.x);
			}
			if ((p.y-bd) < mn.y || (p.y+bd) > mx.y) {
				printf("particle i= %d, ", i);
				printf("outside y: %f, force: %f\n", p.y, f.y);
			}
			if ((p.z-bd) < mn.z || (p.z+bd) > mx.z) {
				printf("particle i= %d, ", i);
				printf("outside z: %f, force: %f\n", p.z, f.z);
			}
			#endif
		}
		return;

		#if 0
		for (int i=1000; i < 1500; i++) {
		//for (int i=0; i < 10; i++) {
			printf("=== i= %d ==========\n", i);
			//printf("dens[%d]= %f, sorted den: %f\n", i, density[i].x, density1[i].x);
			pos[i].print("un pos");
			vel[i].print("un vel");
			force[i].print("un force");
			//density[i].print("un density");
			//pos1[i].print("so pos1");
			//vel1[i].print("so vel1");
			//force1[i].print("so force1");
		} 
		#endif


		//exit(0);
		#endif
}
//----------------------------------------------------------------------
float GE_SPH::computeMax(float* arr, int nb)
{
	float mx = -1.e+50;

	for (int i=0; i < nb; i++) {
		mx = (arr[i] > mx) ? arr[i] : mx;
	}

	return mx;
}
//----------------------------------------------------------------------
float GE_SPH::computeMin(float* arr, int nb)
{
	float mn = 1.e+50;

	for (int i=0; i < nb; i++) {
		mn = (arr[i] < mn) ? arr[i] : mn;
	}

	return mn;
}
//----------------------------------------------------------------------
float GE_SPH::setSmoothingDist(int nb_part, float rest_dist)
{
	// given nb particles in a sphere, what is the radius of the 
	// sphere that includes at least that number of particles? Do this heuristically
	// Center the sphere at (0,0,0)
	// Assume first, a unit rest distance. I'll scale the radius at the end

	printf("---\n");
	nb_part = 40;
	int nb_part_side = ceil(pow((double) nb_part, 1./3.));
	int ns = 2*nb_part_side; // much more than required

	rest_dist = 1.0; // rescale later

	double pi = acos(-1.);
	double coef = 4.*pi/3.;
	double cell_volume = pow(rest_dist, 3.);
	double sphere_vol = cell_volume * nb_part;
	double radius = pow(sphere_vol/coef, 1./3.);
	printf("radius= %f\n", radius);

	double radius2 = radius*radius;
	int cnt;

	for (int i=0; i < 10; i++) {
		radius *= 1.1;
		cnt = countPoints(radius, ns);
		printf("cnt= %d\n", cnt);
	}

	printf("desired nb part: %d\n", nb_part);
	sphere_vol = (float) coef*ns*ns*ns;
	float discrete_vol = (float) cnt;
	printf("sphere_vol= %f\n", sphere_vol);
	printf("discrete_vol= %f\n", discrete_vol);
	exit(0);
}
//----------------------------------------------------------------------
// Alternatively, consider a box of size [-1,1]^3 and a sphere of radius 1.
// As the discretization is refined, how many points lie in the sphere, 
// and how accurately is sum_i dx^3 represent the sphere, and the same
// for the various W functions. 

void GE_SPH::fixedSphere(int nxh)
{
	int nx = 2.*nxh+1;  // nx: nb of points on a side of circumscribed cube
	printf("===== enter fixedSphere, nx= %d\n", nx);


	GE_SPHParams* sphp = (cl_params->getHostPtr());
	float dx = 2./(nx-1);
	// sphere radius = 1.;
	double pi = acos(-1.);
	double coef = 4.*pi/3.;
	double exact_sphere_vol = coef; // unit radius


	int cnt = 0;
	float h = 1.;
	float volPoly6 = 0.;
	float volSpike = 0.;
	float volVisc = 0.;

	for (int k=0; k < nx; k++) {
		double z = -1.+k*dx;
	for (int j=0; j < nx; j++) {
		double y = -1.+j*dx;
	for (int i=0; i < nx; i++) {
		double x = -1.+i*dx;
		double r2 = x*x+y*y+z*z;
		double len = sqrt(r2);
		if (r2 > 1.) {
			continue;
		} else {
		float4 ff = float4(x,y,z,1.);
		//ff.print("ff");
		//float vv = Wpoly6(ff, h, sphp);
		//printf("vv= %f\n", vv);
		volPoly6 += Wpoly6(ff, h, sphp);
		volSpike += Wspiky(len, h, sphp);
		if (len > 1.e-8) {
			volVisc += Wvisc(len, h, sphp);
		}
		cnt++;
		}
	}}}
	printf("cnt= %d\n", cnt);

	volPoly6 *= coef*dx*dx*dx;
	volSpike *= coef*dx*dx*dx;
	volVisc  *= coef*dx*dx*dx;

	//printf("vol= %f, vol ratio: %f\n", vol, exact_sphere_vol/vol);

	float WvolPoly6_rel_error= (exact_sphere_vol-volPoly6)/exact_sphere_vol;
	float WvolSpike_rel_error= (exact_sphere_vol-volSpike)/exact_sphere_vol;
	float WvolVisc_rel_error=  (exact_sphere_vol-volVisc) /exact_sphere_vol;

	double computed_sphere_vol = dx*dx*dx*cnt;
	printf("volPoly6 error: %f\n", WvolPoly6_rel_error);
	printf("volSpike error: %f\n", WvolSpike_rel_error);
	printf("volVisc  error: %f\n", WvolVisc_rel_error);

	// compute error based on wpoly functions. 
}
//----------------------------------------------------------------------
int GE_SPH::countPoints(double radius, int box_size)
{
	double radius2 = radius*radius;
	int cnt = 0;
	int ns = box_size;

	for (int k=-ns-1; k <= ns+1; k++) {
	for (int j=-ns-1; j <= ns+1; j++) {
	for (int i=-ns-1; i <= ns+1; i++) {
		double rad2 = i*i+j*j+k*k;
		if (rad2 < radius2) cnt++;
	}}}
	return cnt;
}
//----------------------------------------------------------------------
int GE_SPH::setupTimers()
{
	int print_freq = 20000;
	int time_offset = 5;

	ts_cl[TI_HASH]     = new GE::Time("hash",       time_offset, print_freq);
	ts_cl[TI_BUILD]    = new GE::Time("build",      time_offset, print_freq);
	ts_cl[TI_NEIGH]    = new GE::Time("neigh",      time_offset, print_freq);
	ts_cl[TI_DENS]     = new GE::Time("density",    time_offset, print_freq);
	ts_cl[TI_PRES]     = new GE::Time("pressure",   time_offset, print_freq);
	ts_cl[TI_COL]      = new GE::Time("color",      time_offset, print_freq);
	ts_cl[TI_COL_NORM] = new GE::Time("color_norm", time_offset, print_freq);
	ts_cl[TI_VISC]     = new GE::Time("viscosity",  time_offset, print_freq);
	ts_cl[TI_EULER]    = new GE::Time("euler",      time_offset, print_freq);
	ts_cl[TI_LEAPFROG] = new GE::Time("leapfrog",   time_offset, print_freq);
	ts_cl[TI_UPDATE]   = new GE::Time("update",     time_offset, print_freq);
	ts_cl[TI_COLLISION_WALL] 
			       = new GE::Time("collision wall", time_offset, print_freq);
	ts_cl[TI_RADIX_SORT]   
	               = new GE::Time("radix sort",     time_offset, print_freq);
	ts_cl[TI_BITONIC_SORT] 
				   = new GE::Time("bitonic sort",   time_offset, print_freq);
}
//----------------------------------------------------------------------
void GE_SPH::initializeData()
{
	printf("4 nb_el= %d\n", nb_el);

	// copy pos, vel, dens into vars_unsorted()
	// COULD DO THIS ON GPU
	float4* vars = cl_vars_unsorted->getHostPtr();
	BufferGE<float4>& un = *cl_vars_unsorted;
	BufferGE<float4>& so = *cl_vars_sorted;

	float* unf = (float*) un.getHostPtr();
	float* sof = (float*) so.getHostPtr();

	for (int i=0; i < nb_el*nb_vars; i++) {
		unf[i] = 0.0;
		sof[i] = 0.0;
	}

	for (int i=0; i < nb_el; i++) {
		//vars[i+DENS*num] = densities[i];
		// PROBLEM: density is float, but vars_unsorted is float4
		// HOW TO DEAL WITH THIS WITHOUT DOUBLING MEMORY ACCESS in 
		// buildDataStructures. 

		//printf("%d, %d, %d, %d\n", DENS, POS, VEL, FOR); exit(0);

		un(i+DENS*nb_el).x = densities[i];
		un(i+DENS*nb_el).y = 1.0; // for surface tension (always 1)
		un(i+POS*nb_el) = positions[i];
		un(i+VEL*nb_el) = velocities[i];
		un(i+FOR*nb_el) = forces[i];

		// SHOULD NOT BE REQUIRED
		so(i+DENS*nb_el).x = densities[i];
		so(i+DENS*nb_el).y = 1.0;  // for surface tension (always 1)
		so(i+POS*nb_el) = positions[i];
		so(i+VEL*nb_el) = velocities[i];
		so(i+FOR*nb_el) = forces[i];
	}

	cl_vars_unsorted->copyToDevice();
	cl_vars_sorted->copyToDevice(); // should not be required
}
//----------------------------------------------------------------------
}

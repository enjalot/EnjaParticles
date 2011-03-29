
#include <GL/glew.h>
#include <math.h>

#include "System.h"
#include "SPH.h"
//#include "../domain/UniformGrid.h"
#include "Domain.h"
#include "IV.h"

#include "common/Hose.h"


//for random
#include<time.h>

namespace rtps
{


    SPH::SPH(RTPS *psfr, int n)
    {
        //store the particle system framework
        ps = psfr;

        max_num = n;
        num = 0;
        nb_var = 10;

        positions.resize(max_num);
        colors.resize(max_num);
        forces.resize(max_num);
        velocities.resize(max_num);
        veleval.resize(max_num);
        densities.resize(max_num);
        xsphs.resize(max_num);

        //seed random
        srand ( time(NULL) );

        grid = ps->settings.grid;

        sphsettings = new SPHSettings(grid, max_num);
        //sphsettings->printSettings();
        sphsettings->updateSPHP(sphp);
        spacing = sphsettings->GetSettingAs<float>("Spacing");

        //SPH settings depend on number of particles used
        //calculateSPHSettings();
        //set up the grid
        setupDomain();

        integrator = LEAPFROG;
        //integrator = EULER;

        //*** end Initialization

        setupTimers();

#ifdef CPU
        printf("RUNNING ON THE CPU\n");
#endif
#ifdef GPU
        printf("RUNNING ON THE GPU\n");

        //setup the sorted and unsorted arrays
        prepareSorted();


        std::string cl_includes(SPH_CL_SOURCE_DIR);
        ps->cli->setIncludeDir(cl_includes);

        loadCollision_wall();
        loadCollision_tri();

        //could generalize this to other integration methods later (leap frog, RK4)
        if (integrator == LEAPFROG)
        {
            loadLeapFrog();
        }
        else if (integrator == EULER)
        {
            loadEuler();
        }

        loadScopy();

        loadPrep();
        loadHash();
        loadBitonicSort();
        loadDataStructures();
        loadNeighbors();


#endif

        // settings defaults to 0
        //renderer = new Render(pos_vbo,col_vbo,num,ps->cli, &ps->settings);
        setRenderer();

    }

    SPH::~SPH()
    {
        printf("SPH destructor\n");
        if (pos_vbo && managed)
        {
            glBindBuffer(1, pos_vbo);
            glDeleteBuffers(1, (GLuint*)&pos_vbo);
            pos_vbo = 0;
        }
        if (col_vbo && managed)
        {
            glBindBuffer(1, col_vbo);
            glDeleteBuffers(1, (GLuint*)&col_vbo);
            col_vbo = 0;
        }

    }

    void SPH::update()
    {
        //call kernels
        //TODO: add timings
#ifdef CPU
        updateCPU();
#endif
#ifdef GPU
        updateGPU();
#endif
    }

    void SPH::updateCPU()
    {
        cpuDensity();
        cpuPressure();
        cpuViscosity();
        cpuXSPH();
        cpuCollision_wall();

        if (integrator == EULER)
        {
            cpuEuler();
        }
        else if (integrator == LEAPFROG)
        {
            cpuLeapFrog();
        }
#if 0
        //printf("positions[0].z %f\n", positions[0].z);
        for (int i = 0; i < 100; i++)
        {
            //if(xsphs[i].z != 0.0)
            //printf("force: %f %f %f  \n", veleval[i].x, veleval[i].y, veleval[i].z);
            printf("force: %f %f %f  \n", xsphs[i].x, xsphs[i].y, xsphs[i].z);
            //printf("force: %f %f %f  \n", velocities[i].x, velocities[i].y, velocities[i].z);
        }
        //printf("cpu execute!\n");
#endif
        glBindBuffer(GL_ARRAY_BUFFER, pos_vbo);
        glBufferData(GL_ARRAY_BUFFER, num * sizeof(float4), &positions[0], GL_DYNAMIC_DRAW);
    }

    void SPH::updateGPU()
    {

        timers["update"]->start();
        glFinish();

        //GE
        int sub_intervals = 3;  //should be a setting
        //this should go in the loop but avoiding acquiring and releasing each sub
        //interval for all the other calls.
        //this does end up acquire/release everytime sprayHoses calls pushparticles
        //should just do try/except?
        for (int i=0; i < sub_intervals; i++)
        {
            sprayHoses();
        }

        cl_position.acquire();
        cl_color.acquire();
        //sub-intervals
        for (int i=0; i < sub_intervals; i++)
        {
            /*
            k_density.execute(num);
            k_pressure.execute(num);
            k_viscosity.execute(num);
            k_xsph.execute(num);
            */
            //printf("hash\n");
            timers["hash"]->start();
            hash();
            timers["hash"]->stop();
            //printf("bitonic_sort\n");
            timers["bitonic"]->start();
            bitonic_sort();
            timers["bitonic"]->stop();
            //printf("data structures\n");
            timers["datastructures"]->start();
            buildDataStructures(); //reorder
            timers["datastructures"]->stop();

            //printf("density\n");
            timers["density"]->start();
            neighborSearch(0);  //density
            timers["density"]->stop();
            //printf("forces\n");
            timers["force"]->start();
            neighborSearch(1);  //forces
            timers["force"]->stop();
            //exit(0);

            //printf("collision\n");
            collision();
            //printf("integrate\n");
            timers["integrate"]->start();
            integrate();
            timers["integrate"]->stop();
            //exit(0);
            //
            //Andrew's rendering emporium
            //neighborSearch(4);
        }

        cl_position.release();
        cl_color.release();

        timers["update"]->stop();

    }

    void SPH::collision()
    {
        //when implemented other collision routines can be chosen here
        timers["collision_wall"]->start();
        collide_wall();
        //k_collision_wall.execute(num, local_size);
        timers["collision_wall"]->stop();

        timers["collision_tri"]->start();
        collide_triangles();
        timers["collision_tri"]->stop();

    }

    void SPH::integrate()
    {
        if (integrator == EULER)
        {
            euler();
        }
        else if (integrator == LEAPFROG)
        {
            leapfrog();
        }

#if 0
        if (num > 0)
        {
            std::vector<float4> pos = cl_position.copyToHost(num);
            for (int i = 0; i < num; i++)
            {
                printf("pos[%d] = %f %f %f\n", i, pos[i].x, pos[i].y, pos[i].z);
            }
        }
#endif


    }

    int SPH::setupTimers()
    {
        //int print_freq = 20000;
        int print_freq = 1000; //one second
        int time_offset = 5;

        /*
        timers[TI_UPDATE]     = new GE::Time("update", time_offset, print_freq);
        timers[TI_HASH]     = new GE::Time("hash", time_offset, print_freq);
        timers[TI_BUILD]     = new GE::Time("build", time_offset, print_freq);
        timers[TI_BITONIC_SORT]     = new GE::Time("bitonic_sort", time_offset, print_freq);
        timers[TI_NEIGH]     = new GE::Time("neigh", time_offset, print_freq);
        timers[TI_DENS]     = new GE::Time("dens", time_offset, print_freq);
        timers[TI_FORCE]     = new GE::Time("force", time_offset, print_freq);
        timers[TI_COLLISION_WALL]     = new GE::Time("collision_wall", time_offset, print_freq);
        timers[TI_COLLISION_TRI]     = new GE::Time("collision_triangle", time_offset, print_freq);
        timers[TI_EULER]     = new GE::Time("euler", time_offset, print_freq);
        timers[TI_LEAPFROG]     = new GE::Time("leapfrog", time_offset, print_freq);
        */

        timers["update"] = new EB::Timer("Update loop", time_offset);
        timers["hash"] = new EB::Timer("Hash function", time_offset);
        timers["hash_gpu"] = new EB::Timer("Hash GPU kernel execution", time_offset);
        timers["datastructures"] = new EB::Timer("Datastructures kernel execution", time_offset);
        timers["bitonic"] = new EB::Timer("Bitonic Sort kernel execution", time_offset);
        //timers["neighbor"] = new EB::Timer("Neighbor Total", time_offset);
        timers["density"] = new EB::Timer("Density kernel execution", time_offset);
        timers["force"] = new EB::Timer("Force kernel execution", time_offset);
        timers["collision_wall"] = new EB::Timer("Collision wall kernel execution", time_offset);
        timers["collision_tri"] = new EB::Timer("Collision triangles kernel execution", time_offset);
        timers["integrate"] = new EB::Timer("Integration kernel execution", time_offset);
		return 0;
    }

    void SPH::printTimers()
    {
        timers.printAll();
        /*
        for (int i = 0; i < 11; i++) //switch to vector of timers and use size()
        {
            timers[i]->print();
        }
        */
        //System::printTimers();
    }

    void SPH::calculateSPHSettings()
    {
        /*!
        * The Particle Mass (and hence everything following) depends on the MAXIMUM number of particles in the system
        */

        float rho0 = 1000;                              //rest density [kg/m^3 ]
        //float mass = (128*1024.0)/max_num * .0002;    //krog's way
        float VP = 2 * .0262144 / max_num;                  //Particle Volume [ m^3 ]
        //float VP = .0262144 / 16000;                  //Particle Volume [ m^3 ]
        float mass = rho0 * VP;                         //Particle Mass [ kg ]
        //constant .87 is magic
        float rest_distance = .87 * pow(VP, 1.f/3.f);     //rest distance between particles [ m ]
        //float rest_distance = pow(VP, 1.f/3.f);     //rest distance between particles [ m ]

        float smoothing_distance = 2.0f * rest_distance;//interaction radius
        float boundary_distance = .5f * rest_distance;

        float4 dmin = grid.getBndMin();
        float4 dmax = grid.getBndMax();
        //printf("dmin: %f %f %f\n", dmin.x, dmin.y, dmin.z);
        //printf("dmax: %f %f %f\n", dmax.x, dmax.y, dmax.z);
        float domain_vol = (dmax.x - dmin.x) * (dmax.y - dmin.y) * (dmax.z - dmin.z);
        //printf("domain volume: %f\n", domain_vol);

        //ratio between particle radius in simulation coords and world coords
        float simulation_scale = pow(.5 * VP * max_num / domain_vol, 1.f/3.f); 
        //float simulation_scale = pow(VP * 16000/ domain_vol, 1.f/3.f); 

        spacing = rest_distance/ simulation_scale;

        float particle_radius = spacing;
        float pi = acos(-1.0);

        //sphp.grid_min = grid.getMin();
        //sphp.grid_max = grid.getMax();
        sphp.mass = mass;
        sphp.rest_distance = rest_distance;
        sphp.smoothing_distance = smoothing_distance;
        sphp.simulation_scale = simulation_scale;
        sphp.boundary_stiffness = 20000.0f;
        sphp.boundary_dampening = 256.0f;
        sphp.boundary_distance = boundary_distance;
        sphp.EPSILON = .00001f;
        sphp.PI = pi;
        sphp.K = 15.0f;
        sphp.num = num;
        sphp.max_num = max_num;
        //sphp.surface_threshold = 2.0 * sphp.simulation_scale; //0.01;
        sphp.viscosity = .01f;
        //sphp.viscosity = 1.0f;
        sphp.gravity = -9.8f;
        //sphp.gravity = 0.0f;
        sphp.velocity_limit = 600.0f;
        sphp.xsph_factor = .1f;

        float h = sphp.smoothing_distance;
        float h9 = pow(h,9.f);
        float h6 = pow(h,6.f);
        float h3 = pow(h,3.f);
        sphp.wpoly6_coef = 315.f/64.0f/pi/h9;
        sphp.wpoly6_d_coef = -945.f/(32.0f*pi*h9);
        sphp.wpoly6_dd_coef = -945.f/(32.0f*pi*h9);
        sphp.wspiky_coef = 15.f/pi/h6;
        sphp.wspiky_d_coef = -45.f/(pi*h6);
        sphp.wvisc_coef = 15./(2.*pi*h3);
        sphp.wvisc_d_coef = 15./(2.*pi*h3);
        sphp.wvisc_dd_coef = 45./(pi*h6);

        printf("spacing: %f\n", spacing);
        sphp.print();

    }

    void SPH::prepareSorted()
    {
#include "sph/cl_src/cl_macros.h"

        //for reading back different values from the kernel
        std::vector<float4> error_check(max_num);
        
        float4 pmax = grid_params.grid_max + grid_params.grid_size;
        //std::fill(positions.begin(), positions.end(), pmax);



        std::fill(forces.begin(), forces.end(),float4(0.0f, 0.0f, 1.0f, 0.0f));
        std::fill(velocities.begin(), velocities.end(),float4(0.0f, 0.0f, 0.0f, 0.0f));
        std::fill(veleval.begin(), veleval.end(),float4(0.0f, 0.0f, 0.0f, 0.0f));

        std::fill(densities.begin(), densities.end(), 0.0f);
        std::fill(xsphs.begin(), xsphs.end(),float4(0.0f, 0.0f, 0.0f, 0.0f));
        std::fill(error_check.begin(), error_check.end(), float4(0.0f, 0.0f, 0.0f, 0.0f));

        // VBO creation, TODO: should be abstracted to another class
        managed = true;
        printf("positions: %zd, %zd, %zd\n", positions.size(), sizeof(float4), positions.size()*sizeof(float4));
        pos_vbo = createVBO(&positions[0], positions.size()*sizeof(float4), GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW);
        printf("pos vbo: %d\n", pos_vbo);
        col_vbo = createVBO(&colors[0], colors.size()*sizeof(float4), GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW);
        printf("col vbo: %d\n", col_vbo);
        // end VBO creation

        //vbo buffers
        cl_position = Buffer<float4>(ps->cli, pos_vbo);
        cl_color = Buffer<float4>(ps->cli, col_vbo);

        //pure opencl buffers: these are deprecated
        cl_force = Buffer<float4>(ps->cli, forces);
        cl_velocity = Buffer<float4>(ps->cli, velocities);
        cl_veleval = Buffer<float4>(ps->cli, veleval);
        cl_density = Buffer<float>(ps->cli, densities);
        cl_xsph = Buffer<float4>(ps->cli, xsphs);

        //cl_error_check= Buffer<float4>(ps->cli, error_check);

        //TODO make a helper constructor for buffer to make a cl_mem from a struct
        std::vector<SPHParams> vparams(0);
        vparams.push_back(sphp);
        cl_SPHParams = Buffer<SPHParams>(ps->cli, vparams);

        //Setup Grid Parameter structs
        std::vector<GridParams> gparams(0);
        gparams.push_back(grid_params);
        cl_GridParams = Buffer<GridParams>(ps->cli, gparams);
        //scaled Grid Parameters
        std::vector<GridParams> sgparams(0);
        sgparams.push_back(grid_params_scaled);
        cl_GridParamsScaled = Buffer<GridParams>(ps->cli, sgparams);


        //setup debug arrays
        std::vector<float4> clfv(max_num);
        std::fill(clfv.begin(), clfv.end(),float4(0.0f, 0.0f, 0.0f, 0.0f));
        std::vector<int4> cliv(max_num);
        std::fill(cliv.begin(), cliv.end(),int4(0.0f, 0.0f, 0.0f, 0.0f));
        clf_debug = Buffer<float4>(ps->cli, clfv);
        cli_debug = Buffer<int4>(ps->cli, cliv);



        //sorted and unsorted arrays
        std::vector<float4> unsorted(max_num*nb_var);
        std::vector<float4> sorted(max_num*nb_var);

        std::fill(unsorted.begin(), unsorted.end(),float4(0.0f, 0.0f, 0.0f, 1.0f));
        std::fill(sorted.begin(), sorted.end(),float4(0.0f, 0.0f, 0.0f, 1.0f));
        //std::fill(unsorted.begin(), unsorted.end(), pmax);
        //std::fill(sorted.begin(), sorted.end(), pmax);



        cl_vars_unsorted = Buffer<float4>(ps->cli, unsorted);
        cl_vars_sorted = Buffer<float4>(ps->cli, sorted);

        std::vector<unsigned int> keys(max_num);
        //to get around limits of bitonic sort only handling powers of 2
#include "limits.h"
        std::fill(keys.begin(), keys.end(), INT_MAX);
        cl_sort_indices  = Buffer<unsigned int>(ps->cli, keys);
        cl_sort_hashes   = Buffer<unsigned int>(ps->cli, keys);

        // for debugging. Store neighbors of indices
        // change nb of neighbors in cl_macro.h as well
        //cl_index_neigh = Buffer<int>(ps->cli, max_num*50);

        // Size is the grid size + 1, the last index is used to signify how many particles are within bounds
        // That is a problem since the number of
        // occupied cells could be much less than the number of grid elements.
        std::vector<unsigned int> gcells(grid_params.nb_cells+1);
        int minus = 0xffffffff;
        std::fill(gcells.begin(), gcells.end(), 666);

        cl_cell_indices_start = Buffer<unsigned int>(ps->cli, gcells);
        cl_cell_indices_end   = Buffer<unsigned int>(ps->cli, gcells);
        //printf("gp.nb_points= %d\n", gp.nb_points); exit(0);



        // For bitonic sort. Remove when bitonic sort no longer used
        // Currently, there is an error in the Radix Sort (just run both
        // sorts and compare outputs visually
        cl_sort_output_hashes = Buffer<unsigned int>(ps->cli, keys);
        cl_sort_output_indices = Buffer<unsigned int>(ps->cli, keys);


        std::vector<Triangle> maxtri(2048);
        cl_triangles = Buffer<Triangle>(ps->cli, maxtri);


    }

    void SPH::setupDomain()
    {
        grid.calculateCells(sphp.smoothing_distance / sphp.simulation_scale);


        grid_params.grid_min = grid.getMin();
        grid_params.grid_max = grid.getMax();
        grid_params.bnd_min  = grid.getBndMin();
        grid_params.bnd_max  = grid.getBndMax();
        grid_params.grid_res = grid.getRes();
        grid_params.grid_size = grid.getSize();
        grid_params.grid_delta = grid.getDelta();
        grid_params.nb_cells = (int) (grid_params.grid_res.x*grid_params.grid_res.y*grid_params.grid_res.z);

        //printf("gp nb_cells: %d\n", grid_params.nb_cells);


        /*
        grid_params.grid_inv_delta.x = 1. / grid_params.grid_delta.x;
        grid_params.grid_inv_delta.y = 1. / grid_params.grid_delta.y;
        grid_params.grid_inv_delta.z = 1. / grid_params.grid_delta.z;
        grid_params.grid_inv_delta.w = 1.;
        */

        float ss = sphp.simulation_scale;

        grid_params_scaled.grid_min = grid_params.grid_min * ss;
        grid_params_scaled.grid_max = grid_params.grid_max * ss;
        grid_params_scaled.bnd_min  = grid_params.bnd_min * ss;
        grid_params_scaled.bnd_max  = grid_params.bnd_max * ss;
        grid_params_scaled.grid_res = grid_params.grid_res;
        grid_params_scaled.grid_size = grid_params.grid_size * ss;
        grid_params_scaled.grid_delta = grid_params.grid_delta / ss;
        //grid_params_scaled.nb_cells = (int) (grid_params_scaled.grid_res.x*grid_params_scaled.grid_res.y*grid_params_scaled.grid_res.z);
        grid_params_scaled.nb_cells = grid_params.nb_cells;
        //grid_params_scaled.grid_inv_delta = grid_params.grid_inv_delta / ss;
        //grid_params_scaled.grid_inv_delta.w = 1.0f;

        //grid_params.print();
        //grid_params_scaled.print();

    }

    int SPH::addBox(int nn, float4 min, float4 max, bool scaled)
    {
        float scale = 1.0f;
        if (scaled)
        {
            scale = sphp.simulation_scale;
        }
        vector<float4> rect = addRect(nn, min, max, spacing, scale);
        float4 velo(0, 0, 0, 0);
        pushParticles(rect, velo);
        return rect.size();
    }

    void SPH::addBall(int nn, float4 center, float radius, bool scaled)
    {
        float scale = 1.0f;
        if (scaled)
        {
            scale = sphp.simulation_scale;
        }
        vector<float4> sphere = addSphere(nn, center, radius, spacing, scale);
        float4 velo(0, 0, 0, 0);
        pushParticles(sphere,velo);
    }

    void SPH::addHose(int total_n, float4 center, float4 velocity, float radius)
    {
        printf("wtf for real\n");
        //in sph we just use sph spacing
        radius *= spacing;
        Hose hose = Hose(ps, total_n, center, velocity, radius, spacing);
        printf("wtf\n");
        hoses.push_back(hose);
        printf("size of hoses: %d\n", hoses.size());
    }

    void SPH::sprayHoses()
    {

        std::vector<float4> parts;
        for (int i = 0; i < hoses.size(); i++)
        {
            parts = hoses[i].spray();
            if (parts.size() > 0)
                pushParticles(parts, hoses[i].getVelocity());
        }
    }

    void SPH::testDelete()
    {

        //cut = 1;
        std::vector<float4> poss(40);
        float4 posx(10.,10.,10.,1.);
        std::fill(poss.begin(), poss.end(),posx);
        cl_vars_unsorted.copyToDevice(poss, max_num + 2);
        ps->cli->queue.finish();


    }
    void SPH::pushParticles(vector<float4> pos, float4 velo)
    {
        //cut = 1;

        int nn = pos.size();
        if (num + nn > max_num)
        {
            return;
        }
        // float rr = (rand() % 255)/255.0f;
        //float4 color(rr, 0.0f, 1.0f - rr, 1.0f);
        //printf("random: %f\n", rr);
        float4 color(1.0f,0.0f,0.0f,1.0f);

        std::vector<float4> cols(nn);
        std::vector<float4> vels(nn);

        std::fill(cols.begin(), cols.end(),color);
        //float v = .5f;
        //float v = 0.0f;
        //float4 iv = float4(v, v, -v, 0.0f);
        //float4 iv = float4(0, v, -.1, 0.0f);
        //std::fill(vels.begin(), vels.end(),iv);
        std::fill(vels.begin(), vels.end(),velo);


#ifdef GPU
        glFinish();
        cl_position.acquire();
        cl_color.acquire();

        //printf("about to prep 0\n");
        prep(0);
        //printf("done with prep 0\n");


        cl_position.copyToDevice(pos, num);
        cl_color.copyToDevice(cols, num);

        cl_color.release();
        cl_position.release();

        //2 is from cl_macros.h should probably not hardcode this number
        cl_velocity.copyToDevice(vels, num);
        //cl_vars_unsorted.copyToDevice(vels, max_num*8+num);

        sphp.num = num+nn;
        updateSPHP();

        num += nn;  //keep track of number of particles we use

        cl_position.acquire();
        //reprep the unsorted (packed) array to account for new particles
        //might need to do it conditionally if particles are added or subtracted
        printf("about to prep\n");
        prep(1);
        printf("done with prep\n");
        cl_position.release();
#else
        num += nn;  //keep track of number of particles we use
#endif
        renderer->setNum(num);
    }

    void SPH::updateSPHP()
    {
        std::vector<SPHParams> vparams(0);
        vparams.push_back(sphp);
        cl_SPHParams.copyToDevice(vparams);
    }

    void SPH::render()
    {
        renderer->render_box(grid.getBndMin(), grid.getBndMax());
        renderer->render_table(grid.getBndMin(), grid.getBndMax());
        System::render();
    }
    void SPH::setRenderer()
    {
        switch(ps->settings.getRenderType())
        {
            case RTPSettings::SPRITE_RENDER:
                renderer = new SpriteRender(pos_vbo,col_vbo,num,ps->cli, &ps->settings);
                printf("spacing for radius %f\n", spacing);
                break;
            case RTPSettings::SCREEN_SPACE_RENDER:
                //renderer = new ScreenSpaceRender();
                renderer = new SSFRender(pos_vbo,col_vbo,num,ps->cli, &ps->settings);
                break;
            case RTPSettings::RENDER:
                renderer = new Render(pos_vbo,col_vbo,num,ps->cli, &ps->settings);
                break;
            default:
                //should be an error
                renderer = new Render(pos_vbo,col_vbo,num,ps->cli, &ps->settings);
                break;
        }
        //renderer->setParticleRadius(spacing*0.5);
        renderer->setParticleRadius(spacing);
    }


} //end namespace

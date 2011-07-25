#include "GL/glew.h"
#include "RTPS.h"
#include "system/Simple.h"
#include "system/SPH.h"
#include "system/FLOCK.h"
#include "system/OUTER.h"


namespace rtps
{

    RTPS::RTPS()
    {
        cli = new CL();
        cl_managed = true;
        //settings will be the default constructor
        Init();
printf("done with constructor\n");
    }

    RTPS::RTPS(RTPSettings *s)
    {
        cli = new CL();
        cl_managed = true;
        settings = s;
        Init();
        printf("done with constructor\n");
    }

    RTPS::RTPS(RTPSettings *s, CL* _cli)
    {
        cli = _cli;
        cl_managed = false;
        settings = s;
        Init();
printf("done with constructor\n");
    }

    RTPS::~RTPS()
    {
        printf("RTPS destructor\n");
        delete system;
        if(cl_managed)
        {
            delete cli;
        }
        //delete renderer;
    }

    void RTPS::Init()
    {
        //this should already be done, but in blender its not
        //whats best way to check if stuff like glGenBuffers has been inited?
        glewInit();

        system = NULL;

        printf("init: settings->system: %d\n", settings->system);
        
        //TODO choose based on settings
        if (settings->system == RTPSettings::Simple)
        {
            printf("simple system\n");
            system = new Simple(this, settings->max_particles);
        }
        else if (settings->system == RTPSettings::SPH)
        {
            printf("*** sph system 1  ***\n");
			settings->setMaxOuterParticles(4096*4);
            system = new SPH(this, settings->max_particles, settings->max_outer_particles);
			printf("max: %d\n", settings->max_outer_particles);
        }
        else if (settings->system == RTPSettings::FLOCK)
        {
            printf("flock system\n");
            system = new FLOCK(this, settings->max_particles);
        }
        else if (settings->system == RTPSettings::OUTER)
        {
            printf("*** outer system ***\n");
            system_outer = new OUTER(this, settings->max_outer_particles);
			printf("settings max particles: %d\n", settings->max_outer_particles);
			//exit(1);
            system = new SPH(this, settings->max_particles); //, settings->max_outer_particles);
			settings->setMaxOuterParticles(10048);
        }

        printf("created system in RTPS\n");
    }

    void RTPS::update()
    {
        //eventually we will support more systems
        //then we will want to iterate over them
        //or have more complex behavior if they interact
        system->update();
    }

    void RTPS::render()
    {
        system->render();
    }

    void RTPS::printTimers()
    {
            system->printTimers();
    }
};


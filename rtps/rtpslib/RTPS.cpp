//#include "GL/glew.h"
#include "RTPS.h"
#include "system/Simple.h"
#include "system/SPH.h"
#include "system/FLOCK.h"


namespace rtps
{

    RTPS::RTPS()
    {
        cli = new CL();
        cl_managed = true;
        //settings will be the default constructor
        Init();
    }

    RTPS::RTPS(RTPSettings s)
    {
        cli = new CL();
        cl_managed = true;
        settings = s;
        Init();
    }

    RTPS::RTPS(RTPSettings s, CL* _cli)
    {
        cli = _cli;
        cl_managed = false;
        settings = s;
        Init();
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
        //glewInit();

        system = NULL;
        //renderer = NULL;

        printf("init: settings.system: %d\n", settings.system);
        //TODO choose based on settings
        //system = new Simple(this, settings.max_particles);
        if (settings.system == RTPSettings::Simple)
        {
            printf("simple system\n");
            system = new Simple(this, settings.max_particles);
        }
        else if (settings.system == RTPSettings::SPH)
        {
            printf("sph system\n");
            system = new SPH(this, settings.max_particles);
        }
        else if (settings.system == RTPSettings::FLOCK)
        {
            printf("flock system\n");
            system = new FLOCK(this, settings.max_particles);
        }


        //pass in the position and color vbo ids to the renderer
        //get the number from the system
        //renderer = new Render(system->getPosVBO(), system->getColVBO(), system->getNum());
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
        /*renderer->render();
        //this functionality should be inside the system's render() function
        //so System should own the renderer object
        if(settings.system == RTPSettings::SPH)
        {
            Domain grid = system->getGrid();
            //should check if grid exists
            renderer->render_box(grid.getBndMin(), grid.getBndMax());
            renderer->render_table(grid.getBndMin(), grid.getBndMax());
        }*/
    }

    void RTPS::printTimers()
    {
            system->printTimers();
    }
}




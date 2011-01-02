#include "RTPS.h"
#include "system/Simple.h"
#include "system/SPH.h"

#include <android/log.h>

namespace rtps{
    
RTPS::RTPS()
{
    //settings will be the default constructor
    Init();
}

RTPS::RTPS(RTPSettings s)
{
    settings = s;
    Init();
}

RTPS::~RTPS()
{
    delete system;
    delete renderer;
}

void RTPS::Init()
{
    //this should already be done, but in blender its not
    //whats best way to check if stuff like glGenBuffers has been inited?
    //glewInit();
    
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
    

    //pass in the position and color vbo ids to the renderer
    //get the number from the system
    renderer = new Render(system->getPosVBO(), system->getColVBO(), system->getNum());
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

    //this functionality should be inside the system's render() function
    //so System should own the renderer object
    //if(settings.system == RTPSettings::SPH)
    {
        //Domain grid = system->getGrid();
        Domain grid = settings.grid;
        //should check if grid exists
        renderer->render_box(grid.getBndMin(), grid.getBndMax());
    }

    //__android_log_print(ANDROID_LOG_INFO, "RTPS", "about to render");
    renderer->render();
}

void RTPS::updateNum(int num)
{
    printf("about to test for renderer\n");
    if(renderer)
    {   
        renderer->setNum(num);
    }   
    //this segfaults for some reason
    //system->setNum(num);
}


}

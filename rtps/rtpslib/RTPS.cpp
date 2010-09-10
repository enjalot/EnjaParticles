#include "RTPS.h"
#include "system/Simple.h"


namespace rtps{
    
RTPS::RTPS()
{
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
    delete cli;
    delete renderer;
}

void RTPS::Init()
{

    cli = new CL();
    //choose based on settings
    system = new Simple(this, settings.max_particles);

    //pass in the position and color vbo ids to the renderer
    //get the number from the system
    renderer = new Render(system->pos_vbo, system->col_vbo, system->num);
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
    renderer->render();
}

}


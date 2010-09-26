#include "RTPS.h"
#include "system/Simple.h"
#include "system/SPH.h"


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
    delete cli;
    delete renderer;
}

void RTPS::Init()
{

    cli = new CL();

    //TODO choose based on settings
    system = new Simple(this, settings.max_particles);
	printf("max_particles: %d\n", settings.max_particles);
    //system = new SPH(this, settings.max_particles);

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

    UniformGrid grid = system->getGrid();
    //should check if grid exists
    renderer->render_box(grid.getMin(), grid.getMax());

    renderer->render();
}

}


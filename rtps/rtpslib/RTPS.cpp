#include "RTPS.h"

// POOR DESIGN: you'd have to load all the particle systems. If you define 
// 30 different types, you'd have to load them all. Ok, but better approach
// might be dynamically loadable libraries. Just a thought. GE. 

#include "system/Simple.h"
#include "system/SPH.h"
#include "system/GE_SPH.h"



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
	printf("*** RTPS destructor ****\n");
	printf("*** before delete system ***\n");
    delete system;
	printf("*** after delete system ***\n");
    delete cli;
	printf("*** after delete cli ***\n");
    delete renderer;
	printf("*** after delete renderer ***\n");
}

void RTPS::Init()
{

    cli = new CL();

    //TODO choose based on settings
    //system = new Simple(this, settings.max_particles);
	printf("settings.max_particles: %d\n", settings.max_particles);
    //system = new SPH(this, settings.max_particles);

	printf("Call GE_SPH\n");
    system = new GE_SPH(this, settings.max_particles);

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

	glPushMatrix();
	//float scale = 30;
	//glScalef(scale,scale,scale);
    renderer->render_box(grid.getMin(), grid.getMax());
    renderer->render();
	glPopMatrix();
}

}


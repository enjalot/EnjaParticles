//
//
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>


//#include <utils.h>

//#include <string.h>
//#include <string>
#include <sstream>
#include <iomanip>

#include <GL/glew.h>
#if defined __APPLE__ || defined(MACOSX)
    #include <GLUT/glut.h>
#else
    #include <GL/glut.h>
    //OpenCL stuff
#endif

#include "enja.h"
#include "timege.h"


int window_width = 800;
int window_height = 600;
int glutWindowHandle = 0;
float translate_z = -4.f;

// mouse controls
int mouse_old_x, mouse_old_y;
int mouse_buttons = 0;
float rotate_x = 0.0, rotate_y = 0.0;
std::vector<Triangle> triangles;
std::vector<Box> boxes;

// offsets into the triangle list. tri_offsets[i] corresponds to the 
// triangle list for box[i]. Number of triangles for triangles[i] is
//    tri_offsets[i+1]-tri_offsets[i]
// Add one more offset so that the number of triangles in 
//   boxes[boxes.size()-1] is tri_offsets[boxes.size()]-tri_offsets[boxes.size()-1]
std::vector<int> tri_offsets;


void init_gl();

void appKeyboard(unsigned char key, int x, int y);
void appRender();
void appDestroy();

void appMouse(int button, int state, int x, int y);
void appMotion(int x, int y);

void timerCB(int ms);

void drawString(const char *str, int x, int y, float color[4], void *font);
void showFPS(float fps, std::string *report);
void *font = GLUT_BITMAP_8_BY_13;

EnjaParticles* enjas;
#define NUM_PARTICLES (1 << 10) << 8


GLuint v_vbo; //vbo id
GLuint c_vbo; //vbo id

//timers
GE::Time *ts[3];

#include "EnjaSimBuffer.h"
#include <SimulationSystem.h>
// SPH global variables
//SnowSim::Config* mSnowConfig;
SimLib::SimulationSystem* mParticleSystem;

struct FluidSettings
{
	bool simpleSPH;
	bool enabled;
	bool enableKernelTiming;
	bool showFluidGrid;
	bool gridWallCollisions;
	bool terrainCollisions;
};
FluidSettings* fluidSettings;


//================
#include "materials_lights.h"

//----------------------------------------------------------------------
float rand_float(float mn, float mx)
{
	float r = random() / (float) RAND_MAX;
	return mn + (mx-mn)*r;
}

//----------------------------------------------------------------------
bool frameRenderingQueued()
{
	if(mParticleSystem) {
		bool mProgress = true;
        printf("simulating!\n");
		mParticleSystem->Simulate(mProgress, fluidSettings->gridWallCollisions);
	}
	return true;
}
//----------------------------------------------------------------------
void appRender()
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glEnable(GL_DEPTH_TEST);

    printf("about to call render\n");
 
    glBegin(GL_TRIANGLES);
    glColor3f(0,1,0);
    	glVertex3f(0, 0, 70);
    	glVertex3f(100, -40, 70);
    	glVertex3f(100, 0, 70);
    glEnd();



	printf("frameRenderQueued\n");
	frameRenderingQueued();

    enjas->render();

    showFPS(enjas->getFPS(), enjas->getReport());
    glutSwapBuffers();
    //if we want to render as fast as possible we do this
    //glutPostRedisplay();

	glDisable(GL_DEPTH_TEST);
}

void SetScene(int scene) 
{	
	if(!mParticleSystem) return;

	//lastScene = scene;
    printf("setting scene: %d\n", scene);
	mParticleSystem->SetScene(scene);
    printf("scene set\n");
}
//----------------------------------------------------------------------
void createScene()
	{
        SimLib::SimCudaHelper* simCudaHelper = new SimLib::SimCudaHelper();
        simCudaHelper->Initialize(0);

		fluidSettings = new FluidSettings();
		fluidSettings->simpleSPH = true;
		fluidSettings->enabled = true;
		fluidSettings->enableKernelTiming = true;
		fluidSettings->showFluidGrid = false;
		fluidSettings->gridWallCollisions = false;
		fluidSettings->terrainCollisions = false;

		//if(mSnowConfig->fluidSettings.enabled) {
			mParticleSystem = new SimLib::SimulationSystem(fluidSettings->simpleSPH);
			//int numParticles = (1 << 10) << 2;
			printf("NUM_PARTICLES= %d\n", NUM_PARTICLES);

			mParticleSystem->SetFluidPosition(make_float3(0., 0., 0.));

            printf("where we at?\n");
            Enja::EnjaCudaHelper* ech = new Enja::EnjaCudaHelper(simCudaHelper);

            ech->RegisterHardwareBuffer(enjas->v_vbo);
            ech->RegisterHardwareBuffer(enjas->c_vbo);
            printf("making pos buffer\n");
            
            Enja::EnjaSimBuffer* pos_vbo = new Enja::EnjaSimBuffer(ech);
            printf("setting pos buffer\n");
            printf("enjas->v_vbo: %d\n", enjas->v_vbo);
            pos_vbo->SetEnjaVertexBuffer(enjas->v_vbo);

            printf("making col buffer\n");
            Enja::EnjaSimBuffer* col_vbo = new Enja::EnjaSimBuffer(ech);
            printf("setting col buffer\n");
            col_vbo->SetEnjaVertexBuffer(enjas->c_vbo);

            printf("setting external pos buffer\n");
			mParticleSystem->SetExternalBuffer(SimLib::Sim::BufferPosition, pos_vbo); 
            printf("setting external col buffer\n");
			mParticleSystem->SetExternalBuffer(SimLib::Sim::BufferColor, col_vbo);
            printf("init:\n");
            mParticleSystem->Init();



            printf("settings: \n");
        	mParticleSystem->GetSettings()->SetValue("Particles Number", NUM_PARTICLES);
            printf("asdfasdf\n");
			mParticleSystem->PrintMemoryUse();


			#if 0
			Ogre::ConfigFile::SettingsIterator iter = mSnowConfig->getCfg()->getSettingsIterator("FluidParams");

			while(iter.hasMoreElements())
			{
				String name = iter.peekNextKey();
				String value = iter.getNext();
				float val =  StringConverter::parseReal(value);

				
				if(!StringUtil::startsWith(name, "//")) 
					mParticleSystem->GetSettings()->SetValue(name, val);
			}
			#endif

			//mVolumeSize = mParticleSystem->GetSettings()->GetValue("Grid World Size");
			//mNumParticles = mParticleSystem->GetSettings()->GetValue("Particles Number");

			//setParticleMaterial(mSnowConfig->generalSettings.fluidShader);
		

			// create material for fluid cube/grid
			#if 0
			Ogre::MaterialPtr gridMaterial = MaterialManager::getSingleton().create("FluidGridMaterial", Ogre::ResourceGroupManager::DEFAULT_RESOURCE_GROUP_NAME);
			gridMaterial->setReceiveShadows(false);
			//gridMaterial->createTechnique()->createPass();
			gridMaterial->getTechnique(0)->setLightingEnabled(false);
			gridMaterial->getTechnique(0)->getPass(0)->setDiffuse(0, 0, 1, 0);
			gridMaterial->getTechnique(0)->getPass(0)->setAmbient(0, 0, 1); 
			gridMaterial->getTechnique(0)->getPass(0)->setSelfIllumination(0, 0, 1);
			gridMaterial->load();
			#endif

			// Draw cube of the fluid grid/simulation volume
	//}

	int scene = 1; // any value from 0 to 9
    printf("setting scene\n");
	SetScene(scene);
}
//----------------------------------------------------------------------
int main(int argc, char** argv)
{
    //initialize glut
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);
    glutInitWindowSize(window_width, window_height);
    glutInitWindowPosition (glutGet(GLUT_SCREEN_WIDTH)/2 - window_width/2, 
                            glutGet(GLUT_SCREEN_HEIGHT)/2 - window_height/2);

    
    std::stringstream ss;
    ss << "EnjaParticles: " << NUM_PARTICLES << std::ends;
    glutWindowHandle = glutCreateWindow(ss.str().c_str());

    glutDisplayFunc(appRender); //main rendering function
    glutTimerFunc(30, timerCB, 30); //determin a minimum time between frames
    glutKeyboardFunc(appKeyboard);
    glutMouseFunc(appMouse);
    glutMotionFunc(appMotion);
	//----------------------

	define_lights_and_materials();

    // initialize necessary OpenGL extensions
    glewInit();
    GLboolean bGLEW = glewIsSupported("GL_VERSION_2_0 GL_ARB_pixel_buffer_object"); 
    printf("GLEW supported?: %d\n", bGLEW);

    //initialize the OpenGL scene for rendering
    init_gl();

    printf("before we call enjas functions\n");

    //parameters: system and number of particles
    //system = 0: lorenz
    //system = 1 gravity
    //system = 2 vfield

	//------------------
	// Create sph particle system
	// NOT DEFINED
	//SnowSim::SnowApplication app;
	//app.go();


    
    printf("INITIALIZE ENJAS\n");
    //default constructor
    enjas = new EnjaParticles(EnjaParticles::GRAVITY, NUM_PARTICLES);
    enjas->particle_radius = 10.0f;
    //enjas->use_glsl();
    enjas->updates = 1;
    enjas->dt = .005;
    //enjas->collision = true;
	
	printf("INITIALIZE SPH CODE\n");
    createScene();
   
   
    glutMainLoop();
    
    printf("doesn't happen does it\n");
    appDestroy();
    return 0;
}
//----------------------------------------------------------------------
//----------------------------------------------------------------------
void init_gl()
{
    // default initialization
    glClearColor(0.0, 0.0, 0.0, 1.0);
    glDisable(GL_DEPTH_TEST);

    // viewport
    glViewport(0, 0, window_width, window_height);

    // projection
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    //gluPerspective(60.0, (GLfloat)window_width / (GLfloat) window_height, 0.1, 100.0);
    //gluPerspective(90.0, (GLfloat)window_width / (GLfloat) window_height, 0.1, 10000.0); //for lorentz
    glOrtho(-100,300, -100,300, 0,10000);
    gluLookAt(0,0,200, 0,0,0, 0,1,0);
    // set view matrix
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glTranslatef(0.0, 0.0, translate_z);
    //glRotatef(-90, 1.0, 0.0, 0.0);

    return;

}

//----------------------------------------------------------------------
void appKeyboard(unsigned char key, int x, int y)
{
    switch(key) 
    {
        case '\033': // escape quits
        case '\015': // Enter quits    
        case 'Q':    // Q quits
        case 'q':    // q (or escape) quits
            // Cleanup up and quit
            appDestroy();
            break;
    }
}


//----------------------------------------------------------------------
void appDestroy()
{

    delete enjas;
    if(glutWindowHandle)glutDestroyWindow(glutWindowHandle);
    printf("about to exit!\n");

    exit(0);
}

//----------------------------------------------------------------------
void timerCB(int ms)
{
    glutTimerFunc(ms, timerCB, ms);
    glutPostRedisplay();
}


//----------------------------------------------------------------------
void appMouse(int button, int state, int x, int y)
{
    if (state == GLUT_DOWN) {
        mouse_buttons |= 1<<button;
    } else if (state == GLUT_UP) {
        mouse_buttons = 0;
    }

    mouse_old_x = x;
    mouse_old_y = y;

    glutPostRedisplay();
}

//----------------------------------------------------------------------
void appMotion(int x, int y)
{
    float dx, dy;
    dx = x - mouse_old_x;
    dy = y - mouse_old_y;

    if (mouse_buttons & 1) {
        rotate_x += dy * 0.2;
        rotate_y += dx * 0.2;
    } else if (mouse_buttons & 4) {
        translate_z += dy * 0.1;
    }

    mouse_old_x = x;
    mouse_old_y = y;

    // set view matrix
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glTranslatef(0.0, 0.0, translate_z);
    glRotatef(rotate_x, 1.0, 0.0, 0.0);
    glRotatef(rotate_y, 0.0, 1.0, 0.0);
    glutPostRedisplay();
}


//----------------------------------------------------------------------
///////////////////////////////////////////////////////////////////////////////
// write 2d text using GLUT
// The projection matrix must be set to orthogonal before call this function.
///////////////////////////////////////////////////////////////////////////////
void drawString(const char *str, int x, int y, float color[4], void *font)
{
    glPushAttrib(GL_LIGHTING_BIT | GL_CURRENT_BIT); // lighting and color mask
    glDisable(GL_LIGHTING);     // need to disable lighting for proper text color

    glColor4fv(color);          // set text color
    glRasterPos2i(x, y);        // place text position

    // loop all characters in the string
    while(*str)
    {
        glutBitmapCharacter(font, *str);
        ++str;
    }

    glEnable(GL_LIGHTING);
    glPopAttrib();
}

///////////////////////////////////////////////////////////////////////////////
// display frame rates
///////////////////////////////////////////////////////////////////////////////
void showFPS(float fps, std::string* report)
{
    static std::stringstream ss;

    // backup current model-view matrix
    glPushMatrix();                     // save current modelview matrix
    glLoadIdentity();                   // reset modelview matrix

    // set to 2D orthogonal projection
    glMatrixMode(GL_PROJECTION);        // switch to projection matrix
    glPushMatrix();                     // save current projection matrix
    glLoadIdentity();                   // reset projection matrix
    gluOrtho2D(0, 400, 0, 300);         // set to orthogonal projection

    float color[4] = {1, 1, 0, 1};

    // update fps every second
    ss.str("");
    ss << std::fixed << std::setprecision(1);
    ss << fps << " FPS" << std::ends; // update fps string
    ss << std::resetiosflags(std::ios_base::fixed | std::ios_base::floatfield);
    drawString(ss.str().c_str(),  15, 286, color, font);
    drawString(report[0].c_str(), 15, 273, color, font);
    drawString(report[1].c_str(), 15, 260, color, font);

    // restore projection matrix
    glPopMatrix();                      // restore to previous projection matrix

    // restore modelview matrix
    glMatrixMode(GL_MODELVIEW);         // switch to modelview matrix
    glPopMatrix();                      // restore to previous modelview matrix
}
//----------------------------------------------------------------------


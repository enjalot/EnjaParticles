/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and 
 * proprietary rights in and to this software and related documentation. 
 * Any use, reproduction, disclosure, or distribution of this software 
 * and related documentation without an express license agreement from
 * NVIDIA Corporation is strictly prohibited.
 *
 * Please refer to the applicable NVIDIA end user license agreement (EULA) 
 * associated with this source code for terms and conditions that govern 
 * your use of this NVIDIA software.
 * 
 */

//#include "oclUtils.h"

#include <GL/glew.h>
#if defined __APPLE__ || defined(MACOSX)
    #include <GLUT/glut.h>
#else
    #include <GL/glut.h>
    //OpenCL stuff
#endif

#if defined (_WIN32)
    #include <GL/wglew.h>
#endif


#include <paramgl.h>
#include "particleSystem_class.h"
#include "particleSystem_common.h"
#include "particleSystem_engine.h"
#include "render_particles.h"

#define MAX_EPSILON_ERROR 5.00f
#define THRESHOLD         0.30f
#define GRID_SIZE         64
#define NUM_PARTICLES     16384*4

// view, GLUT and display params
int ox, oy;
int buttonState = 0;
float camera_trans[] = {0, 0, -3};
float camera_rot[]   = {0, 0, 0};
float camera_trans_lag[] = {0, 0, -3};
float camera_rot_lag[] = {0, 0, 0};
const float inertia = 0.1;
ParticleRenderer::DisplayMode displayMode = ParticleRenderer::PARTICLE_SPHERES;
int mode = 0;
bool displayEnabled = true;
bool bPause = false;
bool displaySliders = false;
enum { M_VIEW = 0, M_MOVE };
uint numParticles = 0;
uint3 gridSize;
ParticleRenderer* renderer = 0;     // Main particle renderer instance
float modelView[16];
ParamListGL* params;
int iGLUTWindowHandle;              // handle to the GLUT window
int iGLUTMenuHandle;                // handle to the GLUT menu
int iGraphicsWinPosX = 0;           // GLUT Window X location
int iGraphicsWinPosY = 0;           // GLUT Window Y location
int iGraphicsWinWidth = 1024;       // GLUT Window width
int iGraphicsWinHeight = 768;       // GL Window height

// Simulation parameters
float timestep = 0.5f;              // time slice for re-computation iteration
float gravity = 0.0005f;            // Strength of gravity
float damping = 1.0f;
int iterations = 1;
float fParticleRadius = 0.023f;     // Radius of individual particles
int ballr = 8;                      // Radius (in particle diameter equivalents) of dropped/shooting sphere of particles for keys '3' and '4'
float fShootVelocity = -0.07f;      // Velocity of shooting sphere of particles for key '4' (- is away from viewer)
float fColliderRadius = 0.17f;      // Radius of collider for interacting with particles in 'm' mode
float collideSpring = 0.4f;         // Elastic spring constant for impact between particles
float collideDamping = 0.025f;      // Inelastic loss component for impact between particles
float collideShear = 0.12f;         // Friction constant for particles in contact
float collideAttraction = 0.0012f;  // Attraction between particles (~static or Van der Waals) 
ParticleSystem* psystem = 0;        // Main particle system instance

// fps, quick test and qatest vars
const char* cProcessor [] = {"OpenCL GPU", "Host C++ CPU"};
int iProcFlag = 0;                  // 0 = GPU, 1 = CPU
int iFrameCount = 0;                // FPS count for averaging
int iFrameTrigger = 90;             // FPS trigger for sampling
int iFramesPerSec = 60;             // frames per second
bool bFullScreen = false;           // state var for full screen mode or not
shrBOOL bNoPrompt = shrFALSE;       // false = normal GL loop, true = Finite period of GL loop (a few seconds)
shrBOOL bQATest = shrFALSE;         // false = normal GL loop, true = run No-GL test sequence
shrBOOL bTour = shrTRUE;            // true = cycles between modes, false = stays on selected 1 mode (manually switchable)
int iTestSets = 3;                  // # of loop set retriggers before auto exit when bNoPrompt = shrTrue
int iSetCount = 0;                  // Var for present set count 

// Forward Function declarations
//*****************************************************************************
// OpenCL Simulation, test and demo
void ResetSim(int iOption);
void initParticleSystem(int numParticles, uint3 gridSize);
void initParams();

// OpenGL (GLUT) functionality
void InitGL(int argc, const char** argv);
void DisplayGL(void);
void ReshapeGL(int w, int h);
void IdleGL(void);
void KeyboardGL(unsigned char key, int x, int y);
void MouseGL(int button, int state, int x, int y);
void MotionGL(int x, int y);
void SpecialGL (int key, int x, int y);
void MenuGL(int i);

// Helpers
void TestNoGL();
void TriggerFPSUpdate();
void ResetViewTransform();
void Cleanup(int iExitCode);

// Main program
//*****************************************************************************
int main(int argc, const char** argv) 
{
    numParticles = NUM_PARTICLES;
    uint gridDim = GRID_SIZE;

    // Start logs and timers
    shrSetLogFileName ("oclParticles.txt");
    //shrLog("%s Starting...\n\n", argv[0]); 

    // check command line flags and parameters
    if (argc > 1) 
    {
        shrGetCmdLineArgumenti(argc, argv, "n", (int*)&numParticles);
        shrGetCmdLineArgumenti(argc, argv, "grid", (int*)&gridDim);
        bQATest = shrCheckCmdLineFlag(argc, argv, "qatest");
        bNoPrompt = shrCheckCmdLineFlag(argc, argv, "noprompt");
    }

    // Set and log grid size and particle count, after checking optional command-line inputs
    gridSize.x = gridSize.y = gridSize.z = gridDim;
    //shrLog(" grid: %d x %d x %d = %d cells\n", gridSize.x, gridSize.y, gridSize.z, gridSize.x * gridSize.y * gridSize.z);
    //shrLog(" particles: %d\n\n", numParticles);

    // initialize GLUT and GLEW
    if(!bQATest) 
    {
        InitGL(argc, argv);
    }

    // initialize OpenCL
    startupOpenCL(argc, argv);

    // init simulation parameters and objects
    initParticleSystem(numParticles, gridSize);
    initParams();

    // Init timers 
    shrDeltaT(0);  // timer 0 is for processing time measurements
    shrDeltaT(1);  // timer 1 is for fps measurement   

    // Start main GLUT rendering loop for processing and rendering, 
	// or otherwise run No-GL Q/A test sequence
    if(!bQATest) 
    {
	    glutMainLoop();
    }
    else 
    {
        TestNoGL();
    }

    // Normally unused return path
    Cleanup(EXIT_FAILURE);
}

// initialize particle system
//*****************************************************************************
void initParticleSystem(int numParticles, uint3 gridSize)
{
    psystem = new ParticleSystem(numParticles, gridSize, fParticleRadius, fColliderRadius, bQATest); 
    psystem->reset(ParticleSystem::CONFIG_GRID);
    psystem->setIterations(iterations);
    psystem->setDamping(damping);
    psystem->setGravity(-gravity);
    psystem->setCollideSpring(collideSpring);
    psystem->setCollideDamping(collideDamping);
    psystem->setCollideShear(collideShear);
    psystem->setCollideAttraction(collideAttraction);

    if (!bQATest)
    {
        renderer = new ParticleRenderer;
        renderer->setParticleRadius(psystem->getParticleRadius());
        renderer->setColorBuffer(psystem->getColorBuffer());
    }
}

// Init simulations parameters
//*****************************************************************************
void initParams(void)
{
     // create a new parameter list
    params = new ParamListGL("misc");
    params->AddParam(new Param<float>("time step", timestep, 0.0, 1.0, 0.01, &timestep));
    params->AddParam(new Param<float>("damping", damping, 0.0, 1.0, 0.001, &damping));
    params->AddParam(new Param<float>("gravity", gravity, 0.0, 0.001, 0.0001, &gravity));
    params->AddParam(new Param<int>("ball radius", ballr, 1, 20, 1, &ballr));

    params->AddParam(new Param<float>("collide spring", collideSpring, 0.0, 1.0, 0.001, &collideSpring));
    params->AddParam(new Param<float>("collide damping", collideDamping, 0.0, 0.1, 0.001, &collideDamping));
    params->AddParam(new Param<float>("collide shear", collideShear, 0.0, 0.1, 0.001, &collideShear));
    params->AddParam(new Param<float>("collide attract", collideAttraction, 0.0, 0.1, 0.001, &collideAttraction));
}

// Setup function for GLUT parameters and loop
//*****************************************************************************
void InitGL(int argc, const char** argv)
{  
    // init GLUT 
    glutInit(&argc, (char**)argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_DEPTH | GLUT_DOUBLE);
    glutInitWindowPosition (glutGet(GLUT_SCREEN_WIDTH)/2 - iGraphicsWinWidth/2, 
                            glutGet(GLUT_SCREEN_HEIGHT)/2 - iGraphicsWinHeight/2);
    glutInitWindowSize(iGraphicsWinWidth, iGraphicsWinHeight);
    iGLUTWindowHandle = glutCreateWindow("OpenCL particles");

    // register GLUT callbacks
    glutDisplayFunc(DisplayGL);
    glutReshapeFunc(ReshapeGL);
    glutMouseFunc(MouseGL);
    glutMotionFunc(MotionGL);
    glutKeyboardFunc(KeyboardGL);
    glutSpecialFunc(SpecialGL);
    glutIdleFunc(IdleGL);
        
    // init GLEW
    glewInit();
    if (!glewIsSupported("GL_VERSION_2_0 "
                         "GL_VERSION_1_5 "
                         "GL_ARB_multitexture "
                         " GL_ARB_vertex_buffer_object")) 
    {
        //shrLog("Required OpenGL extensions missing !!!\n");
        Cleanup(EXIT_FAILURE);
    }

    #ifdef _WIN32
        if (wglewIsSupported("WGL_EXT_swap_control")) 
        {
            // disable vertical sync
            wglSwapIntervalEXT(0);
        }
    #endif
    glEnable(GL_DEPTH_TEST);
    glClearColor(0.25, 0.25, 0.25, 1.0);
    glutReportErrors();
    
    // create GLUT menu    
    iGLUTMenuHandle = glutCreateMenu(MenuGL);
    glutAddMenuEntry("Reset Stacked Block [1]", '1');
    glutAddMenuEntry("Reset Random Paricle Cloud [2]", '2');
    glutAddMenuEntry("Drop particle sphere [3]", '3');
    glutAddMenuEntry("Shoot particle sphere [4]", '4');
    glutAddMenuEntry("Change to 'View' mode [v]", 'v');
    glutAddMenuEntry("Change to 'Move cue-ball' mode [m]", 'm');
    glutAddMenuEntry("Toggle Tour mode [t]", 't');
    glutAddMenuEntry("Toggle point rendering [p]", 'p');
    glutAddMenuEntry("Toggle animation On/Off <spacebar>", ' ');
    glutAddMenuEntry("Toggle between Full Screen and Windowed [f]", 'f');
    glutAddMenuEntry("Step animation <return>", 13);
    glutAddMenuEntry("Toggle sliders [h]", 'h');
    glutAddMenuEntry("Quit <esc>", '\033');
    glutAttachMenu(GLUT_RIGHT_BUTTON);

    // Init view transform
    ResetViewTransform();
}

// Primary GLUT callback loop function
//*****************************************************************************
void DisplayGL()
{
    // update the simulation, if not paused
    double dProcessingTime = 0.0;
    if (!bPause)
    {
        // start timer 0 if it's update time
        if (iFrameCount >= iFrameTrigger)
        {
            shrDeltaT(0); 
        }

        // do the processing
        psystem->update(timestep); 
        renderer->setVertexBuffer(psystem->getCurrentReadBuffer(), psystem->getNumParticles());

        // get processing time from timer 0, if it's update time
        if (iFrameCount >= iFrameTrigger)
        {
            dProcessingTime = shrDeltaT(0); 
        }
    }

    // Clear last frame
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);  

    // Add cube
    glColor3f(1.0, 1.0, 1.0);
    glutWireCube(2.0);

    // Add collider
    glPushMatrix();
    float3 p = psystem->getColliderPos();
    glTranslatef(p.x, p.y, p.z);
    glColor3f(0.75, 0.0, 0.5);
    glutWireSphere(psystem->getColliderRadius(), 30, 15);
    glPopMatrix();

    // Render
    if (displayEnabled)
    {
        renderer->display(displayMode);
    }

    // Display user interface if enabled
    if (displaySliders) 
    {
        glDisable(GL_DEPTH_TEST);
        glBlendFunc(GL_ONE_MINUS_DST_COLOR, GL_ZERO); // invert color
        glEnable(GL_BLEND);
        params->Render(0, 0);
        glDisable(GL_BLEND);
        glEnable(GL_DEPTH_TEST);
    }
    
    //  Flip backbuffer to screen
    glutSwapBuffers();

    //  Increment the frame counter, and do fps stuff if it's time
    if (iFrameCount++ > iFrameTrigger) 
    {
        // Set the display window title
        char cTitle[256];
        iFramesPerSec = (int)((double)iFrameCount/ shrDeltaT(1));
#define GPU_PROFILING
#ifdef GPU_PROFILING
        if(!bPause)
        {
            #ifdef _WIN32
                sprintf_s(cTitle, 256, "%s Particles Simulation (%u particles) | %i fps | Proc. t = %.4f s", 
                    cProcessor[iProcFlag], numParticles, iFramesPerSec, dProcessingTime);
            #else
                sprintf(cTitle, "%s OpenCL Particles Simulation (%u particles) | %i fps | Proc. t = %.4f s", 
                    cProcessor[iProcFlag], numParticles, iFramesPerSec, dProcessingTime);
            #endif
        }
        else 
        {
            #ifdef _WIN32
                sprintf_s(cTitle, 256, "%s Particles Simulation (%u particles) (Paused) | %i fps", 
                    cProcessor[iProcFlag], numParticles, iFramesPerSec);
            #else
                sprintf(cTitle, "%s OpenCL Particles Simulation (%u particles) (Paused) | %i fps", 
                    cProcessor[iProcFlag], numParticles, iFramesPerSec);
            #endif
        }
#else
        if(!bPause)
        {
            #ifdef _WIN32
                sprintf_s(cTitle, 256, "%s Particles Simulation (%u particles)", 
                    cProcessor[iProcFlag], numParticles);
            #else
                sprintf(cTitle, "%s OpenCL Particles Simulation (%u particles)", 
                    cProcessor[iProcFlag], numParticles);
            #endif
        }
        else 
        {
            #ifdef _WIN32
                sprintf_s(cTitle, 256, "%s Particles Simulation (%u particles) (Paused)", 
                    cProcessor[iProcFlag], numParticles);
            #else
                sprintf(cTitle, "%s OpenCL Particles Simulation (%u particles) (Paused)", 
                    cProcessor[iProcFlag], numParticles);
            #endif
        }
#endif
        glutSetWindowTitle(cTitle);

        // Log fps and processing info to console and file 
        //shrLog("%s\n", cTitle); 

        // Set based options:  QuickTest or cycle demo
        if (iSetCount++ == iTestSets)
        {
            if (bNoPrompt) 
            {
                Cleanup(EXIT_SUCCESS);
            }
            if (bTour) 
            {
                static int iOption = 1;
                ResetSim(++iOption);
                if (iOption > 3)iOption = 0;
            }
            iSetCount = 0;
        }

        // reset the frame count and trigger
        iFrameCount = 0; 
        iFrameTrigger = (iFramesPerSec > 1) ? iFramesPerSec * 2 : 1;
    }
}

// Inject a sphere of particles
//*****************************************************************************
void addSphere()
{
    float pr = psystem->getParticleRadius();
    float tr = pr + (pr * 2.0f) * ballr;
    float pos[4], vel[4];
    pos[0] = -1.0f + tr + (2.0f - tr * 2.0f) * (float)rand() / (float)RAND_MAX;
    pos[1] = 1.0f - tr;
    pos[2] = -1.0f + tr + (2.0f - tr * 2.0f) * (float)rand() / (float)RAND_MAX;
    pos[3] = 0.0f;
    vel[0] = vel[1] = vel[2] = vel[3] = 0.0f;
    psystem->addSphere(0, pos, vel, ballr, pr * 2.0f);
}

// Handler for GLUT window resize event
//*****************************************************************************
void ReshapeGL(int w, int h)
{
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60.0, (float) w / (float) h, 0.1, 100.0);

    glMatrixMode(GL_MODELVIEW);
    glViewport(0, 0, w, h);

    renderer->setWindowSize(w, h);
    renderer->setFOV(60.0);
}

// Handler for GLUT Mouse events
//*****************************************************************************
void MouseGL(int button, int state, int x, int y)
{
    int mods;

    if (state == GLUT_DOWN)
        buttonState |= 1<<button;
    else if (state == GLUT_UP)
        buttonState = 0;

    mods = glutGetModifiers();
    if (mods & GLUT_ACTIVE_SHIFT) 
    {
        buttonState = 2;
    } 
    else if (mods & GLUT_ACTIVE_CTRL) 
    {
        buttonState = 3;
    }

    ox = x; 
    oy = y;

    if (displaySliders) 
    {
        if (params->Mouse(x, y, button, state)) 
        {
            glutPostRedisplay();
            return;
        }
    }

    glutPostRedisplay();
}

// transform vector by matrix
//*****************************************************************************
void xform(float *v, float *r, GLfloat *m)
{
  r[0] = v[0]*m[0] + v[1]*m[4] + v[2]*m[8] + m[12];
  r[1] = v[0]*m[1] + v[1]*m[5] + v[2]*m[9] + m[13];
  r[2] = v[0]*m[2] + v[1]*m[6] + v[2]*m[10] + m[14];
}

// transform vector by transpose of matrix
//*****************************************************************************
void ixform(float *v, float *r, GLfloat *m)
{
  r[0] = v[0]*m[0] + v[1]*m[1] + v[2]*m[2];
  r[1] = v[0]*m[4] + v[1]*m[5] + v[2]*m[6];
  r[2] = v[0]*m[8] + v[1]*m[9] + v[2]*m[10];
}

//*****************************************************************************
void ixformPoint(float *v, float *r, GLfloat *m)
{
    float x[4];
    x[0] = v[0] - m[12];
    x[1] = v[1] - m[13];
    x[2] = v[2] - m[14];
    x[3] = 1.0f;
    ixform(x, r, m);
}

// GLUT mouse motion callback
//*****************************************************************************
void MotionGL(int x, int y)
{
    float dx, dy;
    dx = x - ox;
    dy = y - oy;

    if (displaySliders) 
    {
        if (params->Motion(x, y)) 
        {
            ox = x; 
            oy = y;
            glutPostRedisplay();
            return;
        }
    }

    switch(mode) 
    {
    case M_VIEW:
        if (buttonState == 3) 
        {
            // left+middle = zoom
            camera_trans[2] += (dy / 100.0) * 0.5 * fabs(camera_trans[2]);
        } 
        else if (buttonState & 2) 
        {
            // middle = translate
            camera_trans[0] += dx / 100.0;
            camera_trans[1] -= dy / 100.0;
        }
        else if (buttonState & 1) 
        {
            // left = rotate
            camera_rot[0] += dy / 5.0;
            camera_rot[1] += dx / 5.0;
        }
        break;

    case M_MOVE:
        {
            float translateSpeed = 0.003f;
            float3 p = psystem->getColliderPos();
            if (buttonState == 1) 
            {
                float v[3], r[3];
                v[0] = dx * translateSpeed;
                v[1] = -dy * translateSpeed;
                v[2] = 0.0f;
                ixform(v, r, modelView);
                p.x += r[0];
                p.y += r[1];
                p.z += r[2];
            } 
            else if (buttonState == 2) 
            {
                float v[3], r[3];
                v[0] = 0.0f;
                v[1] = 0.0f;
                v[2] = dy * translateSpeed;
                ixform(v, r, modelView);
                p.x += r[0];
                p.y += r[1];
                p.z += r[2];
            }
            psystem->setColliderPos(p);
        }
        break;
    }

    ox = x; 
    oy = y;

    // update view transform based upon mouse inputs
    ResetViewTransform();

    glutPostRedisplay();
}

// GLUT key event handler
// params commented out to remove unused parameter warnings in Linux
//*****************************************************************************
void ResetViewTransform()
{
    // Set view transform
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    for (int c = 0; c < 3; ++c)
    {
        camera_trans_lag[c] += (camera_trans[c] - camera_trans_lag[c]) * inertia;
        camera_rot_lag[c] += (camera_rot[c] - camera_rot_lag[c]) * inertia;
    }
    glTranslatef(camera_trans_lag[0], camera_trans_lag[1], camera_trans_lag[2]);
    glRotatef(camera_rot_lag[0], 1.0, 0.0, 0.0);
    glRotatef(camera_rot_lag[1], 0.0, 1.0, 0.0);
    glGetFloatv(GL_MODELVIEW_MATRIX, modelView);
}

// GLUT key event handler
// params commented out to remove unused parameter warnings in Linux
//*****************************************************************************
void KeyboardGL(unsigned char key, int /*x*/, int /*y*/)
{
    switch (key) 
    {
    case ' ':   // toggle pause in simulation computations
        bPause = !bPause;
        //shrLog("\nSimulation %s...\n", bPause ? "Paused" : "Running");
        break;
    case 13:
        psystem->update(timestep); 
        renderer->setVertexBuffer(psystem->getCurrentReadBuffer(), psystem->getNumParticles());
        break;
    case '\033':// Escape quits    
    case 'Q':   // Q quits
    case 'q':   // q quits
        Cleanup(EXIT_SUCCESS);
        break;
    case 'T':   // Toggles from (T)our mode to standard mode and back
    case 't':   // Toggles from (t)our mode to standard mode and back
        bTour = bTour ? shrFALSE : shrTRUE;
        //shrLog("\nTour Mode %s...\n", bTour ? "ON" : "OFF");
        break;
    case 'F':   // F toggles main graphics display full screen
    case 'f':   // f toggles main graphics display full screen
        bFullScreen = !bFullScreen;
        if (bFullScreen)
        {
            iGraphicsWinPosX = glutGet(GLUT_WINDOW_X);
            iGraphicsWinPosY = glutGet(GLUT_WINDOW_Y);
            iGraphicsWinWidth = glutGet(GLUT_WINDOW_WIDTH); 
            iGraphicsWinHeight = glutGet(GLUT_WINDOW_HEIGHT);
            glutFullScreen();
        }
        else
        {
            glutReshapeWindow(iGraphicsWinWidth, iGraphicsWinHeight);
            glutPositionWindow(iGraphicsWinPosX, iGraphicsWinPosY);
        }
        //shrLog("\nMain Graphics %s...\n", bFullScreen ? "FullScreen" : "Windowed");
        break;
    case 'V':
    case 'v':
        if (M_VIEW != mode)
        {
            //shrLog("\nMouse View Mode...\n");
            mode = M_VIEW;
        }
        break;
    case 'M':
    case 'm':
        if (M_MOVE != mode)
        {
            //shrLog("\nMouse Move Mode...\n");
            mode = M_MOVE;
        }
        break;
    case 'P':
    case 'p':
        displayMode = (ParticleRenderer::DisplayMode)
                      ((displayMode + 1) % ParticleRenderer::PARTICLE_NUM_MODES);
        break;
    case 'D':
    case 'd':
        psystem->dumpGrid();
        break;
    case 'U':
    case 'u':
        psystem->dumpParticles(0, numParticles-1);
        break;
    case 'R':
    case 'r':
        displayEnabled = !displayEnabled;
        break;
    case '1':
        ResetSim(1);
        break;
    case '2':
        ResetSim(2);
        break;
    case '3':
        ResetSim(3);
        break;
    case '4':
        ResetSim(4);
        break;
    case 'H':
    case 'h':
        displaySliders = !displaySliders;
        break;
    }

    // Trigger fps update and call for refresh
    TriggerFPSUpdate();
    glutPostRedisplay();
}

// Helper function to loop through different modes
//*****************************************************************************
void ResetSim(int iOption)
{
    if((iOption < 1) || (iOption > 4))return;
    switch (iOption)
    {
        case 1:
            psystem->reset(ParticleSystem::CONFIG_GRID);
            break;
        case 2:
            psystem->reset(ParticleSystem::CONFIG_RANDOM);
            break;
        case 3:
            addSphere();
            break;
        case 4:
            {
                // shoot ball from camera
                float pr = psystem->getParticleRadius();
                float vel[4], velw[4], pos[4], posw[4];
                vel[0] = 0.0f;
                vel[1] = 0.0f;
                vel[2] = fShootVelocity;
                vel[3] = 0.0f;
                ixform(vel, velw, modelView);

                pos[0] = 0.0f;
                pos[1] = 0.0f;
                pos[2] = -2.5f;
                pos[3] = 1.0;
                ixformPoint(pos, posw, modelView);
                posw[3] = 0.0f;

                psystem->addSphere(0, posw, velw, ballr, pr * 2.0f);
            }
            break;
    }
}

//*****************************************************************************
void SpecialGL(int k, int x, int y)
{
    if (displaySliders) 
    {
        params->Special(k, x, y);
    }
}

//*****************************************************************************
void IdleGL(void)
{
    glutPostRedisplay();
}

// Helper to trigger reset of fps vars at transition 
//*****************************************************************************
void TriggerFPSUpdate()
{
    iFrameCount = 0; 
    shrDeltaT(1);
    iFramesPerSec = 1;
    iFrameTrigger = 2;
}

// GLUT Menu callback
//*****************************************************************************
void MenuGL(int i)
{
    KeyboardGL((unsigned char) i, 0, 0);
}

// QATest sequence without any GL calls
//*****************************************************************************
void TestNoGL()
{
    // Warmup call to assure OpenCL driver is awake
    psystem->update(timestep); 

	// Start timer 0 and process n loops on the GPU
    shrDeltaT(0); 
    for (int i = 0; i < 10; i++)
    {
        psystem->update(timestep); 
    }

    // Get elapsed time and throughput, then log to sample and master logs
    double dAvgTime = shrDeltaT(0)/10.0;
    //shrLogEx(LOGBOTH | MASTER, 0, "oclParticles, Throughput = %.4f KParticles/s, Time = %.5f s, Size = %u particles, NumDevsUsed = %u\n", 
           //(1.0e-3 * numParticles)/dAvgTime, dAvgTime, numParticles, 1); 

    // Cleanup and exit
    Cleanup (EXIT_SUCCESS);
}

// Function to clean up and exit
//*****************************************************************************
void Cleanup(int iExitCode)
{
    //shrLog("\nStarting Cleanup...\n\n");

    // Delete main particle system instance
    if (psystem) delete psystem;

    // Cleanup OpenCL
    shutdownOpenCL();

    // kill the GLUT menu and window
    if(iGLUTMenuHandle)glutDestroyMenu(iGLUTMenuHandle);
    if(iGLUTWindowHandle)glutDestroyWindow(iGLUTWindowHandle);
    //shrLog("%s\n\n", iExitCode == 0 ? "PASSED" : "FAILED"); 

    // finalize logs and leave
    if (bNoPrompt || bQATest)
    {
        //shrLogEx(LOGBOTH | CLOSELOG, 0, "oclParticles.exe Exiting...\n");
    }
    else 
    {
        //shrLogEx(LOGBOTH | CLOSELOG, 0, "oclParticles.exe Exiting...\nPress <Enter> to Quit\n");
        #ifdef WIN32
            getchar();
        #endif
    }
    exit (iExitCode);
}

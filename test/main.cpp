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

#include <RTPS.h>
//#include "timege.h"
#include "../rtpslib/render/util/stb_image_write.h"

using namespace rtps;

//int window_width = 640;
//int window_height = 480;
int window_width = 640*2;
int window_height = 480*2;
int glutWindowHandle = 0;

float4 mouse_old;
float4 mouse_new;


#define DTR 0.0174532925

struct camera
{
    GLdouble leftfrustum;
    GLdouble rightfrustum;
    GLdouble bottomfrustum;
    GLdouble topfrustum;
    GLfloat modeltranslation;
} leftCam, rightCam;

bool stereo_enabled = false;
bool render_movie = false;
GLubyte* image = new GLubyte[window_width*window_height*4];
const char* render_dir = "./frames/";

char filename[512] = {'\0'};
unsigned int frame_counter = 0;
float depthZ = -10.0;                                      //depth of the object drawing

double fovy = 65.;                                          //field of view in y-axis
double aspect = double(window_width)/double(window_height);  //screen aspect ratio
double nearZ = 0.3;                                        //near clipping plane
double farZ = 100.0;                                        //far clipping plane
double screenZ = 10.0;                                     //screen projection plane
double IOD = 0.5;                                          //intraocular distance

float translate_x = -2.00f;
float translate_y = -2.70f;//300.f;
float translate_z = 3.50f;

float cloud_translate_x = 0.;
float cloud_translate_y = 0.;
float cloud_translate_z = 0.;
float cloud_velocity_x = 0.;
float cloud_velocity_y = 0.;
float cloud_velocity_z = 0.;

float4 rotation_axis;
void computeRotationAxis(float4& v1, float4& v2);

// track whether shift-key is down
bool shift_down = false;
bool ctrl_down = false;
// if shift is down, left mouse tracks arm (translation)
// if ctrl is down, left mouse tracks arm (rotation)
bool cloud_movement = false;


// mouse controls
int mouse_old_x, mouse_old_y;
int mouse_buttons = 0;
float rotate_x = 0.0, rotate_y = 0.0;
std::vector<Triangle> triangles;



void init_gl();
void render_stereo();
void setFrustum();

void appKeyboard(unsigned char key, int x, int y);
void keyUp(unsigned char key, int x, int y);
void appRender();
void appDestroy();

void appMouse(int button, int state, int x, int y);
void appMotion(int x, int y);
void resizeWindow(int w, int h);
void setShiftDown();

long Vectorize(float4& hit, float4& origin, float radius, float4& vec);

void timerCB(int ms);

void drawString(const char *str, int x, int y, float color[4], void *font);
void showFPS(float fps, std::string *report);
int write_movie_frame(const char* name);
void draw_collision_boxes();
void rotate_img(GLubyte* img, int size);

void *font = GLUT_BITMAP_8_BY_13;

rtps::RTPS* ps;

//#define NUM_PARTICLES 2000000
//#define NUM_PARTICLES 1000000
//#define NUM_PARTICLES 524288
//#define NUM_PARTICLES 262144
//#define NUM_PARTICLES 131072
//#define NUM_PARTICLES 65536
//#define NUM_PARTICLES 32768
//#define NUM_PARTICLES 16384
//#define NUM_PARTICLES 10000
//#define NUM_PARTICLES 8192
//#define NUM_PARTICLES 4096
//#define NUM_PARTICLES 2048
//#define NUM_PARTICLES 1024
//#define NUM_PARTICLES 256
//
//
//#define NUM_PARTICLES 4000
#define NUM_PARTICLES 100000

#define DT .003f

// contorl 
float4 color = float4(0.5, 0.3, 0.73, .05);
//float4 color = float4(1., 0.5, 0.0, 1.);
int hindex; 

// for rotation control via quaternion emulation
float4 screen_center = float4(500.,500.,0.,1.);



//timers
//GE::Time *ts[3];

//================
#include "materials_lights.h"

//----------------------------------------------------------------------
float rand_float(float mn, float mx)
{
    float r = rand() / (float) RAND_MAX;
    return mn + (mx-mn)*r;
}
//----------------------------------------------------------------------
//----------------------------------------------------------------------
int main(int argc, char** argv)
{
    //initialize glut
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_ALPHA | GLUT_DOUBLE | GLUT_DEPTH
		//|GLUT_STEREO //if you want stereo you must uncomment this.
		);
    glutInitWindowSize(window_width, window_height);
    glutInitWindowPosition (glutGet(GLUT_SCREEN_WIDTH)/2 - window_width/2, 
                            glutGet(GLUT_SCREEN_HEIGHT)/2 - window_height/2);


    int max_num = rtps::nlpo2(NUM_PARTICLES);
    std::stringstream ss;
    ss << "Real-Time Particle System: " << max_num << std::ends;
    glutWindowHandle = glutCreateWindow(ss.str().c_str());

    glutDisplayFunc(appRender); //main rendering function
    glutTimerFunc(30, timerCB, 30); //determin a minimum time between frames
    glutKeyboardFunc(appKeyboard);
    glutMouseFunc(appMouse);
    glutMotionFunc(appMotion);
    glutReshapeFunc(resizeWindow);

    define_lights_and_materials();

    // initialize necessary OpenGL extensions
    glewInit();
    GLboolean bGLEW = glewIsSupported("GL_VERSION_2_0 GL_ARB_pixel_buffer_object"); 
    printf("GLEW supported?: %d\n", bGLEW);


    //default constructor
    //rtps::RTPSettings settings;
    //rtps::Domain grid = Domain(float4(-5,-.3,0,0), float4(2, 2, 12, 0));
    rtps::Domain* grid = new Domain(float4(0,0,0,0), float4(5, 5, 5, 0));
    //grid->setBndMin( float4(1, 1, 1,0));
    //grid->setBndMax( float4(4, 4, 4, 0));

    //rtps::Domain grid = Domain(float4(0,0,0,0), float4(2, 2, 2, 0));
	rtps::RTPSettings* settings = new rtps::RTPSettings(rtps::RTPSettings::SPH, NUM_PARTICLES, DT, grid);
    

    //should be argv[0]
#ifdef WIN32
    settings->SetSetting("rtps_path", ".");
#else
    settings->SetSetting("rtps_path", "./bin");
    //settings->SetSetting("rtps_path", argv[0]);
    //printf("arvg[0]: %s\n", argv[0]);
#endif

    settings->setRenderType(RTPSettings::SCREEN_SPACE_RENDER);
    //settings->setRenderType(RTPSettings::RENDER);
    //settings.setRenderType(RTPSettings::SPRITE_RENDER);
    settings->setRadiusScale(0.4);
    settings->setBlurScale(1.0);
    settings->setUseGLSL(1);

    settings->SetSetting("sub_intervals", 1);
    settings->SetSetting("render_texture", "firejet_blast.png");
    settings->SetSetting("render_frag_shader", "sprite_tex_frag.glsl");
    settings->SetSetting("render_use_alpha", true);
    //settings->SetSetting("render_use_alpha", false);
    settings->SetSetting("render_alpha_function", "add");
    settings->SetSetting("lt_increment", -.00);
    settings->SetSetting("lt_cl", "lifetime.cl");

    ps = new rtps::RTPS(settings);
    //ps = new rtps::RTPS();


    ps->settings->SetSetting("Gravity", -9.8f); // -9.8 m/sec^2
    ps->settings->SetSetting("Gas Constant", 1.0f);
    ps->settings->SetSetting("Viscosity", .001f);
    ps->settings->SetSetting("Velocity Limit", 600.0f);
    ps->settings->SetSetting("XSPH Factor", .15f);
    ps->settings->SetSetting("Friction Kinetic", 0.0f);
    ps->settings->SetSetting("Friction Static", 0.0f);
    ps->settings->SetSetting("Boundary Stiffness", 20000.0f);
    ps->settings->SetSetting("Boundary Dampening", 256.0f);


    //initialize the OpenGL scene for rendering
    init_gl();

printf("about to start main loop\n");
    glutMainLoop();
    return 0;
}



void init_gl()
{
    // default initialization
    //glDisable(GL_DEPTH_TEST);

    // viewport
    glViewport(0, 0, window_width, window_height);

    // projection
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    //gluPerspective(60.0, (GLfloat)window_width / (GLfloat) window_height, 0.1, 100.0);
    //gluPerspective(fov, (GLfloat)window_width / (GLfloat) window_height, 0.3, 100.0);
    //gluPerspective(90.0, (GLfloat)window_width / (GLfloat) window_height, 0.1, 10000.0); //for lorentz

    // set view matrix
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glClearColor(.2, .2, .6, 1.0);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    ps->system->getRenderer()->setWindowDimensions(window_width,window_height);
    //glTranslatef(0, 10, 0);
    /*
    gluLookAt(  0,10,0,
                0,0,0,
                0,0,1);
    */


    //glTranslatef(0, translate_z, translate_y);
    //glRotatef(-90, 1.0, 0.0, 0.0);

    return;

}

void appKeyboard(unsigned char key, int x, int y)
{
    int nn;
    float4 min;
    float4 max;
    switch (key)
    {
        case 'e': //dam break
            {
                nn = 16384;
                min = float4(.1, .1, .1, 1.0f);
                max = float4(3.9, 3.9, 3.9, 1.0f);
                min = float4(.1, .1, .1, 1.0f);
                max = float4(4.9, 4.9, 4.9, 1.0f);
                //float4 color = float4(0.1, 0.1, 0.3, .01);
                ps->system->addBox(nn, min, max, false,color);
                return;
            }
        case 'p': //print timers
            ps->system->printTimers();
            return;
        case '\033': // escape quits
        case '\015': // Enter quits    
        case 'Q':    // Q quits
        case 'q':    // q (or escape) quits
            // Cleanup up and quit
            appDestroy();
            return;
        case 'b':
            printf("deleting willy nilly\n");
            ps->system->testDelete();
            return;
        case 'h':
        {
            //spray hose
            printf("about to make hose\n");
            float4 center(1., 2., 2., 1.);
            //float4 velocity(.6, -.6, -.6, 0);
            //float4 velocity(2., 5., -.8, 0);
            float4 velocity(2., .5, 2., 0);
            //sph sets spacing and multiplies by radius value
            //float4 color = float4(.0, 0.0, 1.0, 1.0);
            //float4 color = float4(0.1, 0.1, 0.3, .01);
            hindex = ps->system->addHose(5000, center, velocity, 4, color);
            return;
		}
        case 'H':
        {
            //spray hose
            printf("about to move hose\n");
            float4 center(.1, 2., 1., 1.);
            //float4 velocity(.6, -.6, -.6, 0);
            //float4 velocity(2., 5., -.8, 0);
            float4 velocity(2., -.5, -1., 0);
            //sph sets spacing and multiplies by radius value
            //float4 color = float4(.0, 0.0, 1.0, 1.0);
            //float4 color = float4(0.1, 0.1, 0.3, .01);
            ps->system->updateHose(hindex, center, velocity, 4, color);
            return;
		}

        case 'n':
            render_movie=!render_movie;
            break;
        case '`':
            stereo_enabled = !stereo_enabled;
            break;
        case 't': //place a cube for collision
            {
                nn = 512;
                float cw = .25;
                float4 cen = float4(cw, cw, cw-.1, 1.0f);
                make_cube(triangles, cen, cw);
                cen = float4(1+cw, 1+cw, cw-.1, 1.0f);
                make_cube(triangles, cen, cw);
                cen = float4(1+3*cw, 1+3*cw, cw-.1, 1.0f);
                make_cube(triangles, cen, cw);
                cen = float4(3.5, 3.5, cw-.1, 1.0f);
                make_cube(triangles, cen, cw);

                cen = float4(1.5, 1.5, cw-.1, 1.0f);
                make_cube(triangles, cen, 1.);

                ps->system->loadTriangles(triangles);
                return;
            }
        case 'r': //drop a rectangle
            {
                //nn = 20;
                nn = 2048;
                //test out of bounds (max)
                //with max_num = 8192 this will have 2 particles in bounds and 18 out
                min = float4(5.7, 5.7, 5.7, 1.0f);
                max = float4(6.5, 6.5, 6.5, 1.0f);

                //test negative bounds
                //with max_num = 8192 this will have 8 particles in bounds and 12 out
                min = float4(-1.5, -1.5, -1.0, 1.0f);
                max = float4(1.0, 1.0, 1.0, 1.0f);


                //min = float4(15.8, 15.8, 15.8, 1.0f);
                //max = float4(16.5, 16.5, 16.5, 1.0f);

                min = float4(1.2, 1.2, 3.2, 1.0f);
                max = float4(2., 2., 4., 1.0f);

				// release particles symmetrically wrt center 
                min = float4(2.0, 2.0, 3.2, 1.0f);
                max = float4(3.0, 3.0, 4.0, 1.0f);
                
                //float4 color = float4(rand()/(10.*RAND_MAX), rand()/(RAND_MAX+1.0), rand()/(RAND_MAX+1.0), 0.2);
                ps->system->addBox(nn, min, max, false, color);
                return;
            }

        case 'o':
            ps->system->getRenderer()->writeBuffersToDisk();
            return;
        case 'c':
            ps->system->getRenderer()->setDepthSmoothing(Render::NO_SHADER);
            return;
        case 'C':
            ps->system->getRenderer()->setDepthSmoothing(Render::BILATERAL_GAUSSIAN_SHADER);
            return;

        case 'w':
            translate_z -= 0.1;
            break;
        case 'a':
            translate_x += 0.1;
            break;
        case 's':
            translate_z += 0.1;
            break;
        case 'd':
            translate_x -= 0.1;
            break;

        case 'W':
            cloud_translate_z -= 0.1;
            cloud_velocity_z  += 0.1;
            break;
        case 'A':
			// move hand at constant velocity in x
            cloud_translate_x += 0.1;
            cloud_velocity_x  += 0.1;
            break;
        case 'S':
			// turn arm rotation on and off
			cloud_movement = cloud_movement ? false : true;
			printf("cloud_movement = %d\n", cloud_movement);
			ps->setCloudMovement(cloud_movement);
            //cloud_translate_z += 0.1;
            //cloud_velocity_z  += 0.1;
            break;
        case 'D':
            //cloud_translate_x -= 0.1;
            //cloud_velocity_x  -= 0.1;
            break;
        case 'Z':
            //cloud_translate_y += 0.1;
            //cloud_velocity_y  += 0.1;
            break;
        case 'X':
            //cloud_translate_y -= 0.1;
            //cloud_velocity_y  -= 0.1;
            break;

        case 'z':
            translate_y += 0.1;
            break;
        case 'x':
            translate_y -= 0.1;
            break;
        default:
            return;
    }

    glutPostRedisplay();
    // set view matrix
    /*glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glRotatef(-90, 1.0, 0.0, 0.0);
    glRotatef(rotate_x, 1.0, 0.0, 0.0);
    glRotatef(rotate_y, 0.0, 0.0, 1.0); //we switched around the axis so make this rotate_z
    glTranslatef(translate_x, translate_z, translate_y);*/
}

void timerCB(int ms)
{
    glutTimerFunc(ms, timerCB, ms);
	//printf("****** BEFORE ps->update ******\n");
    ps->update();
	//printf("****** AFTER ps->update ******\n");
    glutPostRedisplay();
}

//----------------------------------------------------------------------
void appRender()
{

    //ps->system->sprayHoses();

    glEnable(GL_DEPTH_TEST);
    if (stereo_enabled)
    {
        render_stereo();
    }
    else
    {
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        gluPerspective(fovy, aspect, nearZ, farZ);

        // set view matrix
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();
        glRotatef(-90, 1.0, 0.0, 0.0);
        glRotatef(rotate_x, 1.0, 0.0, 0.0);
        glRotatef(rotate_y, 0.0, 0.0, 1.0); //we switched around the axis so make this rotate_z
        glTranslatef(translate_x, translate_z, translate_y);
		printf("main: before ps->render()\n");
        ps->render();
        draw_collision_boxes();
        if(render_movie)
        {
            write_movie_frame("image");
        }

    }

    if(render_movie)
    {
        frame_counter++;
    }
    //showFPS(enjas->getFPS(), enjas->getReport());
    glutSwapBuffers();

    //glDisable(GL_DEPTH_TEST);
}

void appDestroy()
{

    delete ps;


    if (glutWindowHandle)glutDestroyWindow(glutWindowHandle);
    printf("about to exit!\n");

    exit(0);
}


//----------------------------------------------------------------------
void appMouse(int button, int state, int x, int y)
{
	// The shift must be down when I click the mouse. Then 
	// the shift is detected. 
    if (state == GLUT_DOWN)
    {
		// Only called when the mouse is pressed for the first time. 
        mouse_buttons |= 1<<button;
		//printf("mouse down\n");
		setShiftDown();
    }
    else if (state == GLUT_UP)
    {
		setShiftDown();
        mouse_buttons = 0;
    }

    mouse_old_x = x;
    mouse_old_y = y;

    //glutPostRedisplay();
}

//----------------------------------------------------------------------
void appMotion(int x, int y)
{
// If SHIFT was detected, then move arm. 
// HOW TO DETECTED WHEN SHIFT IS RELEASED? 

	if (shift_down) {
		printf("***** SHIFT DOWN ****\n");
		//cloud_movement();
		if (mouse_buttons & 1)
		{
			// Change sensitivity of mouse
			cloud_translate_x += 0.004 * (x - mouse_old_x);
			// so that mouse negative y moves downward on screen
			cloud_translate_z -= 0.004 * (y - mouse_old_y);
			ps->setCloudTranslate(cloud_translate_x, 
			   cloud_translate_y, cloud_translate_z);
		}
		printf("appMotion, shift down\n");
	} else if (ctrl_down) {
		printf("***** CTRL DOWN ****\n");
		mouse_old = float4(x,y, 0.,1.);
		mouse_new = float4(mouse_old_x,mouse_old_y, 0.,1.);
		float4 vec_old, vec_new;
		// 300: radius for mouse
		Vectorize(mouse_old, screen_center, 300, vec_old);
		Vectorize(mouse_new, screen_center, 300, vec_new);
		computeRotationAxis(vec_old, vec_new);
		rotation_axis.print("**** axis");
		ps->setRotationAxis(rotation_axis);
        //ZeroHysteresisRotation(vec_old, vec_new);
		// cloud rotation();
	} else {
		// domain manipulation
		printf("appMotion, shift up\n");
    	float dx, dy;
    	dx = x - mouse_old_x;
    	dy = y - mouse_old_y;

    	if (mouse_buttons & 1)
    	{
        	rotate_x += dy * 0.2;
        	rotate_y += dx * 0.2;
    	}
    	else if (mouse_buttons & 4)
    	{
        	translate_z -= dy * 0.1;
    	}
	}

    mouse_old_x = x;
    mouse_old_y = y;

    // set view matrix
    glutPostRedisplay();
}


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
    while (*str)
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
    drawString(ss.str().c_str(), 15, 286, color, font);
    drawString(report[0].c_str(), 15, 273, color, font);
    drawString(report[1].c_str(), 15, 260, color, font);

    // restore projection matrix
    glPopMatrix();                      // restore to previous projection matrix

    // restore modelview matrix
    glMatrixMode(GL_MODELVIEW);         // switch to modelview matrix
    glPopMatrix();                      // restore to previous modelview matrix
}
//----------------------------------------------------------------------
void resizeWindow(int w, int h)
{
    if (h==0)
    {
        h=1;
    }
    glViewport(0, 0, w, h);

    // projection
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    //gluPerspective(fov, aspect, nearZ, farZ);
    ps->system->getRenderer()->setWindowDimensions(w,h);
    window_width = w;
    window_height = h;
    delete[] image;
    image = new GLubyte[w*h*4];
    setFrustum();
    glutPostRedisplay();
}

void render_stereo()
{

    glDrawBuffer(GL_BACK_LEFT);                              //draw into back left buffer
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();                                        //reset projection matrix
    glFrustum(leftCam.leftfrustum, leftCam.rightfrustum,     //set left view frustum
              leftCam.bottomfrustum, leftCam.topfrustum,
              nearZ, farZ);
    glTranslatef(leftCam.modeltranslation, 0.0, 0.0);        //translate to cancel parallax
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glPushMatrix();
    {
        //glTranslatef(0.0, 0.0, depthZ);                        //translate to screenplane
        glRotatef(-90, 1.0, 0.0, 0.0);
        glRotatef(rotate_x, 1.0, 0.0, 0.0);
        glRotatef(rotate_y, 0.0, 0.0, 1.0); //we switched around the axis so make this rotate_z
        glTranslatef(translate_x, translate_z, translate_y);
        ps->render();
        draw_collision_boxes();
    }
    glPopMatrix();

    if(render_movie)
    {
        write_movie_frame("stereo/image_left_");
    }

    glDrawBuffer(GL_BACK_RIGHT);                             //draw into back right buffer
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();                                        //reset projection matrix
    glFrustum(rightCam.leftfrustum, rightCam.rightfrustum,   //set left view frustum
              rightCam.bottomfrustum, rightCam.topfrustum,
              nearZ, farZ);
    glTranslatef(rightCam.modeltranslation, 0.0, 0.0);       //translate to cancel parallax
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glPushMatrix();
    {
        glRotatef(-90, 1.0, 0.0, 0.0);
        glRotatef(rotate_x, 1.0, 0.0, 0.0);
        glRotatef(rotate_y, 0.0, 0.0, 1.0); //we switched around the axis so make this rotate_z
        glTranslatef(translate_x, translate_z, translate_y);
        ps->render();
        draw_collision_boxes();
    }
    glPopMatrix();
    if(render_movie)
    {
        write_movie_frame("stereo/image_right_");
    }
}


void setFrustum(void)
{
    double top = nearZ*tan(DTR*fovy/2);                    //sets top of frustum based on fovy and near clipping plane
    double right = aspect*top;                             //sets right of frustum based on aspect ratio
    double frustumshift = (IOD/2)*nearZ/screenZ;

    leftCam.topfrustum = top;
    leftCam.bottomfrustum = -top;
    leftCam.leftfrustum = -right + frustumshift;
    leftCam.rightfrustum = right + frustumshift;
    leftCam.modeltranslation = IOD/2;

    rightCam.topfrustum = top;
    rightCam.bottomfrustum = -top;
    rightCam.leftfrustum = -right - frustumshift;
    rightCam.rightfrustum = right - frustumshift;
    rightCam.modeltranslation = -IOD/2;
}

int write_movie_frame(const char* name)
{
        sprintf(filename,"%s%s_%08d.png",render_dir,name,frame_counter);
        glReadPixels(0, 0, window_width, window_height, GL_RGBA, GL_UNSIGNED_BYTE, image);
        if (!stbi_write_png(filename,window_width,window_height,4,(void*)image,0))
        {
            printf("failed to write image %s\n",filename);
            return -1;
        }
        return 0;
}
void rotate_img(GLubyte* img, int size)
{
    GLubyte tmp=0;
    for(int i = 0; i<size; i++)
    {
        for(int j = 0; j<4; j++)
        {
            tmp = img[(i*4)+j];
            img[(i*4)+j] = img[size-((i*4)+j)-1];
            img[size-((i*4)+j)-1] = tmp;
        }
    }
}

void draw_collision_boxes()
{
    glColor4f(0,0,1,.5);

    //glDisable(GL_DEPTH_TEST);
    //glDepthMask(GL_TRUE);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    glBegin(GL_TRIANGLES);
    //printf("num triangles %zd\n", triangles.size());
    for (int i=0; i < triangles.size(); i++)
    {
        //for (int i=0; i < 20; i++) {
        Triangle& tria = triangles[i];
        glNormal3fv(&tria.normal.x);
        glVertex3f(tria.verts[0].x, tria.verts[0].y, tria.verts[0].z);
        glVertex3f(tria.verts[1].x, tria.verts[1].y, tria.verts[1].z);
        glVertex3f(tria.verts[2].x, tria.verts[2].y, tria.verts[2].z);
    }
    glEnd();

    glDisable(GL_BLEND);
    //glEnable(GL_DEPTH_TEST);
    //glDepthMask(GL_TRUE);
}
//----------------------------------------------------------------------
void setShiftDown()
{
		int mod = glutGetModifiers();
		shift_down = mod == GLUT_ACTIVE_SHIFT ? true : false;
		ctrl_down = mod == GLUT_ACTIVE_CTRL ? true : false;

		//printf("ctrl: %d, shift: %d\n", ctrl_down, shift_down);

		//if (shift_down) exit(0);
		//if (ctrl_down) exit(0);
}
//----------------------------------------------------------------------
//Vectorize()
// Create a 3D position vector on a unit sphere based on
// mouse hitPt, sphere's screen origin, and sphere radius in
// pixels. Return zero on success or non-zero if hit point 
// was outside the sphere. This routine uses Quickdraw[TM] 3D
// and assumes that the relevant QD3D headers and library
// files have been included. Also need to #include <math.h>.

#define BOUNDS_ERR 1L

long Vectorize( 
   float4& hit, 
   float4& origin, 
   float radius, 
   float4& vec ) 
   //TQ3Vector3D *vec ) 
{
   float x,y,z, modulus;
   
   x = (float)(hit.x - origin.x)/radius;
   y = (float)(hit.y - origin.y)/radius;

   //hit.print("hit");
   //origin.print("origin");
   //printf("before x,y= %f, %f\n", x, y);

   y *= -1.0;         // compensate for "inverted" screen y-axis!
   
   modulus = sqrt(x*x + y*y);
   //printf("modulus= %f\n", modulus);
   //printf("radius(%f)/modulus= %f\n", radius, radius/modulus);
   
   if (modulus > .95) {    // outside radius!
      x /= (modulus+.06);  // avoid singularities
      y /= (modulus+.06);
      //return BOUNDS_ERR;
   }

   //printf("after x,y= %f, %f\n", x, y);
   //printf("x^2+y^2= %f\n", x*x+y*y);
   
   z = sqrt(1. - (x*x + y*y) );    // compute fictitious 'z' value
   vec = float4(x,y,z,1.);
   //vec.print("vec");
   
   return 0L;
}
//----------------------------------------------------------------------
//Rotate::ZeroHysteresisRotation()
// From two 3D vectors representing the positions of 
// points on a unit sphere, calculate an axis of rotation
// and an amount of rotation such that float4 A can be
// moved along a geodesic to float4 B.
// CAUTION: Error-checking omitted for clarity.

void computeRotationAxis(float4& v1, float4& v2) 
{
   float4           cross;
   //TQ3Matrix4x4     theMatrix;
   //TQ3float43D      orig = { 0.,0.,0. };
   float            dot,angle;
   
   dot = v1.x*v2.x + v1.y*v2.y + v1.z*v2.z;

   if (dot == 1.0) 
      return;                                        // nothing to do

   cross = float4(v1.y*v2.z-v1.z*v2.y, 
                  v1.z*v2.x-v1.x*v2.z,
                  v1.x*v2.y-v1.y*v2.x, 1.);

   //Q3Vector3D_Cross( &v1, &v2, &cross ); // axis of rotation

   // EVER ZERO?
   float normi = sqrt(1./(cross.x*cross.x+cross.y*cross.y+cross.z*cross.z));
   cross.x *= normi;
   cross.y *= normi;
   cross.z *= normi;
   cross.w  = 0.0;
   //Q3Vector3D_Normalize( &cross,&cross );   

   angle = 2.*acos( dot );                           // angle of rotation
   printf("angle= %f\n", angle);
   cross.w = angle;
   cross.print("cross");
   rotation_axis = cross;
   
   #if 0
   // set up a rotation around our chosen axis...
   Q3Matrix4x4_SetRotateAboutAxis(&theMatrix,
                     &orig, &cross, angle);
   
   Q3Matrix4x4_Multiply(   &gDocument.fRotation,
                     &theMatrix,
                     &gDocument.fRotation);     // multiply

   DocumentDraw( &gDocument ) ;                 // draw
   #endif
}
//----------------------------------------------------------------------
#if 0
//Rotate::FreeRotateWithMouse()
// Call this function from the main event loop to
// do free-rotations of 3D objects

// the mouse action radius in pixels:
#define RADIUS_VALUE 300.0

void FreeRotateWithMouse(void) 
{
   float4             now, oldPt, center;
   WindowPtr       win = FrontWindow();
   float            radius = RADIUS_VALUE;
   TQ3Vector3D   v1,v2;
   long               err;

   GetMouse( &oldPt );

   center.h = (win->portRect.right - win->portRect.left)/2;
   center.v = (win->portRect.bottom - win->portRect.top)/2;

   while (StillDown()) 
   {
      GetMouse( &now );

      err = Vectorize(&oldPt, &center, RADIUS_VALUE, &v1 );
      err += Vectorize(&now, &center, RADIUS_VALUE, &v2 );

      if (!err)
         getRotationAxis( v1,v2 );

      oldPt = now;
   }
}
#endif
//----------------------------------------------------------------------
//----------------------------------------------------------------------
//----------------------------------------------------------------------

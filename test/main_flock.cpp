/****************************************************************************************
* Real-Time Particle System - An OpenCL based Particle system developed to run on modern GPUs. Includes SPH fluid simulations.
* version 1.0, September 14th 2011
* 
* Copyright (C) 2011 Ian Johnson, Andrew Young, Gordon Erlebacher, Myrna Merced, Evan Bollig
* 
* This software is provided 'as-is', without any express or implied
* warranty.  In no event will the authors be held liable for any damages
* arising from the use of this software.
* 
* Permission is granted to anyone to use this software for any purpose,
* including commercial applications, and to alter it and redistribute it
* freely, subject to the following restrictions:
* 
* 1. The origin of this software must not be misrepresented; you must not
* claim that you wrote the original software. If you use this software
* in a product, an acknowledgment in the product documentation would be
* appreciated but is not required.
* 2. Altered source versions must be plainly marked as such, and must not be
* misrepresented as being the original software.
* 3. This notice may not be removed or altered from any source distribution.
****************************************************************************************/


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

int window_width = 640;
int window_height = 640;
int glutWindowHandle = 0;

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

float translate_x = -2.5f;  //-2.00f;
float translate_y = -2.5f;  //-2.70f;//300.f;
float translate_z =  4.0f;  //3.50f;

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

void timerCB(int ms);

void drawString(const char *str, int x, int y, float color[4], void *font);
void showFPS(float fps, std::string *report);

int write_movie_frame(const char* name);
void draw_collision_boxes();
void rotate_img(GLubyte* img, int size);

void *font = GLUT_BITMAP_8_BY_13;

rtps::RTPS* ps;

//#define NUM_PARTICLES 524288
//#define NUM_PARTICLES 262144
//#define NUM_PARTICLES 65536
//#define NUM_PARTICLES 16384
//#define NUM_PARTICLES 10000
#define NUM_PARTICLES 8192
//#define NUM_PARTICLES 4096
//#define NUM_PARTICLES 2048
//#define NUM_PARTICLES 1024
//#define NUM_PARTICLES 256

#define DT              0.001f

#define maxspeed        2.0f
#define mindist         1.0f
#define searchradius    1.50f

float4 color  = float4(255.f, 0.f, 0.f, 0.f);
float4 target = float4(3.,2.,4.,1.);
int hindex;
//----------------------------------------------------------------------
float rand_float(float mn, float mx)
{
    float r = rand() / (float) RAND_MAX;
    return mn + (mx-mn)*r;
}
//----------------------------------------------------------------------
int main(int argc, char** argv)
{
    //initialize glut
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_ALPHA | GLUT_DOUBLE | GLUT_DEPTH);
    glutInitWindowSize(window_width, window_height);
    glutInitWindowPosition (glutGet(GLUT_SCREEN_WIDTH)/2 - window_width/2, 
                            glutGet(GLUT_SCREEN_HEIGHT)/2 - window_height/2);


    std::stringstream ss;
    ss << "Real-Time Particle System: " << NUM_PARTICLES << std::ends;
    glutWindowHandle = glutCreateWindow(ss.str().c_str());

    glutDisplayFunc(appRender); //main rendering function
    glutTimerFunc(30, timerCB, 30); //determin a minimum time between frames
    glutKeyboardFunc(appKeyboard);
    glutMouseFunc(appMouse);
    glutMotionFunc(appMotion);
    glutReshapeFunc(resizeWindow);

    //define_lights_and_materials();

    // initialize necessary OpenGL extensions
    glewInit();
    GLboolean bGLEW = glewIsSupported("GL_VERSION_2_0 GL_ARB_pixel_buffer_object"); 
    printf("GLEW supported?: %d\n", bGLEW);


    printf("before we call enjas functions\n");


    float w_sep = .50f;     //15
    float w_align = .0f;   //7.5
    float w_coh = 0.0f;     //2.5

    float w_goal = 1.0f;
    float w_avoid = 0.0f;
    float w_wander = 0.0f;

    float w_leadfoll = 0.f;
   
    float ang_vel = 0.f; 
    float slow_dist = .0f;

    bool two_dimension = false;

    //default constructor
    rtps::Domain* grid = new Domain(float4(0,0,0,0), float4(5, 5, 5, 0));
    //rtps::RTPSettings* settings(rtps::RTPSettings::FLOCK, NUM_PARTICLES, DT, grid, maxspeed, mindist, searchradius, color, w_sep, w_align, w_coh);
    rtps::RTPSettings* settings = new rtps::RTPSettings(rtps::RTPSettings::FLOCK, NUM_PARTICLES, DT, grid);

#ifdef WIN32
    settings->SetSetting("rtps_path", ".");
#else
    settings->SetSetting("rtps_path", "./bin");
    //settings->SetSetting("rtps_path", argv[0]);
    //printf("arvg[0]: %s\n", argv[0]);
#endif

   // settings->setRenderType(RTPSettings::RENDER);
    //settings->setRenderType(RTPSettings::SPRITE_RENDER);
    settings->setRenderType(RTPSettings::SPHERE3D_RENDER);
    
    settings->setRadiusScale(1.0);
    settings->setBlurScale(1.0);
    settings->setUseGLSL(1);

    settings->setTarget(target);
    settings->setDimension(two_dimension);
    settings->SetSetting("render_texture", "nemo.png");
    settings->SetSetting("render_frag_shader", "boid_tex_frag.glsl");
    //settings->SetSetting("render_use_alpha", true);
    settings->SetSetting("render_use_alpha", false);
    //settings->SetSetting("render_alpha_function", "add");
    settings->SetSetting("lt_increment", -.00);
    settings->SetSetting("lt_cl", "lifetime.cl");

    ps = new rtps::RTPS(settings);

    ps->settings->SetSetting("Max Speed", maxspeed);
    ps->settings->SetSetting("Min Separation Distance", mindist);
    ps->settings->SetSetting("Searching Radius", searchradius);
    ps->settings->SetSetting("Angular Velocity", ang_vel);
    ps->settings->SetSetting("Slowing Distance", slow_dist);

    ps->settings->SetSetting("Separation Weight", w_sep);
    ps->settings->SetSetting("Alignment Weight", w_align);
    ps->settings->SetSetting("Cohesion Weight", w_coh);
    ps->settings->SetSetting("Goal Weight", w_goal);
    ps->settings->SetSetting("Avoid Weight", w_avoid);
    ps->settings->SetSetting("LeaderFollowing Weight", w_leadfoll);

    //initialize the OpenGL scene for rendering
    init_gl();
    
    printf("about to start mainloop\n");
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
    glClearColor(1.0, 1.0, 1.0, 1.0);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    //glRotatef(-90, 1.0, 0.0, 0.0);
    //glRotatef(rotate_x, 1.0, 0.0, 0.0);
    //glRotatef(rotate_y, 0.0, 0.0, 1.0); //we switched around the axis so make this rotate_z
    //glTranslatef(translate_x, translate_z, translate_y);
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
    float4 center;
    float radius;

    switch (key)
    {
        case 'e': //dam break
            nn = 16384;
            min = float4(.1, .1, .1, 1.0f);
            max = float4(3.9, 3.9, 3.9, 1.0f);
            ps->system->addBox(nn, min, max, false, color);
            return;
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
                float cw = .1;
                //float4 cen = float4(3., 1., 4., 1.0f);
                make_cube(triangles, target, cw);
                //cen = float4(1+cw, 1+cw, cw-.1, 1.0f);
                //make_cube(triangles, cen, cw);
                //cen = float4(1+3*cw, 1+3*cw, cw-.1, 1.0f);
                //make_cube(triangles, cen, cw);
                //cen = float4(3.5, 3.5, cw-.1, 1.0f);
                //make_cube(triangles, cen, cw);
                
                //cen = float4(1.5, 1.5, cw-.1, 1.f);
                //make_cube(triangles, cen, 1.);
                
                ps->system->loadTriangles(triangles);
                return;
            }

        case 'r': //drop a rectangle
        {
            //nn = 65536;
	        //nn = 1024;
            nn = 8192;
            //nn = 256;
            
            // timings input cube
            //max = float4(4.5, 4.5 , 4.5, 1.0f);
            //min = float4(0.5, 0.5 , 0.5, 1.0f);
            
            // symmetry demo input square
            //max = float4(3.5, 1., 3.5, 1.0f);
            //min = float4(1.5, 1., 1.5, 1.0f);
            
            // bees demo input cube
            max = float4(4., 2, 2, 1.0f);
            min = float4(1., 1, 1, 1.0f);
            
            ps->system->addBox(nn, min, max, false, color);
            return;
        }

        case 'k': //add a sphere
        {
            nn = 1000;
            center = float4(2.5f, 2.5f, 2.5f, 1.0f);
            radius = 1.f;
            printf("added a spehere at: %f %f %f with radius %f\n", center.x, center.y, center.z, radius);
            ps->system->addBall(nn, center, radius, false);
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
    glTranslatef(translate_x, translate_z, translate_y);
    */
}

void timerCB(int ms)
{
    glutTimerFunc(ms, timerCB, ms);
    ps->update();
    glutPostRedisplay();
}

void appRender()
{

    //ps->system->sprayHoses();
    GLfloat *ambient, *diffuse, *specular, *position;

    ambient = new GLfloat[4];   //(.9,.9,.9,1.);
    position = new GLfloat[4];  //(5,1,3,1);
    diffuse = new GLfloat[4];
    specular = new GLfloat[4];

    ambient[0] = 1.0f;   diffuse[0] = 1.f;  specular[0] = .0f;  position[0] = 3.50f;
    ambient[1] = 1.0f;   diffuse[1] = .0f;  specular[1] = .0f;  position[1] = 0.0f;
    ambient[2] = 1.0f;   diffuse[2] = .0f;  specular[2] = 1.f;  position[2] = 1.50f;
    ambient[3] = 1.0f;   diffuse[3] = 1.f;  specular[3] = 1.f;  position[3] = 1.f;

    glEnable(GL_LIGHT0);
    glLightfv(GL_LIGHT0, GL_AMBIENT, ambient); 
    glLightfv(GL_LIGHT0, GL_DIFFUSE, diffuse); 
    glLightfv(GL_LIGHT0, GL_SPECULAR, specular); 
    glLightfv(GL_LIGHT0, GL_POSITION, position); 


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

void appMouse(int button, int state, int x, int y)
{
    if (state == GLUT_DOWN)
    {
        mouse_buttons |= 1<<button;
    }
    else if (state == GLUT_UP)
    {
        mouse_buttons = 0;
    }

    mouse_old_x = x;
    mouse_old_y = y;

    //glutPostRedisplay();
}

void appMotion(int x, int y)
{
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

    //glDepthMask(GL_FALSE);
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
}

/*
 *  RTPS
 *  Ian Johnson
 *  jni-opengl stuff based on google's DemoActivity san-angelas
 *
 *
 *  some particles inspiration from: http://www.naturewizard.com/tutorial08.html
 */


#include <jni.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <time.h>
#include <android/log.h>
#include <assert.h>

#include "importgl.h"

#include "app.h"

#include "rtpslib/RTPS.h"

//everybody get raaaandom

//keep track of our viewport size
static int width;
static int height;

//rotation stuff
static float fingerX = 0;
static float fingerY = 0;
static float cameraAngleX = 0;
static float cameraAngleY = 0;


//5 second particle length
#define RUN_LENGTH  (5 * 1000)
#undef PI
#define PI 3.1415926535897932f

#define NUM_PARTICLES 100
#define DT .001f

rtps::RTPS* ps;

static long sStartTick = 0;
static long sTick = 0;

//returns a random float between 0 and 1
float randf()
{
    //random hack since no floating point random function
    //optimize later
    return (lrand48() % 255) / 255.f;
}


void appTouch(float* xp, float* yp)
{
    float x = *xp;
    float y = *yp;
    //__android_log_print(ANDROID_LOG_INFO, "RTPS", "touch!");

    //__android_log_print(ANDROID_LOG_INFO, "RTPS", "new verts! x=%g y=%f", x, y);
    x = 2.*(x/width) - 1.;
    y = 1. - 2.*(y/height);
    //__android_log_print(ANDROID_LOG_INFO, "RTPS", "new verts scaled x=%f y=%f", x, y);
    //make_vertices(vertices, NUM_PARTICLES, x, y);
    //init_generator(x,y);

}
//when someone touches we need first place they touched
void appDown(float* xp, float* yp)
{
    float x = *xp;
    float y = *yp;
    fingerX = x;
    fingerY = y;
}
//when they move the mouse they change the camera angle
void appMove(float* xp, float* yp)
{
    float x = *xp;
    float y = *yp;
    cameraAngleY += (x -fingerX);
    cameraAngleX += (y -fingerY);
    fingerX = x;
    fingerY = y;
    //__android_log_print(ANDROID_LOG_INFO, "RTPS", "camera angle! x=%f y=%f", cameraAngleX, cameraAngleY);
}
// Called from the app framework.
void appInit()
{
    srand48(time(NULL));
    rtps::RTPSettings settings(rtps::RTPSettings::SPH, NUM_PARTICLES, DT);
    //rtps::RTPSettings settings(rtps::RTPSettings::Simple, NUM_PARTICLES, DT);
    ps = new rtps::RTPS(settings);
    
    __android_log_print(ANDROID_LOG_INFO, "RTPS", "RTPS INITIALIZED");
    //__android_log_print(ANDROID_LOG_INFO, "RTPS", "can i haz init vertices[0]=%f generator[0].x=%f", vertices[0], generator[0].x);
    //__android_log_print(ANDROID_LOG_INFO, "RTPS", "can i haz init vertices[3]=%f generator[1].x=%f", vertices[3], generator[1].x);
    //__android_log_print(ANDROID_LOG_INFO, "RTPS", "can i haz init alpha[0]=%f life[0]=%f colors[3]=%f", alpha[0], life[0], colors[3]);
}


// Called from the app framework.
void appDeinit()
{
}

static void gluPerspective(GLfloat fovy, GLfloat aspect,
                           GLfloat zNear, GLfloat zFar)
{
    GLfloat xmin, xmax, ymin, ymax;

    ymax = zNear * (GLfloat)tan(fovy * PI / 360);
    ymin = -ymax;
    xmin = ymin * aspect;
    xmax = ymax * aspect;

    glFrustumx((GLfixed)(xmin * 65536), (GLfixed)(xmax * 65536),
               (GLfixed)(ymin * 65536), (GLfixed)(ymax * 65536),
               (GLfixed)(zNear * 65536), (GLfixed)(zFar * 65536));
}


static void prepareFrame()
{
    glViewport(0, 0, width, height);
    glClearColor(0.0, 0.0, 0.0, 0.0);
    glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();

    //rotation works but wont be nice till we draw a box
    gluPerspective(45, (float)width / height, 0.5f, 1500);
    //glRotatef(-90, 1, 0, 0);
    glRotatef(cameraAngleX, 1, 0, 0);   // pitch
    glRotatef(cameraAngleY, 0, 0, 1);   // heading
    //__android_log_print(ANDROID_LOG_INFO, "RTPS", "rotate? x=%f y=%f", cameraAngleX, cameraAngleY);


    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    /*
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(45, (float)width / height, 0.5f, 150);

    glMatrixMode(GL_MODELVIEW);

    glLoadIdentity();
    */
}


// Called from the app framework.
/* The tick is current time in milliseconds, width and height
 * are the image dimensions to be rendered.
 */
void appRender(long tick, int w, int h)
{
    width = w;
    height = h;
    if (sStartTick == 0)
        sStartTick = tick;
    if (!gAppAlive)
        return;

    // Actual tick value is "blurred" a little bit.
    sTick = (sTick + tick - sStartTick) >> 1;

    // Terminate application after running through the demonstration once.
    if (sTick >= RUN_LENGTH)
    {
        //gAppAlive = 0;
        //return;
        sTick = 0;
        sStartTick = 0;
    }

    //__android_log_print(ANDROID_LOG_INFO, "RTPS", "sTick=%ld", sTick);

    //update particles
    ///update_particles(sTick);
    ps->update();

    prepareFrame();


    ps->render();
    /*
    //draw some particles!
    //glColor4f(1., 1., 1., 1.);
    glVertexPointer(3, GL_FLOAT, 0, vertices);
    glColorPointer(4, GL_FLOAT, 0, colors);

    // Already done in initialization:
    //glEnableClientState(GL_VERTEX_ARRAY);
    //glEnableClientState(GL_COLOR_ARRAY);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    
    glEnable(GL_POINT_SMOOTH); //wonder how much this costs
    glPointSize(10.);
   
    //glDrawArrays(GL_POINTS, 0, NUM_PARTICLES);
    glDrawElements(GL_POINTS, NUM_PARTICLES, GL_UNSIGNED_SHORT, elements);
    
    glDisable(GL_POINT_SMOOTH);
    glDisable(GL_BLEND);

    */

}

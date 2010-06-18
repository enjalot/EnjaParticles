/*
 *  EnjaParticles
 *  Ian Johnson
 *
 */


#include <jni.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <android/log.h>
#include <assert.h>

#include "importgl.h"

#include "app.h"

//5 second particle length
#define RUN_LENGTH  (5 * 1000)
#undef PI
#define PI 3.1415926535897932f

#define NUM_PARTICLES 20

static long sStartTick = 0;
static long sTick = 0;

static GLfloat vertices[NUM_PARTICLES * 3];
static GLfloat colors[NUM_PARTICLES * 3];

//keep track of our viewport size
static int width;
static int height;

//rotation stuff
static float fingerX = 0;
static float fingerY = 0;
static float cameraAngleX = 0;
static float cameraAngleY = 0;


void appTest(int a, float* b)
{
    __android_log_print(ANDROID_LOG_INFO, "EnjaParticles", "app test a=%d b=%f", a, *b);
}


void make_vertices(GLfloat* verts, int num, float x, float y)
{   
    //assuming 3D array
    int i = 0;
    float f = 0.;
    int nums = num * 3;
    for(i=0;i<nums;i+=3)
    {
        f = i/3.f;
        verts[i] = x + .5*cos(2.*PI*(f/num));
        //cout << verts[i] << endl;
        verts[i+1] = y + .5*sin(2.*PI*(f/num));
        verts[i+2] = 0.;// + f/nums;
        //__android_log_print(ANDROID_LOG_INFO, "EnjaParticles", "particle i=%d f/nums=%f", i, f/nums);
    }
}

void make_colors(GLfloat* cols, int num)
{   
    int i = 0;
    //float f = 0.;
    int nums = num * 3;
    for(i=0;i<nums;i+=3)
    {
        //float f = i;
        cols[i] = 1.;
        cols[i+1] = 1.;
        cols[i+2] = 1.;
    }
}   

void appTouch(float* xp, float* yp)
{
    float x = *xp;
    float y = *yp;
    //__android_log_print(ANDROID_LOG_INFO, "EnjaParticles", "touch!");

    //__android_log_print(ANDROID_LOG_INFO, "EnjaParticles", "new verts! x=%g y=%f", x, y);
    x = 2.*(x/width) - 1.;
    y = 1. - 2.*(y/height);
    //__android_log_print(ANDROID_LOG_INFO, "EnjaParticles", "new verts scaled x=%f y=%f", x, y);
    make_vertices(vertices, NUM_PARTICLES, x, y);    

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
    __android_log_print(ANDROID_LOG_INFO, "EnjaParticles", "camera angle! x=%f y=%f", cameraAngleX, cameraAngleY);
}
// Called from the app framework.
void appInit()
{
    int a;

    /*
    glEnable(GL_NORMALIZE);
    glEnable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);
    glShadeModel(GL_FLAT);

    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);
    glEnable(GL_LIGHT1);
    glEnable(GL_LIGHT2);
    */

    glEnableClientState(GL_VERTEX_ARRAY);
    //glEnableClientState(GL_COLOR_ARRAY);
    //assert(sGroundPlane != NULL);

    
    make_vertices(vertices, NUM_PARTICLES, 0., 0.);

}


// Called from the app framework.
void appDeinit()
{
    /*
    int a;
    for (a = 0; a < SUPERSHAPE_COUNT; ++a)
        freeGLObject(sSuperShapeObjects[a]);
    freeGLObject(sGroundPlane);
    */
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

    /*
    glClearColorx((GLfixed)(0.1f * 65536),
                  (GLfixed)(0.2f * 65536),
                  (GLfixed)(0.3f * 65536), 0x10000);
    */
    glClearColor(0.0, 0.0, 0.0, 1.0);
    glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    //make our viewport go from 0,0 at bottom left to 1,1 at top right
    //glTranslatef(-1., -1., 0.);
    //glScalef(2,2,0.);

    //rotation works but wont be nice till we draw a box
    glRotatef(cameraAngleX, 1, 0, 0);   // pitch
    glRotatef(cameraAngleY, 0, 1, 0);   // heading
    //__android_log_print(ANDROID_LOG_INFO, "EnjaParticles", "rotate? x=%f y=%f", cameraAngleX, cameraAngleY);


    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    //glTranslatef(0,-.5,-1);

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
    float col;
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
    }

    //col = 1.f + sin(sTick / 100.f)/2.f;
    col = 0.;
    // Prepare OpenGL ES for rendering of the frame.
    //prepareFrame(col, width, height);
    
    //update particles
    
    prepareFrame();

    /*
    vertices[0] = 0.;
    vertices[1] = 0.;
    vertices[2] = 0.;
    vertices[3] = 1.;
    vertices[4] = 0.;
    vertices[5] = 0.;
    vertices[6] = 0.;
    vertices[7] = 1.;
    vertices[8] = 0.;
    vertices[9] = 1.;
    vertices[10] = 1.;
    vertices[11] = 0.;
    */
    //draw some particles!
    glColor4f(1., 1., 1., 1.);
    glVertexPointer(3, GL_FLOAT, 0, vertices);
    //glColorPointer(NUM_PARTICLES, GL_FLOAT, 0, colors);

    // Already done in initialization:
    //glEnableClientState(GL_VERTEX_ARRAY);
    //glEnableClientState(GL_COLOR_ARRAY);
    glPointSize(5);
    glDrawArrays(GL_POINTS, 0, NUM_PARTICLES);
    //glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);


    // Update the camera position and set the lookat.
    //camTrack();

    // Configure environment.
    //configureLightAndMaterial();

    // Draw the reflection by drawing models with negated Z-axis.
    //glPushMatrix();
    //drawModels(-1);
    //glPopMatrix();

    // Blend the ground plane to the window.
    //drawGroundPlane();

    // Draw all the models normally.
    //drawModels(1);

    // Draw fade quad over whole window (when changing cameras).
    //drawFadeQuad();
}

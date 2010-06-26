/*
 *  EnjaParticles
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
#include <android/log.h>
#include <assert.h>

#include "importgl.h"

#include "app.h"

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

static long sStartTick = 0;
static long sTick = 0;

static GLfloat vertices[NUM_PARTICLES * 3];
static GLfloat colors[NUM_PARTICLES * 4];
static GLushort elements[NUM_PARTICLES];

typedef struct
{
    float x;
    float y;
    float z;
} Vec3;

static Vec3 generator[NUM_PARTICLES];   //keep track of generator (origin) for each particle
static Vec3 velocity[NUM_PARTICLES];    //keep track of velocity vector for each particle
static float alpha[NUM_PARTICLES];      //keep track of alpha for display
static float life[NUM_PARTICLES];       //keep track of life of particle

//returns a random float between 0 and 1
float randf()
{
    //random hack since no floating point random function
    //optimize later
    return (lrand48() % 255) / 255.f;
}

void appTest(int a, float* b)
{
    __android_log_print(ANDROID_LOG_INFO, "EnjaParticles", "app test a=%d b=%f", a, *b);
}

void init_elements()
{
    int i = 0;
    for(i=0;i<NUM_PARTICLES;i++)
    {
        elements[i] = i;
    }
}

//void make_vertices(GLfloat* verts, int num, float x, float y)
void init_generator(float x, float y)
{   
    //assuming 3D array
    int i = 0;
    float f = 0.;
    //int nums = num * 3;
    //for(i=0;i<nums;i+=3)
    for(i=0;i<NUM_PARTICLES;i++)
    {
        f = i;
        generator[i].x = x + .01*cos(2.*PI*(f/NUM_PARTICLES));
        generator[i].y = y + .01*sin(2.*PI*(f/NUM_PARTICLES));
        generator[i].z = 0.;// + f/nums;
        /*
        f = i/3.f;
        verts[i] = x + .1*cos(2.*PI*(f/num));
        verts[i+1] = y + .1*sin(2.*PI*(f/num));
        verts[i+2] = 0.;// + f/nums;
        */
        //cout << verts[i] << endl;
        //__android_log_print(ANDROID_LOG_INFO, "EnjaParticles", "particle i=%d f/nums=%f", i, f/nums);
    }
}

void reset_vertex(int index)
{
    int i = index * 3;
    vertices[i] = generator[index].x;
    vertices[i + 1] = generator[index].y;
    vertices[i + 2] = generator[index].z;
}

void init_vertices()
{
    int i = 0;
    for(i=0;i<NUM_PARTICLES;i++)
    {
        reset_vertex(i);
    }
}

void init_colors(GLfloat* cols, int num)
{   
    //scale the tick to get parameter from [0,1.]
    float st = sTick / RUN_LENGTH;
    int i = 0;
    int k = 0; //particle index
    //float f = 0.;
    int nums = num * 4;
    for(i=0;i<nums;i+=4)
    {
        //float f = i;
        cols[i] = 1.;
        cols[i+1] = 1.;
        cols[i+2] = 1.;
        cols[i+3] = alpha[k];
        k+= 1;
    }
}   
void random_velocity(int index)
{
    velocity[index].x = .01 * (1. - 2.*randf()); // between -.02 and .02
    velocity[index].y = .05 * randf();
    velocity[index].z = .01 * (1. - 2.*randf());
}
void init_velocity()
{
    int i=0;
    for(i=0;i<NUM_PARTICLES;i++)
    {
        random_velocity(i);
    }
}

void init_life()
{
    int i=0;
    for(i=0;i<NUM_PARTICLES;i++)
    {
        //each particle starts with some random amount of visibility
        life[i] = randf();
       //__android_log_print(ANDROID_LOG_INFO, "EnjaParticles", "can i haz random %f", drand48()); //there is no drand
    }
}

void init_alpha()
{
    //memcpy(life, alpha, NUM_PARTICLES);
    //memset(alpha, 1.0f, NUM_PARTICLES);
    int i = 0;
    for(i=0;i<NUM_PARTICLES;i++)
    {
        alpha[i] = 1.f;
    }
}

void reset_particle(int index)
{
    //life and alpha are cyclic
    //right now we only need to reset the vertex to the generator
    reset_vertex(index);
    //may want to do other things here later
    random_velocity(index);
}

void update_vertex(int index, int tick)
{
    int i = index * 3;
    vertices[i] += velocity[index].x;
    vertices[i+1] += velocity[index].y;
    vertices[i+2] += velocity[index].z;
    velocity[index].y -= .0007; //this needs to depend on time or life
}

void update_color(int index, int tick)   
{
    //simple change in color
    int i = index * 4;
    colors[i] = 1.;
    colors[i+1] = life[i];
    colors[i+2] = life[i];
    colors[i+3] = 1-life[i];
}

void update_particles(long tick)
{
    int i = 0; //particle index
    for(i=0;i<NUM_PARTICLES;i++)
    {
        //update the lifetime
        life[i] -= .01;    //should probably depend on time somehow
        if(life[i] <= 0.)
        {
            //reset this particle
            reset_particle(i);
            life[i] = 1.;
        }    
        //alpha[i] = life[i];
        update_color(i, tick);
        update_vertex(i, tick);
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
    //make_vertices(vertices, NUM_PARTICLES, x, y);
    init_generator(x,y);

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
    //__android_log_print(ANDROID_LOG_INFO, "EnjaParticles", "camera angle! x=%f y=%f", cameraAngleX, cameraAngleY);
}
// Called from the app framework.
void appInit()
{
    srand48(time(NULL));
    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_COLOR_ARRAY);
    
    //make_vertices(vertices, NUM_PARTICLES, 0., 0.);
    init_elements();
    //physics stuff
    init_life();
    init_alpha();
    init_velocity();
    init_generator(0.,0.);
    //rendering objects
    init_vertices();
    init_colors(colors, NUM_PARTICLES);
    
    //__android_log_print(ANDROID_LOG_INFO, "EnjaParticles", "can i haz init vertices[0]=%f generator[0].x=%f", vertices[0], generator[0].x);
    //__android_log_print(ANDROID_LOG_INFO, "EnjaParticles", "can i haz init vertices[3]=%f generator[1].x=%f", vertices[3], generator[1].x);
    //__android_log_print(ANDROID_LOG_INFO, "EnjaParticles", "can i haz init alpha[0]=%f life[0]=%f colors[3]=%f", alpha[0], life[0], colors[3]);
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
    glRotatef(cameraAngleX, 1, 0, 0);   // pitch
    glRotatef(cameraAngleY, 0, 1, 0);   // heading
    //__android_log_print(ANDROID_LOG_INFO, "EnjaParticles", "rotate? x=%f y=%f", cameraAngleX, cameraAngleY);


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

    //__android_log_print(ANDROID_LOG_INFO, "EnjaParticles", "sTick=%ld", sTick);

    //update particles
    update_particles(sTick);

    prepareFrame();

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
}

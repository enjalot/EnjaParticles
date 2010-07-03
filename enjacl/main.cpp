#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

#include <string.h>
#include <string>

#include <GL/glew.h>
#if defined __APPLE__ || defined(MACOSX)
    #include <GLUT/glut.h>
#else
    #include <GL/glut.h>
    //OpenCL stuff
#endif

#include "enja.h"
#include "timege.h"

int window_width = 400;
int window_height = 300;
int glutWindowHandle = 0;
float translate_z = -30.f;

void init_gl();

void appKeyboard(unsigned char key, int x, int y);
void appRender();
void appDestroy();

EnjaParticles* enjas;
int NUM_PARTICLES;

GLuint v_vbo; //vbo id
GLuint c_vbo; //vbo id

//timers
GE::Time *ts[3];

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
    gluPerspective(90.0, (GLfloat)window_width / (GLfloat) window_height, 0.1, 10000.0); //for lorentz

    // set view matrix
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    //glTranslatef(0.0, 0.0, translate_z);
    glTranslatef(0.0, 0.0, -100.f);    //for lorentz
/*
    glRotatef(rotate_x, 1.0, 0.0, 0.0);
    glRotatef(rotate_y, 0.0, 1.0, 0.0);
*/
    return;

}

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

void appRender()
{
    //update the buffers with new vertices and colors

	ts[2]->start();

    ts[0]->start();
    enjas->update(.001);
    ts[0]->end();

    ts[1]->start();
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    //printf("render!\n");
    
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_POINT_SMOOTH); 
    
    glBindBuffer(GL_ARRAY_BUFFER, c_vbo);
    glColorPointer(4, GL_FLOAT, 0, 0);

    glBindBuffer(GL_ARRAY_BUFFER, v_vbo);
    glVertexPointer(4, GL_FLOAT, 0, 0);

    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_COLOR_ARRAY);

    glColor3f(0,1,0);
    glPointSize(10.);
    glDrawArrays(GL_POINTS, 0, NUM_PARTICLES);

    glDisableClientState(GL_COLOR_ARRAY);
    glDisableClientState(GL_VERTEX_ARRAY);
    glDisable(GL_POINT_SMOOTH);
    glDisable(GL_BLEND);

    glutSwapBuffers();
    ts[1]->end();
//    glutPostRedisplay();

	ts[2]->end();
}

void appDestroy()
{

    delete enjas;
    if(glutWindowHandle)glutDestroyWindow(glutWindowHandle);

    ts[0]->print();
    ts[1]->print();
    ts[2]->print();
    delete ts[0];
    delete ts[1];
    delete ts[2];

    exit(0);
}

void timerCB(int ms)
{
    glutTimerFunc(ms, timerCB, ms);
    glutPostRedisplay();
}

int main(int argc, char** argv)
{

    //initialize glut
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);
    glutInitWindowSize(window_width, window_height);
    glutInitWindowPosition (glutGet(GLUT_SCREEN_WIDTH)/2 - window_width/2, 
                            glutGet(GLUT_SCREEN_HEIGHT)/2 - window_height/2);

    glutWindowHandle = glutCreateWindow("EnjaParticles");

    glutDisplayFunc(appRender); //main rendering function
    glutTimerFunc(30, timerCB, 30);
    glutKeyboardFunc(appKeyboard);
/*
    glutMouseFunc(appMouse);
    glutMotionFunc(appMotion);
*/

    // initialize necessary OpenGL extensions
    glewInit();
    GLboolean bGLEW = glewIsSupported("GL_VERSION_2_0 GL_ARB_pixel_buffer_object"); 
    printf("GLEW supported?: %d\n", bGLEW);

    //initialize the OpenGL scene for rendering
    init_gl();

    printf("before we call enjas functions\n");

    enjas = new EnjaParticles();
    v_vbo = enjas->getVertexVBO();
    c_vbo = enjas->getColorVBO();
    NUM_PARTICLES = enjas->getNum();

    ts[0] = new GE::Time("update", 5);
    ts[1] = new GE::Time("render", 5);
    ts[2] = new GE::Time("appRender", 5);

    glutMainLoop();
    
    printf("doesn't happen does it\n");
    appDestroy();
    return 0;
}



























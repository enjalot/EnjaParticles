#include<stdio.h>

#include <GL/glew.h>
#if defined __APPLE__ || defined(MACOSX)
    #include <GLUT/glut.h>
    #include <OpenGL/glu.h>
#else
    #include <GL/glut.h>
    #include <GL/glu.h>
#endif

#include "RTPS.h"

// window size
int window_width 	    = 800;
int window_height 	    = 600;
int glutWindowHandle    = 0;
float translate_x       = -2.0f;
float translate_y       = -2.0f;
float translate_z       = 5.0f;

// mouse controls
int mouse_old_x;
int mouse_old_y;
int mouse_buttons       = 0;
float rotate_x          = 0.0;
float rotate_y          = 0.0;

// system 
rtps::RTPS* mbs;
#define NUM_BOIDS       10


// callbacks functions
void appRender();
void appKeyboard(unsigned char key, int x, int y);
void appDestroy();
void appMouse(int button, int state, int x, int y);
void appMotion(int x, int y);

// initialize the scene
void init_gl();

int main(int argc, char** argv){

	// initiaze GLUT
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);
    glutInitWindowSize(window_width, window_height);
    glutInitWindowPosition (glutGet(GLUT_SCREEN_WIDTH)/2 - window_width/2, glutGet(GLUT_SCREEN_HEIGHT)/2 - window_height/2);
	glutWindowHandle = glutCreateWindow("Swarm System");

	// callbacks
	glutDisplayFunc(appRender);
	glutKeyboardFunc(appKeyboard);
    glutMouseFunc(appMouse);
    glutMotionFunc(appMotion);

	// initialize OpenGL extensions
	glewInit();
	GLboolean bGLEW = glewIsSupported("GL_VERSION_2_0 GL_ARB_pixel_buffer_object");
    printf("GLEW supported?: %d\n", bGLEW);

	// initialize OpenGL scene
	init_gl();

	// Swarm system
	rtps::RTPSettings settings(rtps::RTPSettings::Swarm, NUM_BOIDS); 
    mbs = new rtps::RTPS(settings);

	// main loop
	glutMainLoop();

	return 0;
}

void init_gl(){
	// default initialization
        glClearColor(0.0, 0.0, 0.0, 1.0);
        glDisable(GL_DEPTH_TEST);

        // viewport
        glViewport(0, 0, window_width, window_height);

        // projection
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glOrtho(-10., 10., -10., 10., -100., 100.);

        // set view matrix
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();
        gluLookAt(0., 0., 10., 0., 0., 0., 0., 1., 0.);

        return;
}

void appRender(){
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

       	mbs->update();
        mbs->render();

	glutSwapBuffers();
}

void appKeyboard(unsigned char key, int x, int y){
        switch(key){
                case '\033': // escape quits
                case '\015': // Enter quits    
                case 'Q':    // Q quits
                case 'q':    // q (or escape) quits
                        // Cleanup up and quit
                        appDestroy();
                        break;
        }

}

void appDestroy(){	        
	delete mbs;

        if(glutWindowHandle)
                glutDestroyWindow(glutWindowHandle);

        printf("Exit from appDestroy!\n");

        exit(0);
}

void appMouse(int button, int state, int x, int y){
                if (state == GLUT_DOWN) {
                    mouse_buttons |= 1<<button;
                }
                else if (state == GLUT_UP) {
                    mouse_buttons = 0;
                                              
                }
                mouse_old_x = x;
                mouse_old_y = y;

                glutPostRedisplay();
}

void appMotion(int x, int y){
            float dx, dy;
                    
            dx = x - mouse_old_x;
            dy = y - mouse_old_y;
 
            if (mouse_buttons & 1) {
                rotate_x += dy * 0.2;
                rotate_y += dx * 0.2;
            }
            else if (mouse_buttons & 4) {
                translate_z -= dy * 0.5;
            }

            mouse_old_x = x;
            mouse_old_y = y;

            // set view matrix
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            glMatrixMode(GL_MODELVIEW);

            glLoadIdentity();

            glRotatef(-90, 1.0, 0.0, 0.0);
            glTranslatef(translate_x, translate_z, translate_y);
            glRotatef(rotate_x, 1.0, 0.0, 0.0);
            glRotatef(rotate_y, 0.0, 0.0, 1.0); //we switched around the axis so make this rotate_z
            
            glutPostRedisplay();
}



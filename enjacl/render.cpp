#include <GL/glew.h>
#if defined __APPLE__ || defined(MACOSX)
    #include <GLUT/glut.h>
#else
    #include <GL/glut.h>
    //OpenCL stuff
#endif

#include "enja.h"
#include "timege.h"


int EnjaParticles::render(float dt, int type=0)
{
    // Render the particles with OpenGL
    // dt is the time step
    // type is how to render (several options will be made available
    // and this should get more sophisticated)
 
	ts[2]->start();

    ts[0]->start();
    update(dt);     //call the particle update function (executes the opencl)
    ts[0]->stop();

    ts[1]->start();
    //glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    //printf("render!\n");
    
    glDisable(GL_LIGHTING);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    //glBlendFunc(GL_SRC_ALPHA, GL_ONE);
    glEnable(GL_POINT_SMOOTH); 
    
    glBindBuffer(GL_ARRAY_BUFFER, c_vbo);
    glColorPointer(4, GL_FLOAT, 0, 0);

    glBindBuffer(GL_ARRAY_BUFFER, v_vbo);
    glVertexPointer(4, GL_FLOAT, 0, 0);

    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_COLOR_ARRAY);

    //glColor3f(0,1,0);
    glPointSize(10.);
    glDrawArrays(GL_POINTS, 0, num);

    glDisableClientState(GL_COLOR_ARRAY);
    glDisableClientState(GL_VERTEX_ARRAY);
    glDisable(GL_POINT_SMOOTH);
    glDisable(GL_BLEND);
    glEnable(GL_LIGHTING);
    glBindBuffer(GL_ARRAY_BUFFER, 0);


    ts[1]->stop();
//    glutPostRedisplay();

	ts[2]->stop();

}

//#include <Array3D.h>

#include <vector>
#include "glincludes.h"
//#include "struct.h"
#include <domain/IV.h>
#include "boids.h"

using namespace rtps;
using namespace std;

typedef vector<float4> VF;
typedef vector<int> VI;

/****
TODO: 
-- cone of vision
-- external velocity field
****/

// one global variable
Boids* boids;

//----------------------------------------------------------------------
//----------------------------------------------------------------------
//----------------------------------------------------------------------
Boids* initBoids()
{
	float4 center = float4(-75.,0.,0.,1.);
	float radius = 30.;
	float spacing = 10.0f;
	float scale = 1.f;

	float edge = 100.f;
	float offsetx = 0.f;
	float offsety = 0.f;
	float4 min = float4(-edge+offsetx, -edge+offsety, 0., 0.);
	float4 max = float4( edge+offsetx,  edge+offsety, 0., 0.);

	//int num = 4;// 2024;
	int num = 2024;
	VF pos(num);
	//pos = addCircle(num, center, radius, spacing, scale);
	GE_addRect(num, min, max, spacing, scale, pos);
	//addRandRect(num, min, max, spacing, scale, min, max, pos);
	#if 0
	pos[0] = float4(-edge, -edge, 0., 1.);
	pos[1] = float4( edge, -edge, 0., 1.);
	pos[2] = float4( edge,  edge, 0., 1.);
	pos[3] = float4(-edge,  edge, 0., 1.);
	#endif

	VF vel, acc;
	acc.resize(pos.size());
	vel.resize(pos.size());

	//printf("before constructor: pos.size= %d\n", pos.size());
	Boids* boids = new Boids(pos);

	for (int i=0; i < vel.size(); i++) {
		vel[i] = float4(0.,0.,0.,1.);
		acc[i] = float4(0.,0.,0.,1.);
	}

	#if 0
	// random velocities
	float rscale = 5.;
	for (int i=0; i < vel.size(); i++) {
        vel[i] = float4((float) rand()/RAND_MAX, (float) rand()/RAND_MAX,0.f,1.0f);
		vel[i] = rscale*vel[i];
	}
	#endif


	boids->set_ic(pos, vel, acc);
	return boids;
}
//----------------------------------------------------------------------
void vector_field(VF& pos, VF& vel, float scale)
{
	glBegin(GL_LINES);
		for (int i=0; i < pos.size(); i++) {
			glVertex2f(pos[i].x, pos[i].y);
			glVertex2f(pos[i].x+scale*vel[i].x, pos[i].y+scale*vel[i].y);
		}
	glEnd();
}
//----------------------------------------------------------------------
void display()
{
	static int count=0;

   boids->update();
   count++;
   //if (count > 1) for (;;) ;
      //exit(0);

   glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
   glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

   glPushMatrix();
   //glScalef(2.,2.,1.);

   // grid overlay based on desired min boid separation
   glBegin(GL_LINES);
   	glColor3f(.2,.2,.2);
	float dim = boids->getDomainSize();
	float sep = boids->getDesiredSeparation();
	int nb = 2*dim/sep;
	for (int j=0; j < nb; j++) {
	for (int i=0; i < nb; i++) {
		glVertex2f(-dim+i*sep, -dim+j*sep);
		glVertex2f(-dim+(i+1)*sep, -dim+j*sep);
	}}
	for (int i=0; i < nb; i++) {
	for (int j=0; j < nb; j++) {
		glVertex2f(-dim+i*sep, -dim+j*sep);
		glVertex2f(-dim+i*sep, -dim+(j+1)*sep);
	}}
   glEnd();

   VF& pos = boids->getPos();
   glBegin(GL_POINTS);
   	  glColor3f(1.,1.,1.);
   	  for (int i=0; i < pos.size(); i++) {
	  	glVertex2f(pos[i].x, pos[i].y);
	  }
   glEnd();

	#if 1
    // Draw velocity field

    //VF& vc = boids->vel_coh;
    //glColor3f(1., 0., 0.);
    vector_field(pos, boids->vel_coh, 100.); // vel, scale
   
    glColor3f(0., 1., 0.);
    //vector_field(pos, boids->vel_sep, 100.); // vel, scale
    
    glColor3f(1.,1.,0.);
    //VF& va = boids->vel_align;
    //vector_field(pos, boids->vel_align, 100.); // vel, scale
    //for (int i=0; i < va.size(); i++) {
   	    //va[i].print("va");
    //}

   #endif

   glPopMatrix();
   glutSwapBuffers();
}
//----------------------------------------------------------------------
void idleFunc()
{
   glutPostRedisplay();
}
//----------------------------------------------------------------------
void reshapeFunc(int w, int h) 
{
  float dim = 300.;

  glViewport (0, 0, w, h);

  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glOrtho(-dim, dim, -dim, dim, -1, 1);
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();

  boids->setDomainSize(dim);
}
//----------------------------------------------------------------------

void timerCB(int ms)
{
    //this makes sure the appRender function is called every ms miliseconds
    glutTimerFunc(ms, timerCB, ms);
    glutPostRedisplay();
}


//----------------------------------------------------------------------
void appKeyboard(unsigned char key, int x, int y)
{
    //this way we can exit the program cleanly
    switch(key)
    {
        case '\033': // escape quits
        case '\015': // Enter quits    
        case 'Q':    // Q quits
        case 'q':    // q (or escape) quits
            exit(0);
            break;
    }
}


//----------------------------------------------------------------------
int main(int argc, char** argv)
{
	// initialize GL graphics

   glutInit(&argc, argv);	// added by Myrna Merced

   glutInitDisplayMode(GLUT_RGBA | GLUT_DEPTH | GLUT_DOUBLE);
   glutInitWindowPosition(200, 0);
   glutInitWindowSize(512, 512);
   //glutInitWindowSize(gWindowWidth, gWindowHeight);
   glutCreateWindow("Gordon's Flocking");

   glutDisplayFunc(display);
   glutKeyboardFunc(appKeyboard);
   //glutSpecialFunc(KeyboardSpecialFunc);
   glutReshapeFunc(reshapeFunc);
   glutTimerFunc(30, timerCB, 30);
   //glutIdleFunc(timerCB);
  
   boids = initBoids();

   //glutSwapBuffers();
   glutMainLoop();

   return 0;
}
//----------------------------------------------------------------------

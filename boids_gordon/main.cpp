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
	float4 center = float4(-150.,0.,0.,1.);
	float radius = 30.;
	float spacing = 3.0f;
	float scale = 1.;
	//float avg_vel = 00.00025;
	float avg_vel = 250.;

	//float wcoh = 1.;
	//float wsep = 1.;
	//float walign = 1.;

	float edge = 50;
	float4 min = float4(-edge+50, -edge+50, 0., 0.);
	float4 max = float4( edge+50,  edge+50, 0., 0.);

	int num = 2024;
	VF pos(num);
	//pos = addCircle(num, center, radius, spacing, scale);
	pos = addRect(num, min, max, spacing, scale);

	VF vel, acc;
	acc.resize(pos.size());
	vel.resize(pos.size());

	Boids* boids = new Boids(pos);

	for (int i=0; i < vel.size(); i++) {
		vel[i] = float4(avg_vel,0.,0.,1.);
		acc[i] = float4(0.,0.,0.,1.);
	}

	boids->set_ic(pos, vel, acc);
	return boids;
}
//----------------------------------------------------------------------
void display()
{
   boids->update();
   glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
   glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

   VF& pos = boids->getPos();
   glBegin(GL_POINTS);
   	  for (int i=0; i < pos.size(); i++) {
	  	glVertex2d(pos[i].x, pos[i].y);
	  }
   glEnd();

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
  float dim = 200.;

  glViewport (0, 0, w, h);

  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glOrtho(-dim, dim, -dim, dim, -1, 1);
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();

  boids->setDomainSize(dim);
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
   //glutKeyboardFunc(KeyboardFunc);
   //glutSpecialFunc(KeyboardSpecialFunc);
   glutReshapeFunc(reshapeFunc);
   glutIdleFunc(idleFunc);
  
   boids = initBoids();

   //glutSwapBuffers();
   glutMainLoop();

   return 0;
}
//----------------------------------------------------------------------

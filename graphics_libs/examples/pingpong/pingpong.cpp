#include <glincludes.h>
#include <GLUT/glut.h>
#include <Vec3.h>
#include <tex_ogl.h>
#include <ping_pong.h>
#include <vbo.h>
#include <vector>

using namespace std;


// demonstrate use of VBOs
struct VERT { float x, y, z; };
struct COL { float r, g, b, a; };
VBO<VERT,COL>* vbo;

// demonstrate use of PingPong buffers
PingPong* pp;

//----------------------------------------------------------------------
// GLUT idle function
void idle()
{
    glutPostRedisplay();
}
//----------------------------------------------------------------------
// GLUT display function
void display()
{
	// Draw into the pingpong buffer
	pp->begin();

	glClear(GL_COLOR_BUFFER_BIT);
	vbo->draw(GL_LINES, 2);

	// Stop drawing into the pingpong buffer
	pp->end();

	// swap the two pingpong buffers
	pp->swap();

	// transfer the contents of the pingpong buffer to the back screen buffer
	pp->toBackBuffer();

	// swap the screen buffers
    glutSwapBuffers();
}
//----------------------------------------------------------------------
// GLUT reshape function
void reshape(int w, int h)
{
    if (h == 0) h = 1;

    glViewport(0, 0, w, h);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(-1, 1, -1, 1);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
}
//----------------------------------------------------------------------
// Called at startup
void initialize()
{ }
//----------------------------------------------------------------------
void fillVBO()
{
	VBO<VERT,COL>* vbo = new VBO<VERT,COL>();
	VERT v;
	COL c;
	vector<VERT> vert;
	vector<COL> col;

	v.x = -1.; v.y = -1.; v.z = 0.;
	vert.push_back(v);
	v.x = +1.; v.y = +1.; v.z = 0.;
	vert.push_back(v);

	c.r = 0.; c.b = 1.; c.r = 0.; c.a = 1.;
	col.push_back(c);
	c.r = 0.; c.b = 1.; c.r = 0.; c.a = 1.;
	col.push_back(c);

	vbo->create(&vert, &col);
}
//----------------------------------------------------------------------
int main(int argc, char** argv)
{

    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
    //glutInitDisplayMode(GLUT_DOUBLE | GLUT_DEPTH | GLUT_RGBA);
    glutInit(&argc, argv);
    glutCreateWindow("Hello, GPGPU! (GLSL version)");

	GLenum err = glewInit();
	printf("glewInit error: %d\n", (int) err);

    glutInitWindowSize(512, 512);
	glutIdleFunc(idle);
	glutDisplayFunc(display);
	glutReshapeFunc(reshape);
	
	pp = new PingPong(512, 512);

	fillVBO();

	glutMainLoop();

	return(0);
}
//----------------------------------------------------------------------

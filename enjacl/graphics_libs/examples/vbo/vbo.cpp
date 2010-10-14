#include <glincludes.h>
#include <GLUT/glut.h>
#include <Vec3.h>
#include <vbo.h>
#include <vector>

using namespace std;

// demonstrate use of VBOs

struct VERT {
	float x, y, z;
};

struct COL {
	float r, g, b, a;
};

VBO<VERT,COL>* vbo;

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
	glDrawBuffer(GL_BACK);
	glClear(GL_COLOR_BUFFER_BIT);

	vbo->draw(GL_LINES, 2);

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
{
#if 0
    // Initialize the "OpenGL Extension Wrangler" library

    // Ensure we have the necessary OpenGL Shading Language extensions.
    if (glewGetExtension("GL_ARB_fragment_shader")      != GL_TRUE ||
        glewGetExtension("GL_ARB_vertex_shader")        != GL_TRUE ||
        glewGetExtension("GL_ARB_shader_objects")       != GL_TRUE ||
        glewGetExtension("GL_ARB_shading_language_100") != GL_TRUE)
    {
        fprintf(stderr, "Driver does not support OpenGL Shading Language\n");
        exit(1);
    }

    // Create the example object
    //g_pHello = new HelloGPGPU(512, 512);
    g_pHello = new HelloGPGPU(512, 512);
#endif
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

    glutInitWindowSize(750, 750);
	glutIdleFunc(idle);
	glutDisplayFunc(display);
	glutReshapeFunc(reshape);


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

	glutMainLoop();

	return(0);
}
//----------------------------------------------------------------------

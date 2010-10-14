#include <glincludes.h>
#include <GLUT/glut.h>
#include <vbo.h>

// demonstrate use of VBOs

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
	glCallList(1);

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
    glewInit();

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
    glutInitWindowSize(750, 750);
	glutIdleFunc(idle);
	glutDisplayFunc(display);
	glutReshapeFunc(reshape);

	glNewList(1, GL_COMPILE);
	glBegin(GL_LINES);
		glColor3f(1.,0.,0.);
		glVertex2f(-1.,-1.);
		glVertex2f(1.,1.);
	glEnd();
	glEndList();


	glutMainLoop();


	return(0);
}
//----------------------------------------------------------------------

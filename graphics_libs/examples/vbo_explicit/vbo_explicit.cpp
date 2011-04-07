#include <glincludes.h>
#include <GLUT/glut.h>
#include <Vec3.h>
//#include <vbo.h>
#include <vector>

using namespace std;

// demonstrate use of VBOs

typedef float VEC3[3];
GLuint vbo_v;
GLuint vbo_c;

//VBO<float,float>* vbo;

//----------------------------------------------------------------------
void draw_vbo(GLenum mode, int count)
{
    // First argument is not used\n");
    printf("draw\n");
    printf("draw, vbo_v= %d\n", (int) vbo_v);
    printf("draw, vbo_c= %d\n", (int) vbo_c);
    // what about if int, etc. ?
    //int nbCh_p = sizeof(float) / sizeof(float);
    //int nbCh_c = sizeof(float) / sizeof(float);
	int nbCh_p = 3;  // number coordinates per point
	int nbCh_c = 4;

    //printf("nbCh_p= %d, nbCh_c= %d\n", nbCh_p, nbCh_c);

    //glBindBuffer(GL_ARRAY_BUFFER, vbo_v);
    glBindBuffer(GL_ARRAY_BUFFER, 1);
        glVertexPointer (nbCh_p, GL_FLOAT, 0, 0);
    //glBindBuffer(GL_ARRAY_BUFFER, vbo_c);
    glBindBuffer(GL_ARRAY_BUFFER, 2);
        glColorPointer(nbCh_c, GL_FLOAT, 0, 0);

    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_COLOR_ARRAY);

    //glDrawArrays(mode, first, count); // error
    // first gives "cannot accsess memory"
    glDrawArrays(mode, 0, count); // error ERROR
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_COLOR_ARRAY);
}
//-----------------------------------------
//----------------------------------------------------------------------
//template <class P, class C>
void create_vbo(const vector<float>* vertex, const vector<float>* colorr)
{
    const vector<float>* pts = vertex;
    const vector<float>* color = colorr;
    // vertex info
    int nbBytes = pts->size() * sizeof(float);


    glGenBuffers(1, &vbo_v);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_v); // leave empty to write into it
    //glBufferData(GL_ARRAY_BUFFER, nbBytes, &(*pts)[0], GL_STREAM_COPY);
    glBufferData(GL_ARRAY_BUFFER, nbBytes, &(*pts)[0], GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // color information
    nbBytes = color->size() * sizeof(float); // usually 4 color channels

    printf("VBO::create, nbBytes= %d\n", nbBytes);

    glGenBuffers(1, &vbo_c);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_c); // leave empty to write into it
    //glBufferData(GL_ARRAY_BUFFER, nbBytes, &(*color)[0], GL_STREAM_COPY);
    glBufferData(GL_ARRAY_BUFFER, nbBytes, &(*color)[0], GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}
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
	//glCallList(1);

	printf("GL_POINTS= %d\n", GL_POINTS);
	draw_vbo(GL_LINES, 2);

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


	//if (GLEW_VERSION_1_5) {
		//printf("OpenGL 1.3 is supported\n");
	//}


	#if 0
	glNewList(1, GL_COMPILE);
	glBegin(GL_LINES);
		glColor3f(1.,0.,0.);
		glVertex2f(-1.,-1.);
		glVertex2f(1.,1.);
	glEnd();
	glEndList();
	#endif

	//VBO<float,float>* vbo = new VBO<float,float>();

	vector<float> vert, col;
	vert.push_back(-1.);
	vert.push_back(-1.);
	vert.push_back(0.);
	vert.push_back(1.);
	vert.push_back(1.);
	vert.push_back(0.);

	col.push_back(0.);
	col.push_back(1.);
	col.push_back(0.);
	col.push_back(1.);

	col.push_back(0.);
	col.push_back(1.);
	col.push_back(0.);
	col.push_back(1.);

	create_vbo(&vert, &col);

	glutMainLoop();


	return(0);
}
//----------------------------------------------------------------------

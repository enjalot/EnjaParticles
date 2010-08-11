#include <GL/glew.h>
#if defined __APPLE__ || defined(MACOSX)
    #include <GLUT/glut.h>
#else
    #include <GL/glut.h>
    //OpenCL stuff
#endif

#include <stdlib.h>
#include "enja.h"
#include "timege.h"

//OpenCV include
#include "highgui.h"
#include "cv.h"
using namespace cv;

void EnjaParticles::drawArrays()
{

    //glMatrixMode(GL_MODELVIEW_MATRIX);
    //glPushMatrix();
    //glLoadMatrixd(gl_transform);

    if(blending)
    {
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        //glBlendFunc(GL_SRC_ALPHA, GL_ONE);
    }

    //printf("color buffer\n");
    glBindBuffer(GL_ARRAY_BUFFER, c_vbo);
    glColorPointer(4, GL_FLOAT, 0, 0);

    //printf("vertex buffer\n");
    glBindBuffer(GL_ARRAY_BUFFER, v_vbo);
    glVertexPointer(4, GL_FLOAT, 0, 0);
    
    //printf("index buffer\n");
    glBindBuffer(GL_ARRAY_BUFFER, i_vbo);
    glIndexPointer(GL_INT, 0, 0);

    //printf("enable client state\n");
    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_COLOR_ARRAY);
    glEnableClientState(GL_INDEX_ARRAY);
    
    //Need to disable these for blender
    glDisableClientState(GL_NORMAL_ARRAY);
    //glDisableClientState(GL_EDGE_FLAG_ARRAY);

    //printf("draw arrays\n");
    glDrawArrays(GL_POINTS, 0, num);

    //printf("disable stuff\n");
    glDisableClientState(GL_INDEX_ARRAY);
    glDisableClientState(GL_COLOR_ARRAY);
    glDisableClientState(GL_VERTEX_ARRAY);

    //glPopMatrix();
}

int EnjaParticles::render()
{
    // Render the particles with OpenGL
    // dt is the time step
    // type is how to render (several options will be made available
    // and this should get more sophisticated)
 
    //printf("in EnjaParticles::render\n");
	ts[2]->start();

    printf("loaded cam? %d\n", loadedcam);
    if(!loadedcam)
    {
        loadWebCam();
        loadedcam = true;
    }
    loadTexture();
    //printf("about to update\n");
    ts[0]->start();
    
    for(int i = 0; i < updates; i++)
    {
        update();     //call the particle update function (executes the opencl)
        //cpu_update();
    }

    ts[0]->stop();

    ts[1]->start();
    //glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    //printf("render!\n");
 
    glPushAttrib(GL_ALL_ATTRIB_BITS);
    glPushClientAttrib(GL_CLIENT_ALL_ATTRIB_BITS);


    if(glsl)
    {
        //printf("GLSL\n");
        glEnable(GL_POINT_SPRITE_ARB);
        glTexEnvi(GL_POINT_SPRITE_ARB, GL_COORD_REPLACE_ARB, GL_TRUE);
        glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);
        glDepthMask(GL_TRUE);
        glEnable(GL_DEPTH_TEST);
        //glDisable(GL_DEPTH_TEST);


        glEnable(GL_TEXTURE_2D);
        glActiveTexture(GL_TEXTURE0);


        glUseProgram(glsl_program);
        //glUniform1f( glGetUniformLocation(m_program, "pointScale"), m_window_h / tanf(m_fov * m_fHalfViewRadianFactor));
        glUniform1f( glGetUniformLocation(glsl_program, "pointScale"), point_scale);
        glUniform1f( glGetUniformLocation(glsl_program, "blending"), blending );
        glUniform1f( glGetUniformLocation(glsl_program, "pointRadius"), particle_radius );

        glUniform1i( glGetUniformLocation(glsl_program, "texture_color"), 0);

        //glColor4f(1, 1, 1, 1);

        glBindTexture(GL_TEXTURE_2D, gl_tex);

        drawArrays();

        glUseProgram(0);
        glDisable(GL_TEXTURE_2D);
        glDisable(GL_POINT_SPRITE_ARB);
        //glEnable(GL_DEPTH_TEST);
    }
    else
    {

        glDisable(GL_LIGHTING);
        //glBlendFunc(GL_SRC_ALPHA, GL_ONE);

		// draws circles instead of squares
        glEnable(GL_POINT_SMOOTH); 
        glPointSize(5.*particle_radius);

        drawArrays();
    }
    //printf("done rendering, clean up\n");
   
    glPopClientAttrib();
    glPopAttrib();
    //glDisable(GL_POINT_SMOOTH);
    //glDisable(GL_BLEND);
    glEnable(GL_LIGHTING);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    //make sure rendering timing is accurate
    glFinish();
    ts[1]->stop();
	ts[2]->stop();
    //printf("done rendering\n");
    if (ts[2]->getCount() == 50)
    {
        printf("%s\n", printReport().c_str());
    }

}

int EnjaParticles::loadWebCam()
{
    printf("initialize the webcam\n");
    CvCapture* c = cvCreateCameraCapture(0);
    printf("set cam properties\n");
    //cvSetCaptureProperty( c, CV_CAP_PROP_FRAME_WIDTH, 320);
    //cvSetCaptureProperty( c, CV_CAP_PROP_FRAME_WIDTH, 240);
    capture = c;
}
//int EnjaParticles::loadTexture(std::vector<unsigned char> image, int w, int h)
int EnjaParticles::loadTexture()
{

    /*
    //load the image with OpenCV
    std::string path(CL_SOURCE_DIR);
    //path += "/tex/particle.jpg";
    //path += "/tex/enjalot.jpg";
    path += "/tex/reddit.png";
    Mat img = imread(path, 1);
*/

    //capture = cvCreateCameraCapture(0);
    printf("about to capture frame!\n");
    IplImage* frame = cvQueryFrame((CvCapture*)capture);
    printf("got frame?\n");
    Mat img(frame);
    //Mat img = imread("tex/enjalot.jpg", 1);
    //convert from BGR to RGB colors
    //cvtColor(img, img, CV_BGR2RGB);
    //this is ugly but it makes an iterator over our image data
    //MatIterator_<Vec<uchar, 3> > it = img.begin<Vec<uchar,3> >(), it_end = img.end<Vec<uchar,3> >();
    MatIterator_<Vec<uchar, 3> > it = img.begin<Vec<uchar,3> >(), it_end = img.end<Vec<uchar,3> >();
    int w = img.size().width;
    int h = img.size().height;
    int n = w * h;

    printf("w: %d h: %d: n: %d\n", w, h, n);
    std::vector<unsigned char> image;//there are n bgr values 

    printf("read image data %d \n", n);
    for(; it != it_end; ++it)
    {
   //     printf("asdf: %d\n", it[0][0]);
        image.push_back(it[0][0]);
        image.push_back(it[0][1]);
        image.push_back(it[0][2]);
    }
    /*
    unsigned char* asdf = &image[0];
    printf("char string:\n");
    for(int i = 0; i < 3*n; i++)
    {
        printf("%d,", asdf[i]);
    }
    printf("\n charstring over\n");
    */
    /*
    int w = 32;
    int h = 32;
    //#include "tex/particle.txt"
    #include "tex/reddit.txt"
    /*
    int w = 96;
    int h = 96;
    #include "tex/enjalot.txt"
    */
    //load as gl texture
    glGenTextures(1, &gl_tex);
    glBindTexture(GL_TEXTURE_2D, gl_tex);

    glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0,
     GL_BGR_EXT, GL_UNSIGNED_BYTE, &image[0]);


}

int EnjaParticles::compileShaders()
{

    //this may not be the cleanest implementation
    #include "shaders.cpp"

    //printf("vertex shader:\n%s\n", vertex_shader_source);
    //printf("fragment shader:\n%s\n", fragment_shader_source);


    GLuint vertex_shader = glCreateShader(GL_VERTEX_SHADER);
    GLuint fragment_shader = glCreateShader(GL_FRAGMENT_SHADER);

    glShaderSource(vertex_shader, 1, &vertex_shader_source, 0);
    glShaderSource(fragment_shader, 1, &fragment_shader_source, 0);
    
    glCompileShader(vertex_shader);
    GLint len;
    glGetShaderiv(vertex_shader, GL_INFO_LOG_LENGTH, &len);
    if(len > 0)
    {
        char log[1024];
        glGetShaderInfoLog(vertex_shader, 1024, 0, log);
        printf("Vertex Shader log:\n %s\n", log);
    }
    glCompileShader(fragment_shader);
    glGetShaderiv(fragment_shader, GL_INFO_LOG_LENGTH, &len);
    if(len > 0)
    {
        char log[1024];
        glGetShaderInfoLog(fragment_shader, 1024, 0, log);
        printf("Vertex Shader log:\n %s\n", log);
    }

    GLuint program = glCreateProgram();

    glAttachShader(program, vertex_shader);
    glAttachShader(program, fragment_shader);

    glLinkProgram(program);

    // check if program linked
    GLint success = 0;
    glGetProgramiv(program, GL_LINK_STATUS, &success);

    if (!success) {
        char temp[256];
        glGetProgramInfoLog(program, 256, 0, temp);
        printf("Failed to link program:\n%s\n", temp);
        glDeleteProgram(program);
        program = 0;
    }

    return program;


}

void EnjaParticles::use_glsl()
{
    glsl_program = compileShaders();
    if(glsl_program != 0)
    {
        glsl = true;
        //loadTexture();
    }
    else
    {
        glsl = false;
    }

}


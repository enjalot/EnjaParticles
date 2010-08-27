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

void EnjaParticles::render_slow()
{
    std::vector<float> poses(num*4);
    std::vector<float> coles(num*4);

    glBindBuffer(GL_ARRAY_BUFFER, v_vbo);
    void* ptr = glMapBufferARB(GL_ARRAY_BUFFER, GL_READ_ONLY_ARB);

    for(int i =0; i < num*4; i++)
    {
        poses[i] = ((float*)ptr)[i];
    }

    glUnmapBufferARB(GL_ARRAY_BUFFER); 
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glBindBuffer(GL_ARRAY_BUFFER, c_vbo);
    void* colptr = glMapBufferARB(GL_ARRAY_BUFFER, GL_READ_ONLY_ARB);
    for(int i =0; i < num*4; i++)
    {
        coles[i] = ((float*)colptr)[i];
    }
   
    glUnmapBufferARB(GL_ARRAY_BUFFER); 
    

    // Use glBegin/glEnd to draw
    printf("render POINTS!\n");
    printf("poses[0] %f\n", poses[0]);
    printf("coles[0] %f\n", coles[0]);
    float x, y, z, cx, cy, cz;
    glBegin(GL_POINTS);
    for(int i = 0; i < num*4; i+=4)
    {

        x = poses[i];
        y = poses[i+1];
        z = poses[i+2];

        cx = coles[i];
        cy = coles[i+1];
        cz = coles[i+2];

        glColor3f(cx,cy,cz);
        //glColor3f(1,0,0);
        //glVertex3f(0,0,0);
        //in K_Common.cuh the float4 vecs had .w value being set to 0 by default
        glVertex4f(x, y, z,1);
    }
    glEnd();
}
//----------------------------------------------------------------------
void EnjaParticles::drawArrays()
{

    //glMatrixMode(GL_MODELVIEW_MATRIX);
    //glPushMatrix();
    //glLoadMatrixd(gl_transform);

    if(blending)
    {
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    }

    //printf("color buffer\n");
    glBindBuffer(GL_ARRAY_BUFFER, c_vbo);
    glColorPointer(4, GL_FLOAT, 0, 0);

    /*
    void* colptr = glMapBufferARB(GL_ARRAY_BUFFER, GL_READ_ONLY_ARB);
    printf("col PTR[400]: %f\n", ((float*)colptr)[400]);
    printf("col PTR[401]: %f\n", ((float*)colptr)[401]);
    printf("col PTR[402]: %f\n", ((float*)colptr)[402]);
    glUnmapBufferARB(GL_ARRAY_BUFFER); 
    */


    //printf("vertex buffer\n");
    glBindBuffer(GL_ARRAY_BUFFER, v_vbo);
    glVertexPointer(4, GL_FLOAT, 0, 0);

    // map the buffer object into client's memory
    /*
    void* ptr = glMapBufferARB(GL_ARRAY_BUFFER, GL_READ_ONLY_ARB);
    printf("Pos PTR[400]: %f\n", ((float*)ptr)[400]);
    printf("Pos PTR[401]: %f\n", ((float*)ptr)[401]);
    printf("Pos PTR[402]: %f\n", ((float*)ptr)[402]);
    glUnmapBufferARB(GL_ARRAY_BUFFER); 
    */
    
    //printf("index buffer\n");
    //glBindBuffer(GL_ARRAY_BUFFER, i_vbo);
    //glIndexPointer(GL_INT, 0, 0);

    //printf("enable client state\n");
    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_COLOR_ARRAY);
    //glEnableClientState(GL_INDEX_ARRAY);
    
    //Need to disable these for blender
    glDisableClientState(GL_NORMAL_ARRAY);
    //glDisableClientState(GL_EDGE_FLAG_ARRAY);
    glDisableClientState(GL_INDEX_ARRAY);

    printf("draw arrays num: %d\n", num);

    glDrawArrays(GL_POINTS, 0, num);

    //printf("disable stuff\n");
    //glDisableClientState(GL_INDEX_ARRAY);
    glDisableClientState(GL_COLOR_ARRAY);
    glDisableClientState(GL_VERTEX_ARRAY);

    //glPopMatrix();
}

//----------------------------------------------------------------------
int EnjaParticles::render()
{
    // Render the particles with OpenGL
    // dt is the time step
    // type is how to render (several options will be made available
    // and this should get more sophisticated)
 
    //printf("in EnjaParticles::render\n");
	ts[2]->start();

    //printf("about to update\n");
    ts[0]->start();
    
    /*
    for(int i = 0; i < updates; i++)
    {
        update();     //call the particle update function (executes the opencl)
        //cpu_update();
    }
    */

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
        glEnable(GL_VERTEX_PROGRAM_POINT_SIZE_NV);
        glDepthMask(GL_TRUE);
        glEnable(GL_DEPTH_TEST);

        glUseProgram(glsl_program);
        //glUniform1f( glGetUniformLocation(m_program, "pointScale"), m_window_h / tanf(m_fov * m_fHalfViewRadianFactor));
        glUniform1f( glGetUniformLocation(glsl_program, "pointScale"), point_scale);
        glUniform1f( glGetUniformLocation(glsl_program, "blending"), blending );
        glUniform1f( glGetUniformLocation(glsl_program, "pointRadius"), particle_radius );

        glColor4f(1, 1, 1, 1);

        drawArrays();

        glUseProgram(0);
        glDisable(GL_POINT_SPRITE_ARB);
    }
    else   // do not use glsl
    {

        glDisable(GL_LIGHTING);
        //glBlendFunc(GL_SRC_ALPHA, GL_ONE);

		// draws circles instead of squares
        glEnable(GL_POINT_SMOOTH); 
        glPointSize(5.*particle_radius);

        drawArrays();
        //render_slow();
    }
    printf("done rendering, clean up\n");
   
    glPopClientAttrib();
    glPopAttrib();
    //glDisable(GL_POINT_SMOOTH);
    //glDisable(GL_BLEND);
    printf("make it here??\n");
    glEnable(GL_LIGHTING);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    //make sure rendering timing is accurate
    glFinish();
    printf("make it here?\n");
    ts[1]->stop();
	ts[2]->stop();

    printf("timeres?\n");
    //printf("done rendering\n");
    if (ts[2]->getCount() == 50)
    {
        printf("%s\n", printReport().c_str());
    }
    printf("here?????\n");

}


//----------------------------------------------------------------------
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

//----------------------------------------------------------------------
void EnjaParticles::use_glsl()
{
    glsl_program = compileShaders();
    if(glsl_program != 0)
    {
        glsl = true;
    }
    else
    {
        glsl = false;
    }

}

//----------------------------------------------------------------------

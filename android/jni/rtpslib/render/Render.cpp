#include <stdio.h>

#include <android/log.h>

#include "Render.h"


namespace rtps{

Render::Render(GLuint pos, GLuint col, int n)
{
    rtype = POINTS;
    pos_vbo = pos;
    col_vbo = col;
    num = n;
}

Render::~Render()
{
}

void Render::drawArrays()
{

    //glMatrixMode(GL_MODELVIEW_MATRIX);
    //glPushMatrix();
    //glLoadMatrixd(gl_transform);

    
    //if(blending)
    {
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    }
    

    //printf("color buffer\n");
    glBindBuffer(GL_ARRAY_BUFFER, col_vbo);
    glColorPointer(4, GL_FLOAT, 0, 0);

    //__android_log_print(ANDROID_LOG_INFO, "RTPS", "draw arrays, color done");
    /*
    void* colptr = glMapBufferARB(GL_ARRAY_BUFFER, GL_READ_ONLY_ARB);
    printf("col PTR[400]: %f\n", ((float*)colptr)[400]);
    printf("col PTR[401]: %f\n", ((float*)colptr)[401]);
    printf("col PTR[402]: %f\n", ((float*)colptr)[402]);
    glUnmapBufferARB(GL_ARRAY_BUFFER); 
    */


    //printf("vertex buffer\n");
    glBindBuffer(GL_ARRAY_BUFFER, pos_vbo);
    glVertexPointer(4, GL_FLOAT, 0, 0);
    //__android_log_print(ANDROID_LOG_INFO, "RTPS", "draw arrays, pos done");

    // map the buffer object into client's memory
    /*
    void* ptr = glMapBufferARB(GL_ARRAY_BUFFER, GL_READ_ONLY_ARB);
    printf("Pos PTR[400]: %f\n", ((float*)ptr)[400]);
    printf("Pos PTR[401]: %f\n", ((float*)ptr)[401]);
    printf("Pos PTR[402]: %f\n", ((float*)ptr)[402]);
    glUnmapBufferARB(GL_ARRAY_BUFFER); 
    */
    
    //printf("enable client state\n");
    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_COLOR_ARRAY);
    //glEnableClientState(GL_INDEX_ARRAY);
    
    //Need to disable these for blender
    glDisableClientState(GL_NORMAL_ARRAY);
    //glDisableClientState(GL_EDGE_FLAG_ARRAY);

    //printf("draw arrays num: %d\n", num);

    //__android_log_print(ANDROID_LOG_INFO, "RTPS", "inside drawArrays pos_vbo=%d", pos_vbo);
    //printf("NUM %d\n", num);
    glDrawArrays(GL_POINTS, 0, num);

    //__android_log_print(ANDROID_LOG_INFO, "RTPS", "draw arrays  done");
    //printf("disable stuff\n");
    //glDisableClientState(GL_INDEX_ARRAY);
    glDisableClientState(GL_COLOR_ARRAY);
    glDisableClientState(GL_VERTEX_ARRAY);

    //glPopMatrix();
}

//----------------------------------------------------------------------
void Render::render()
{
    // Render the particles with OpenGL

    ///glPushAttrib(GL_ALL_ATTRIB_BITS);
    ///glPushClientAttrib(GL_CLIENT_ALL_ATTRIB_BITS);


    /*
    //TODO enable GLSL shading 
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
*/
    //__android_log_print(ANDROID_LOG_INFO, "RTPS", "start rendering");
        glDisable(GL_LIGHTING);
        //glBlendFunc(GL_SRC_ALPHA, GL_ONE);

		// draws circles instead of squares
        glEnable(GL_POINT_SMOOTH); 
        //TODO make the point size a setting
        glPointSize(10.0f);
        glColor4f(1.0f, 0.0f, 0.0f, 1.0f);

        drawArrays();
    //}
    //printf("done rendering, clean up\n");
   
    ////glPopClientAttrib();
    ///glPopAttrib();
    //glDisable(GL_POINT_SMOOTH);
    //glDisable(GL_BLEND);
    //glEnable(GL_LIGHTING);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    //make sure rendering timing is accurate
    glFinish();
    //printf("done rendering\n");
    //__android_log_print(ANDROID_LOG_INFO, "RTPS", "done rendering");

}

void Render::render_box(float4 min, float4 max)
{
    //glEnable(GL_DEPTH_TEST);

    GLfloat v[24];
    v[0] = min.x; 
    v[1] = min.y;
    v[2] = min.z;
    v[3] = min.x;
    v[4] = min.y;
    v[5] = max.z;
    
    v[6] = min.x; 
    v[7] = max.y;
    v[8] = min.z;
    v[9] = min.x;
    v[10] = max.y;
    v[11] = max.z;
    
    v[12] = min.x; 
    v[13] = min.y;
    v[14] = min.z;
    v[15] = min.x;
    v[16] = max.y;
    v[17] = min.z;
 
    v[18] = min.x; 
    v[19] = min.y;
    v[20] = max.z;
    v[21] = min.x;
    v[22] = max.y;
    v[23] = max.z;
    
    //__android_log_print(ANDROID_LOG_INFO, "RTPS", "min x: %f, y: %f, z: %f ::: max: x: %f, y: %f, z: %f ", min.x, min.y, min.z, max.x, max.y, max.z);

    glEnableClientState(GL_VERTEX_ARRAY);
    glVertexPointer(3, GL_FLOAT, 0, &v);
    glColor4f(0.0f, 1.0f, 0.0f, 1.0f);
    glDrawArrays( GL_LINES, 0, 8); 
    glDisableClientState(GL_VERTEX_ARRAY);

    //draw axes
    GLfloat xx[8];
    xx[0] = 0;
    xx[1] = 0;
    xx[2] = 0;
    xx[3] = 1;

    xx[4] = 1;
    xx[5] = 0;
    xx[6] = 0;
    xx[7] = 1;
 

    glEnableClientState(GL_VERTEX_ARRAY);
    glVertexPointer(4, GL_FLOAT, 0, &xx);
    glColor4f(1.0f, 0.0f, 0.0f, 1.0f);
    glDrawArrays( GL_LINES, 0, 2); 
    glDisableClientState(GL_VERTEX_ARRAY);

    GLfloat yy[8];
    yy[0] = 0;
    yy[1] = 0;
    yy[2] = 0;
    yy[3] = 1;

    yy[4] = 0;
    yy[5] = 1;
    yy[6] = 0;
    yy[7] = 1;

    glEnableClientState(GL_VERTEX_ARRAY);
    glVertexPointer(4, GL_FLOAT, 0, &yy);
    glColor4f(0.0f, 1.0f, 0.0f, 1.0f);
    glDrawArrays( GL_LINES, 0, 2); 
    glDisableClientState(GL_VERTEX_ARRAY);

    GLfloat zz[8];
    zz[0] = 0;
    zz[1] = 0;
    zz[2] = 0;
    zz[3] = 1;

    zz[4] = 0;
    zz[5] = 0;
    zz[6] = 1;
    zz[7] = 1;
 

    glEnableClientState(GL_VERTEX_ARRAY);
    glVertexPointer(4, GL_FLOAT, 0, &zz);
    glColor4f(0.0f, 0.0f, 1.0f, 1.0f);
    glDrawArrays( GL_LINES, 0, 2); 
    glDisableClientState(GL_VERTEX_ARRAY);



    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glFinish();
    //__android_log_print(ANDROID_LOG_INFO, "RTPS", "render_box");

/*
    //draw grid
    glBegin(GL_LINES);
    //1st face
    glVertex3f(min.x, min.y, min.z);
    glVertex3f(min.x, min.y, max.z);
    
    glVertex3f(min.x, max.y, min.z);
    glVertex3f(min.x, max.y, max.z);

    glVertex3f(min.x, min.y, min.z);
    glVertex3f(min.x, max.y, min.z);
 
    glVertex3f(min.x, min.y, max.z);
    glVertex3f(min.x, max.y, max.z);
    //2nd face
    glVertex3f(max.x, min.y, min.z);
    glVertex3f(max.x, min.y, max.z);
    
    glVertex3f(max.x, max.y, min.z);
    glVertex3f(max.x, max.y, max.z);

    glVertex3f(max.x, min.y, min.z);
    glVertex3f(max.x, max.y, min.z);
 
    glVertex3f(max.x, min.y, max.z);
    glVertex3f(max.x, max.y, max.z);
    //connections
    glVertex3f(min.x, min.y, min.z);
    glVertex3f(max.x, min.y, min.z);
 
    glVertex3f(min.x, max.y, min.z);
    glVertex3f(max.x, max.y, min.z);
 
    glVertex3f(min.x, min.y, max.z);
    glVertex3f(max.x, min.y, max.z);
 
    glVertex3f(min.x, max.y, max.z);
    glVertex3f(max.x, max.y, max.z);
    
    glEnd();
*/
}

//----------------------------------------------------------------------
/*
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
*/

}

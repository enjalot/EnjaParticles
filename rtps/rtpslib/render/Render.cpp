#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include <GL/glew.h>
#if defined __APPLE__ || defined(MACOSX)
    #include <GLUT/glut.h>
#else
    #include <GL/glut.h>
//OpenCL stuff
#endif

#ifdef USE_IL
    #include <IL/il.h>
    #include <IL/ilu.h>
    #include <IL/ilut.h>
#endif

#include "Render.h"
#include "util.h"
#include "stb_image.h" 

using namespace std;

namespace rtps
{

    //----------------------------------------------------------------------
    //Render::Render(GLuint pos, GLuint col, int n, CL* cli, RTPSettings& _settings) :
    //settings(_settings)
    //{
    //}
    //----------------------------------------------------------------------
    Render::Render(GLuint pos, GLuint col, int n, CL* cli, RTPSettings* _settings)
    {
        this->settings = _settings;

        rtype = POINTS;
        pos_vbo = pos;
        col_vbo = col;
        this->cli=cli;
        num = n;
        window_height=600;
        window_width=800;
        near_depth=0.;
        far_depth=1.;
        write_framebuffers = false;

        printf("GL VERSION %s\n", glGetString(GL_VERSION));
        //glsl = false;
        glsl = true;
        //mikep = true;
        //mikep = false;
        //blending = true;
        blending = false;

        // 0: particles
        // 1: textures
        // 2: blurring  (works ok)
        int render_type = settings->getRenderType();

        if (glsl)
        {
            fbos.resize(1);
            glGenFramebuffers(1,&fbos[0]);
            printf("**** RENDER **** render_type = %d\n", render_type);
            if (render_type == 2)
            {
                smoothing = BILATERAL_GAUSSIAN_SHADER;
            }
            else
            {
                smoothing = NO_SHADER;
            }
            //particle_radius = 0.0125f*0.5f;
            particle_radius = 0.0125f*0.5f;

            createFramebufferTextures();

            glBindFramebuffer(GL_DRAW_FRAMEBUFFER,fbos[0]);
            glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D,gl_tex["thickness"],0);
            glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER,GL_COLOR_ATTACHMENT1,GL_TEXTURE_2D,gl_tex["depthColor"],0);
            glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER,GL_COLOR_ATTACHMENT2,GL_TEXTURE_2D,gl_tex["normalColor"],0);
            glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER,GL_COLOR_ATTACHMENT3,GL_TEXTURE_2D,gl_tex["lightColor"],0);
            glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER,GL_COLOR_ATTACHMENT4,GL_TEXTURE_2D,gl_tex["Color"],0);
            glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER,GL_COLOR_ATTACHMENT5,GL_TEXTURE_2D,gl_tex["depthColorSmooth"],0);
            glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER,GL_DEPTH_ATTACHMENT,GL_TEXTURE_2D,gl_tex["depth"],0);
            glBindFramebuffer(GL_DRAW_FRAMEBUFFER,0);


            //glFinish();
            /*
            cl_depth = Buffer<float>(cli,gl_tex["depth"],1);
    
            //printf("OpenCL error is %s\n",oclErrorString(cli->err));
            std::string path(GLSL_BIN_DIR);
            path += "/curvature_flow.cl";
            k_curvature_flow = Kernel(cli, path, "curvature_flow");
    
            k_curvature_flow.setArg(0,cl_depth.getDevicePtr());
            k_curvature_flow.setArg(1,window_width);
            k_curvature_flow.setArg(2,window_height);
            k_curvature_flow.setArg(3,40); 
            */ 

            string vert(GLSL_BIN_DIR);
            string frag(GLSL_BIN_DIR);
            vert+="/sphere_vert.glsl";
            if (smoothing == NO_SHADER)
            {
                //******* for loading textures as sprites
                loadTexture();
                frag+="/sphere_tex_frag.glsl";
                //*******
            }
            else
            {
                frag+="/sphere_frag.glsl";
            }
            glsl_program[SPHERE_SHADER] = compileShaders(vert.c_str(),frag.c_str());
            vert = string(GLSL_BIN_DIR);
            frag = string(GLSL_BIN_DIR);
            vert+="/depth_vert.glsl";
            frag+="/depth_frag.glsl";
            glsl_program[DEPTH_SHADER] = compileShaders(vert.c_str(),frag.c_str());
            vert = string(GLSL_BIN_DIR);
            frag = string(GLSL_BIN_DIR);
            vert+="/gaussian_blur_vert.glsl";
            frag+="/gaussian_blur_x_frag.glsl";
            glsl_program[GAUSSIAN_X_SHADER] = compileShaders(vert.c_str(),frag.c_str());
            frag = string(GLSL_BIN_DIR);
            frag+="/gaussian_blur_y_frag.glsl";
            glsl_program[GAUSSIAN_Y_SHADER] = compileShaders(vert.c_str(),frag.c_str());
            frag = string(GLSL_BIN_DIR);
            frag+="/bilateral_blur_frag.glsl";
            glsl_program[BILATERAL_GAUSSIAN_SHADER] = compileShaders(vert.c_str(),frag.c_str());
            vert = string(GLSL_BIN_DIR);
            frag = string(GLSL_BIN_DIR);
            vert+="/normal_vert.glsl";
            if (smoothing == NO_SHADER)
            {
                frag+="/normal_tex_frag.glsl";
            }
            else
            {
                frag+="/normal_frag.glsl";
            }
            glsl_program[NORMAL_SHADER] = compileShaders(vert.c_str(),frag.c_str()); 

            vert = string(GLSL_BIN_DIR);
            frag = string(GLSL_BIN_DIR);
            vert+="/copy_vert.glsl";
            frag+="/copy_frag.glsl";
            glsl_program[COPY_TO_FB] = compileShaders(vert.c_str(),frag.c_str()); 

        }
        else if (mikep)
        {
            loadTexture();
            GLenum param[] = {GL_GEOMETRY_VERTICES_OUT_EXT,GL_GEOMETRY_INPUT_TYPE_EXT,GL_GEOMETRY_OUTPUT_TYPE_EXT};
            GLint value[] = {4,GL_POINT,GL_TRIANGLE_STRIP};
            string vert(GLSL_BIN_DIR);
            string frag(GLSL_BIN_DIR);
            string geom(GLSL_BIN_DIR);
            vert+="/mpvertex.glsl";
            frag+="/mpfragment.glsl";
            geom+="/mpgeometry.glsl";
            glsl_program[MIKEP_SHADER] = compileShaders(vert.c_str(),frag.c_str(),geom.c_str(),param,value,3);
        }
        setupTimers();
#ifdef USE_IL
        ilInit();
        //iluInit();
        //ilutRenderer(ILUT_OPENGL);
#endif
    }

    //----------------------------------------------------------------------
    Render::~Render()
    {
        printf("Render destructor\n");
        for (map<ShaderType,GLuint>::iterator i = glsl_program.begin();i!=glsl_program.end();i++)
        {
            glDeleteProgram(i->second);
        }


        for (map<string,GLuint>::iterator i = gl_tex.begin();i!=gl_tex.end();i++)
        {
            glDeleteTextures(1,&(i->second));
        }
        //for(vector<GLuint>::iterator i=rbos.begin(); i!=rbos.end();i++)
        //{
        if (rbos.size())
        {
            glDeleteRenderbuffers(rbos.size() ,&rbos[0]);
        }
        //}
        //for(vector<GLuint>::iterator i=fbos.begin(); i!=fbos.end();i++)
        //{
        if (fbos.size())
        {
            glDeleteFramebuffers(fbos.size(),&fbos[0]);
        }
        //}
    }

    //----------------------------------------------------------------------
    void Render::drawArrays()
    {
        glBindBuffer(GL_ARRAY_BUFFER, col_vbo);
        glColorPointer(4, GL_FLOAT, 0, 0);

        glBindBuffer(GL_ARRAY_BUFFER, pos_vbo);
        glVertexPointer(4, GL_FLOAT, 0, 0);

        glEnableClientState(GL_VERTEX_ARRAY);
        glEnableClientState(GL_COLOR_ARRAY);

        //Need to disable these for blender
        glDisableClientState(GL_NORMAL_ARRAY);
        glDrawArrays(GL_POINTS, 0, num);

        glDisableClientState(GL_COLOR_ARRAY);
        glDisableClientState(GL_VERTEX_ARRAY);
    }

    //----------------------------------------------------------------------
    void Render::render()
    {
        //TODO: do this properly
        int xywh[4];
        glGetIntegerv(GL_VIEWPORT, xywh);
        int glwidth = xywh[2];
        int glheight = xywh[3];
        if (glwidth != window_width || glheight != window_height)
        {
            //printf("SETTING DIMENSIONS\n");
            setWindowDimensions(glwidth, glheight);
        }
        float nf[2];
        glGetFloatv(GL_DEPTH_RANGE,nf);
        near_depth = nf[0];
        far_depth = nf[1];

        /*
        printf("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\n");
        printf("w: %d h: %d\n", window_width, window_height);
        printf("x: %d y: %d w: %d h: %d\n", xywh[0], xywh[1], xywh[2], xywh[3]);
        */
        /*
        int whatbuffer;
        glGetIntegerv(GL_DRAW_BUFFER0, &whatbuffer);
        printf("What buffer? %d, GL_BACK: %d\n", whatbuffer, GL_BACK);
        */

        // Render the particles with OpenGL
        //printf("window width = %d window height = %d",window_width,window_height);
        timers[TI_RENDER]->start();

        glPushAttrib(GL_ALL_ATTRIB_BITS);
        glPushClientAttrib(GL_CLIENT_ALL_ATTRIB_BITS);

        if (blending)
        {
            //glDepthMask(GL_FALSE);
            glEnable(GL_BLEND);
            //glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
            glBlendFunc(GL_SRC_ALPHA, GL_ONE);
        }

        //TODO enable GLSL shading 
        if (glsl)
        {

            glViewport(0, 0, window_width, window_height);
            glScissor(0, 0, window_width, window_height);
            //glViewport(0, 0, window_width-xywh[0], window_height-xywh[1]);



            glBindFramebuffer(GL_DRAW_FRAMEBUFFER,fbos[0]);
            //glDrawBuffers(2,buffers);
            glDisable(GL_DEPTH_TEST);
            glDepthMask(GL_FALSE);
            glDrawBuffer(GL_COLOR_ATTACHMENT0);
            glClearColor(0.0f,0.0f,0.0f,1.0f);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            renderPointsAsSpheres();
            glEnable(GL_DEPTH_TEST);
            glDepthMask(GL_TRUE);

            if (blending)
            {
                //glDepthMask(GL_FALSE);
                glDisable(GL_BLEND);
            }
            //GLenum buffers[] = {GL_COLOR_ATTACHMENT4,GL_COLOR_ATTACHMENT5};
            //Render depth and thickness to a textures
            //glDrawBuffers(2,buffers);
            glDrawBuffer(GL_COLOR_ATTACHMENT4);
            glClearColor(0.2f,0.2f,0.8f,1.0f);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            renderPointsAsSpheres();
            //smoothDepth();


            //glDisable(GL_DEPTH_TEST);

            //Smooth the depth texture to emulate a surface.
            //glDrawBuffer(GL_COLOR_ATTACHMENT1);
            //glBindFramebuffer(GL_DRAW_FRAMEBUFFER,0);
            glDrawBuffer(GL_COLOR_ATTACHMENT5);

            glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER,GL_DEPTH_ATTACHMENT,GL_TEXTURE_2D,gl_tex["depth2"],0);
            glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D,gl_tex["depth"]);

            smoothDepth();
            glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER,GL_DEPTH_ATTACHMENT,GL_TEXTURE_2D,gl_tex["depth"],0);
            //If no shader was used to smooth then we need the original depth texture
            if (smoothing!=NO_SHADER)
            {
                glBindTexture(GL_TEXTURE_2D,gl_tex["depth2"]);
            }
            //glCopyTexSubImage2D(GL_TEXTURE_2D,0,0,0,0,0,800,600);


            glDisable(GL_DEPTH_TEST);


            //glBindFramebuffer(GL_DRAW_FRAMEBUFFER,fbos[0]);
            glActiveTexture(GL_TEXTURE1);
            glBindTexture(GL_TEXTURE_2D,gl_tex["Color"]);
            //Render the normals for the new "surface".
            glDrawBuffer(GL_COLOR_ATTACHMENT2);
            glClearColor(0.2f,0.2f,0.8f,1.0f);
            glClear(GL_COLOR_BUFFER_BIT);
            glUseProgram(glsl_program[NORMAL_SHADER]);
            glUniform1i( glGetUniformLocation(glsl_program[NORMAL_SHADER], "depthTex"),0);
            glUniform1i( glGetUniformLocation(glsl_program[NORMAL_SHADER], "colorTex"),1);
            glUniform1f( glGetUniformLocation(glsl_program[NORMAL_SHADER], "del_x"),1.0/((float)window_width));
            glUniform1f( glGetUniformLocation(glsl_program[NORMAL_SHADER], "del_y"),1.0/((float)window_height));
            fullscreenQuad();

            /*
            glUseProgram(0);
            glBindTexture(GL_TEXTURE_2D,0);
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D,0);
            */

            glEnable(GL_DEPTH_TEST);

            glBindFramebuffer(GL_DRAW_FRAMEBUFFER,0);
            glDrawBuffer(GL_BACK);

            glViewport(xywh[0],xywh[1],window_width,window_height);
            glScissor(xywh[0], xywh[1], window_width, window_height);


            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D,gl_tex["normalColor"]);
            glActiveTexture(GL_TEXTURE1);
            if (smoothing!=NO_SHADER)
            {
                glBindTexture(GL_TEXTURE_2D,gl_tex["depth2"]);
            }
            else
            {
                glBindTexture(GL_TEXTURE_2D,gl_tex["depth"]);
            }
            glUseProgram(glsl_program[COPY_TO_FB]);
            glUniform1i( glGetUniformLocation(glsl_program[COPY_TO_FB], "normalTex"),0);
            glUniform1i( glGetUniformLocation(glsl_program[COPY_TO_FB], "depthTex"),1);
            fullscreenQuad();


            glUseProgram(0);
            glBindTexture(GL_TEXTURE_2D,0);
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D,0);


            /*
            //selfish way of rendering
            glDrawBuffer(GL_BACK);
            //glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);
            //glViewport(0,0,window_width,window_height);
            glBindFramebuffer(GL_READ_FRAMEBUFFER,fbos[0]);
            glReadBuffer(GL_COLOR_ATTACHMENT2);
    
            glBlitFramebuffer(0,0,window_width,window_height,
                              0,0,window_width,window_height,
                              //xywh[0],xywh[1],window_width,window_height,
                              GL_DEPTH_BUFFER_BIT|GL_COLOR_BUFFER_BIT,GL_LINEAR);
    
    
            glBindFramebuffer(GL_READ_FRAMEBUFFER,0);
            glReadBuffer(GL_BACK);
            */

            if (blending)
            {
                //glDepthMask(GL_FALSE);
                glEnable(GL_BLEND);
                glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
            }
        }
        else if (mikep)
        {
            //Texture stuff
            glEnable(GL_TEXTURE_2D);
            glActiveTexture(GL_TEXTURE0);


            glUseProgram(glsl_program[MIKEP_SHADER]);
            float emit = 1.f;
            float alpha = .5f;

            glUniform1f( glGetUniformLocation(glsl_program[MIKEP_SHADER], "emit"), emit);
            glUniform1f( glGetUniformLocation(glsl_program[MIKEP_SHADER], "alpha"), alpha);

            //Texture stuff
            glUniform1i( glGetUniformLocation(glsl_program[MIKEP_SHADER], "col"), 0);
            glBindTexture(GL_TEXTURE_2D, gl_tex["texture"]);

            glColor4f(1, 1, 1, 1);

            drawArrays();

            //Texture
            glDisable(GL_TEXTURE_2D);

            glUseProgram(0);

        }
        else   // do not use glsl
        {
            glDisable(GL_LIGHTING);
            //glBlendFunc(GL_SRC_ALPHA, GL_ONE);

            // draws circles instead of squares
            glEnable(GL_POINT_SMOOTH); 
            //TODO make the point size a setting
            glPointSize(5.0f);

            drawArrays();
        }
        //printf("done rendering, clean up\n");

        glDepthMask(GL_TRUE);

        glPopClientAttrib();
        glPopAttrib();
        //glDisable(GL_POINT_SMOOTH);
        if (blending)
        {
            glDisable(GL_BLEND);
        }
        //glEnable(GL_LIGHTING);
        glBindBuffer(GL_ARRAY_BUFFER, 0);

        //make sure rendering timing is accurate
        glFinish();

        //printf("done rendering\n");
        timers[TI_RENDER]->end();
        if (write_framebuffers)
        {
            writeFramebufferTextures();
            write_framebuffers = false;
        }

    }

    void Render::writeBuffersToDisk()
    {
        write_framebuffers = true;
    }


    void Render::writeFramebufferTextures() 
    {
        for (map<string,GLuint>::iterator i = gl_tex.begin();i!=gl_tex.end();i++)
        {
            string s(i->first);
            s+=".png";
            writeTexture(i->second, s.c_str());
        }
    }

    void Render::convertDepthToRGB(const GLfloat* depth, GLuint size, GLuint* rgba) const
    {
        GLfloat minimum = 1.0f;
        for (GLint i = 0;i<size;i++)
        {
            //printf("depth[%d] %f\n",i,depth[i]);
            if (minimum>depth[i])
            {
                minimum = depth[i];
            }
        }
        GLfloat one_minus_min = 1.f-minimum;
        printf("depth 0 %f\n",depth[0]);
        printf("minimum %f \n",minimum);
        for (GLuint i = 0;i<size;i++)
        {
            for (GLuint j = 0;j<4;j++)
            {
                rgba[(i*4)+j]=(GLuint)((depth[i]-minimum)/one_minus_min *255);
                //printf("index = %d i = %d j = %d \n",(i*4)+j,i,j);
            }
        }
        printf("rgba[0] = %d rgba[1] = %d rgba[2] = %d rgba[3] = %d\n",rgba[0],rgba[1],rgba[2],rgba[3]);
        printf("rgba[4] = %d rgba[1] = %d rgba[2] = %d rgba[3] = %d\n",rgba[0],rgba[1],rgba[2],rgba[3]);
    }

    int Render::writeTexture( GLuint tex, const char* filename) const
    {
#ifdef USE_IL
        printf("writing %s texture to disc.\n",filename);
        glBindTexture(GL_TEXTURE_2D,tex);
        ILuint img = ilGenImage();
        ilBindImage(img);
        ilEnable(IL_FILE_OVERWRITE);
        GLuint* image = new GLuint[window_width*window_height*4];
        if (!strcmp(filename,"depth.png") || !strcmp(filename,"depth2.png"))
        {
            GLfloat fimg[window_width*window_height];
            glGetTexImage(GL_TEXTURE_2D,0,GL_DEPTH_COMPONENT,GL_FLOAT,fimg);
            convertDepthToRGB(fimg,window_width*window_height,image);
            //ilTexImage(window_width,window_height,0,4,IL_LUMINANCE, IL_FLOAT, fimg);
        }
        else
        {
            glGetTexImage(GL_TEXTURE_2D,0,GL_RGBA,GL_UNSIGNED_BYTE,image);
        }
        ilTexImage(window_width,window_height,0,4,IL_RGBA, IL_UNSIGNED_BYTE, image);
        glBindTexture(GL_TEXTURE_2D,0);
        ilSave(IL_PNG,filename);
        ilDeleteImage(img);
        delete[] image;
#endif //USE_IL

        return 0;
    }

    void Render::smoothDepth()
    {
        /*glFinish();
        cl_depth.acquire();
        k_curvature_flow.execute(window_width*window_height,128);
        cl_depth.release();
        */
        if (smoothing == NO_SHADER)
        {
            return;
        }
        else if (smoothing == GAUSSIAN_X_SHADER ||smoothing == GAUSSIAN_X_SHADER)
        {
            /*glUseProgram(glsl_program[GAUSSIAN_X_SHADER]);
            glUniform1i( glGetUniformLocation(glsl_program[GAUSSIAN_X_SHADER], "depthTex"),0);
            glUniform1i( glGetUniformLocation(glsl_program[GAUSSIAN_X_SHADER], "width"),window_width);
            fullscreenQuad();
    
            glUseProgram(glsl_program[GAUSSIAN_Y_SHADER]);
            glUniform1i( glGetUniformLocation(glsl_program[GAUSSIAN_Y_SHADER], "depthTex"),0);
            glUniform1i( glGetUniformLocation(glsl_program[GAUSSIAN_Y_SHADER], "height"),window_height);*/
            return; 
        }
        else if (smoothing == BILATERAL_GAUSSIAN_SHADER)
        {
            glUseProgram(glsl_program[BILATERAL_GAUSSIAN_SHADER]);
            glUniform1i(glGetUniformLocation(glsl_program[BILATERAL_GAUSSIAN_SHADER],"depthTex"),0);
            glUniform1f( glGetUniformLocation(glsl_program[BILATERAL_GAUSSIAN_SHADER], "del_x"),1.0/((float)window_width));
            glUniform1f( glGetUniformLocation(glsl_program[BILATERAL_GAUSSIAN_SHADER], "del_y"),1.0/((float)window_height));
            //glUniform1i(glGetUniformLocation(glsl_program[BILATERAL_GAUSSIAN_SHADER],"width"),window_width);
            //glUniform1i(glGetUniformLocation(glsl_program[BILATERAL_GAUSSIAN_SHADER],"height"),window_height);
        }
        else if (smoothing == CURVATURE_FLOW_SHADER)
        {
            /*glUseProgram(glsl_program[CURVATURE_FLOW_SHADER]);
            glUniform1i(glGetUniformLocation(glsl_program[CURVATURE_FLOW_SHADER],"depthTex"),0);
            glUniform1i(glGetUniformLocation(glsl_program[CURVATURE_FLOW_SHADER],"width"),window_width);
            glUniform1i(glGetUniformLocation(glsl_program[CURVATURE_FLOW_SHADER],"height"),window_height); 
            glUniform1i(glGetUniformLocation(glsl_program[CURVATURE_FLOW_SHADER],"iterations"),40); 
            */
            return;
        }
        fullscreenQuad();
    }

    void Render::orthoProjection()
    {
        glMatrixMode(GL_PROJECTION);                    // Select Projection
        glPushMatrix();                         // Push The Matrix
        glLoadIdentity();                       // Reset The Matrix
        gluOrtho2D( 0,1,0,1);
        glMatrixMode(GL_MODELVIEW);                 // Select Modelview Matrix
        glPushMatrix();                         // Push The Matrix
        glLoadIdentity();                       // Reset The Matrix
    }

    void Render::perspectiveProjection()
    {
        glMatrixMode( GL_PROJECTION );                  // Select Projection
        glPopMatrix();                          // Pop The Matrix
        glMatrixMode( GL_MODELVIEW );                   // Select Modelview
        glPopMatrix();                          // Pop The Matrix
    }

    void Render::fullscreenQuad()
    {
        orthoProjection();
        //glColor3f(1.0f,1.0f,1.0f);
        glBegin(GL_QUADS);
        glTexCoord2f(0.f,0.f);
        //glVertex2f(0,0);
        glVertex2f(0.f,0.f);

        glTexCoord2f(1.f,0.f);
        //glVertex2f(window_width,0);
        glVertex2f(1.f,0.f);

        glTexCoord2f(1.f,1.f);
        //glVertex2f(window_width,window_height);
        glVertex2f(1.f,1.f);

        glTexCoord2f(0.f,1.f);
        //glVertex2f(0,window_height);
        glVertex2f(0.f,1.f);
        glEnd();
        perspectiveProjection();
    }

    void Render::renderPointsAsSpheres()
    {

        glEnable(GL_POINT_SPRITE);
        glTexEnvi(GL_POINT_SPRITE, GL_COORD_REPLACE, GL_TRUE);
        glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);

        if (smoothing == NO_SHADER)
        {
            //FOR FACE TEXTURES
            glEnable(GL_TEXTURE_2D);
            glActiveTexture(GL_TEXTURE0);
        }

        glUseProgram(glsl_program[SPHERE_SHADER]);
        //float particle_radius = 0.125f * 0.5f;
        glUniform1f( glGetUniformLocation(glsl_program[SPHERE_SHADER], "pointScale"), ((float)window_width) / tanf(65. * (0.5f * 3.1415926535f/180.0f)));

        //GE PUT particle_radius in the panel (as a test)
        float radius_scale = settings->getRadiusScale(); //GE
        //glUniform1f( glGetUniformLocation(glsl_program[SPHERE_SHADER], "pointRadius"), particle_radius );
        glUniform1f( glGetUniformLocation(glsl_program[SPHERE_SHADER], "pointRadius"), particle_radius*radius_scale ); //GE
        glUniform1f( glGetUniformLocation(glsl_program[SPHERE_SHADER], "near"), near_depth );
        glUniform1f( glGetUniformLocation(glsl_program[SPHERE_SHADER], "far"), far_depth );

        if (smoothing == NO_SHADER)
        {
            //FOR FACE TEXTURES
            glBindTexture(GL_TEXTURE_2D, gl_tex["texture"]);
        }

        glColor3f(1., 1., 1.);

        drawArrays();

        //copy depth value to a texture
        //glBindTexture(GL_TEXTURE_2D,gl_tex["depth"]);
        //glCopyTexSubImage2D(GL_TEXTURE_2D,0,0,0,0,0,800,600);

        glUseProgram(0);

        glDisable(GL_VERTEX_PROGRAM_POINT_SIZE);
        glDisable(GL_POINT_SPRITE);
    }

    void Render::render_box(float4 min, float4 max)
    {

        glEnable(GL_DEPTH_TEST);
        glColor4f(0.0f, 1.0f, 0.0f, 1.0f);
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
        //glDisable(GL_DEPTH_TEST);

    }

    void Render::render_table(float4 min, float4 max)
    {

        glEnable(GL_DEPTH_TEST);
        glColor4f(0.5f, 0.5f, 0.5f, 1.0f);
        glBegin(GL_TRIANGLE_STRIP);
        float4 scale = float4((0.25f)*(max.x-min.x),(0.25f)*(max.y-min.y),(0.25f)*(max.z-min.z),0.0f);
        glVertex3f(min.x-scale.x, max.y+scale.y, min.z);
        glVertex3f(min.x-scale.x, min.y-scale.y, min.z);
        glVertex3f(max.x+scale.x, max.y+scale.y, min.z);
        glVertex3f(max.x+scale.x, min.y-scale.y, min.z);

        glEnd();
        //glDisable(GL_DEPTH_TEST);
    }

    //----------------------------------------------------------------------
    GLuint Render::compileShaders(const char* vertex_file, const char* fragment_file, const char* geometry_file, GLenum* geom_param, GLint* geom_value, int geom_param_len)
    {

        //this may not be the cleanest implementation
        //#include "shaders.cpp"

        printf("vertex_file: %s\n", vertex_file);
        printf("fragment_file: %s\n", fragment_file);
        //printf("vertex shader:\n%s\n", vertex_shader_source);
        //printf("fragment shader:\n%s\n", fragment_shader_source);
        char *vertex_shader_source = NULL,*fragment_shader_source= NULL,*geometry_shader_source=NULL;
        int vert_size,frag_size,geom_size;
        if (vertex_file)
        {
            vertex_shader_source = file_contents(vertex_file,&vert_size);
            if (!vertex_shader_source)
            {
                printf("Vertex shader file not found or is empty! Cannot compile shader");
                return -1;
            }
        }
        else
        {
            printf("No vertex file specified! Cannot compile shader!");
            return -1;
        }

        if (fragment_file)
        {
            fragment_shader_source = file_contents(fragment_file,&frag_size);
            if (!fragment_shader_source)
            {
                printf("Fragment shader file not found or is empty! Cannot compile shader");
                free(vertex_shader_source);
                return -1;
            }
        }
        else
        {
            printf("No fragment file specified! Cannot compile shader!");
            free(vertex_shader_source);
            return -1;
        }

        if (geometry_file)
        {
            geometry_shader_source = file_contents(fragment_file,&frag_size);
            if (!geometry_shader_source)
            {
                printf("Geometry shader file not found or is empty! Cannot compile shader");
                free(vertex_shader_source);
                free(fragment_shader_source);
                return -1;
            }
        }

        GLint len;
        GLuint program = glCreateProgram();

        GLuint vertex_shader = glCreateShader(GL_VERTEX_SHADER);
        glShaderSource(vertex_shader, 1, (const GLchar**)&vertex_shader_source, 0);
        glCompileShader(vertex_shader);
        glGetShaderiv(vertex_shader, GL_INFO_LOG_LENGTH, &len);
        if (len > 0)
        {
            char log[1024];
            glGetShaderInfoLog(vertex_shader, 1024, 0, log);
            printf("Vertex Shader log:\n %s\n", log);
        }
        glAttachShader(program, vertex_shader);

        GLuint fragment_shader = glCreateShader(GL_FRAGMENT_SHADER);
        glShaderSource(fragment_shader, 1, (const GLchar**)&fragment_shader_source, 0);
        glCompileShader(fragment_shader);
        glGetShaderiv(fragment_shader, GL_INFO_LOG_LENGTH, &len);
        if (len > 0)
        {
            char log[1024];
            glGetShaderInfoLog(fragment_shader, 1024, 0, log);
            printf("Fragment Shader log:\n %s\n", log);
        }
        glAttachShader(program, fragment_shader);


        GLuint geometry_shader=0;
        if (geometry_shader_source)
        {
            geometry_shader = glCreateShader(GL_GEOMETRY_SHADER_EXT);
            glShaderSource(geometry_shader, 1, (const GLchar**)&geometry_shader_source, 0);
            glCompileShader(geometry_shader);
            glGetShaderiv(geometry_shader, GL_INFO_LOG_LENGTH, &len);
            printf("geometry len %d\n", len);
            if (len > 0)
            {
                char log[1024];
                glGetShaderInfoLog(geometry_shader, 1024, 0, log);
                printf("Geometry Shader log:\n %s\n", log);
            }
            glAttachShader(program, geometry_shader);
            for (int i = 0;i < geom_param_len; i++)
            {
                glProgramParameteriEXT(program,geom_param[i],geom_value[i]);
            }
        }

        glLinkProgram(program);

        // check if program linked
        GLint success = 0;
        glGetProgramiv(program, GL_LINK_STATUS, &success);

        if (!success)
        {
            char temp[256];
            glGetProgramInfoLog(program, 256, 0, temp);
            printf("Failed to link program:\n%s\n", temp);
            glDeleteProgram(program);
            program = 0;
        }

        //cleanup
        glDeleteShader(vertex_shader);
        glDeleteShader(fragment_shader);
        if (geometry_shader)
        {
            glDeleteShader(geometry_shader);
        }
        free(vertex_shader_source);
        free(fragment_shader_source);
        free(geometry_shader_source);

        return program;
    }

    int Render::setupTimers()
    {
        //int print_freq = 20000;
        int print_freq = 100; //one second
        int time_offset = 5;

        timers[TI_RENDER]     = new GE::Time("render", time_offset, print_freq);
        if (glsl)
        {
            timers[TI_GLSL]     = new GE::Time("glsl", time_offset, print_freq);
        }
    }

    void Render::printTimers()
    {
        timers[TI_RENDER]->print();
    }


    int Render::generateCircleTexture(GLubyte r, GLubyte g, GLubyte b, GLubyte alpha, int diameter)
    {
        unsigned int imageSize = diameter*diameter*4;
        unsigned int radius = diameter/2;
        GLubyte image[imageSize];
        memset(image,0,imageSize);

        for (unsigned int i = 0; i<imageSize; i+=4)
        {
            int x = ((i/4)%diameter)-(radius);
            int y = (radius)-((i/4)/diameter);
            if ((x*x)+(y*y)<=(radius*radius))
            {
                image[i] = r;
                image[i+1] = g;
                image[i+2] = b;
                image[i+3] = alpha;
            }
            else
            {
                image[i] =0;
                image[i+1] =0;
                image[i+2] =0;
                image[i+3] =0;
            }
        }

        glGenTextures(1, &gl_tex["circle"]);
        glBindTexture(GL_TEXTURE_2D, gl_tex["circle"]);

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);

        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, diameter, diameter, 0,
                     GL_RGBA, GL_UNSIGNED_BYTE, image);

        return 0; //success
    }

    int Render::loadTexture()
    {

        std::string path(GLSL_SOURCE_DIR);
        path += "../../../sprites/tomek.jpg";
        //path += "../../../sprites/enjalot.jpg";
        printf("LOAD TEXTURE!!!!!!!!!!!!!!\n");
        printf("path: %s\n", path.c_str());

        //Load an image with stb_image
        int w,h,channels;
        int force_channels = 0;

        unsigned char *im = stbi_load( path.c_str(), &w, &h, &channels, force_channels );
        printf("after load w: %d h: %d channels: %d\n", w, h, channels);
        if (im == NULL)
        {
            printf("fail!: %s\n", stbi_failure_reason());
            printf("WTF\n");
        }

        //load as gl texture
        glGenTextures(1, &gl_tex["texture"]);
        glBindTexture(GL_TEXTURE_2D, gl_tex["texture"]);

        glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0,
                     GL_RGB, GL_UNSIGNED_BYTE, &im[0]);

        return 0; //success
    }

    void Render::deleteFramebufferTextures()
    {
        glDeleteTextures(1,&gl_tex["depth"]);
        glDeleteTextures(1,&gl_tex["depth2"]);
        glDeleteTextures(1,&gl_tex["thickness"]);
        glDeleteTextures(1,&gl_tex["depthColor"]);
        glDeleteTextures(1,&gl_tex["depthColorSmooth"]);
        glDeleteTextures(1,&gl_tex["normalColor"]);
        glDeleteTextures(1,&gl_tex["lightColor"]);
        glDeleteTextures(1,&gl_tex["Color"]);
    }

    void Render::createFramebufferTextures()
    {
        glGenTextures(1, &gl_tex["depth"]);
        glBindTexture(GL_TEXTURE_2D, gl_tex["depth"]);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexImage2D(GL_TEXTURE_2D,0,GL_DEPTH_COMPONENT32,window_width,window_height,0,GL_DEPTH_COMPONENT,GL_FLOAT,NULL);
        glGenTextures(1, &gl_tex["depth2"]);
        glBindTexture(GL_TEXTURE_2D, gl_tex["depth2"]);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexImage2D(GL_TEXTURE_2D,0,GL_DEPTH_COMPONENT32,window_width,window_height,0,GL_DEPTH_COMPONENT,GL_FLOAT,NULL);
        glGenTextures(1,&gl_tex["thickness"]);
        glBindTexture(GL_TEXTURE_2D, gl_tex["thickness"]);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA,window_width,window_height,0,GL_RGBA,GL_UNSIGNED_BYTE,NULL);
        //glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA32F,window_width,window_height,0,GL_RGBA,GL_FLOAT,NULL);
        glGenTextures(1,&gl_tex["depthColor"]);
        glBindTexture(GL_TEXTURE_2D, gl_tex["depthColor"]);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA,window_width,window_height,0,GL_RGBA,GL_UNSIGNED_BYTE,NULL);
        glGenTextures(1,&gl_tex["normalColor"]);
        glBindTexture(GL_TEXTURE_2D, gl_tex["normalColor"]);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA,window_width,window_height,0,GL_RGBA,GL_UNSIGNED_BYTE,NULL);
        //glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA32F,window_width,window_height,0,GL_RGBA,GL_FLOAT,NULL);
        glGenTextures(1,&gl_tex["lightColor"]);
        glBindTexture(GL_TEXTURE_2D, gl_tex["lightColor"]);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA,window_width,window_height,0,GL_RGBA,GL_UNSIGNED_BYTE,NULL);
        glGenTextures(1,&gl_tex["Color"]);
        glBindTexture(GL_TEXTURE_2D, gl_tex["Color"]);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA,window_width,window_height,0,GL_RGBA,GL_UNSIGNED_BYTE,NULL);



        glGenTextures(1,&gl_tex["depthColorSmooth"]);
        glBindTexture(GL_TEXTURE_2D, gl_tex["depthColorSmooth"]);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA,window_width,window_height,0,GL_RGBA,GL_UNSIGNED_BYTE,NULL);
        //glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA32F,window_width,window_height,0,GL_RGBA,GL_FLOAT,NULL);

    }

    void Render::setWindowDimensions(GLuint width, GLuint height)
    {
        if (glsl)
        {
            deleteFramebufferTextures();
            window_width = width;
            window_height = height;
            createFramebufferTextures();
            glBindFramebuffer(GL_DRAW_FRAMEBUFFER,fbos[0]);
            glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D,gl_tex["thickness"],0);
            glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER,GL_COLOR_ATTACHMENT1,GL_TEXTURE_2D,gl_tex["depthColor"],0);
            glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER,GL_COLOR_ATTACHMENT2,GL_TEXTURE_2D,gl_tex["normalColor"],0);
            glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER,GL_COLOR_ATTACHMENT3,GL_TEXTURE_2D,gl_tex["lightColor"],0);
            glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER,GL_COLOR_ATTACHMENT4,GL_TEXTURE_2D,gl_tex["Color"],0);
            glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER,GL_COLOR_ATTACHMENT5,GL_TEXTURE_2D,gl_tex["depthColorSmooth"],0);
            glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER,GL_DEPTH_ATTACHMENT,GL_TEXTURE_2D,gl_tex["depth"],0);
            glBindFramebuffer(GL_DRAW_FRAMEBUFFER,0);
        }
    }

    void Render::setParticleRadius(float pradius)
    {
        particle_radius = pradius;
    }

}





/*#include <GL/glew.h>
#if defined __APPLE__ || defined(MACOSX)
    #include <GLUT/glut.h>
#else
    #include <GL/glut.h>
#endif*/
#include <GL/glew.h>

#include "SSFRender.h"

using namespace std;
namespace rtps
{
    SSFRender::SSFRender(GLuint pos, GLuint col, int n, CL* cli, RTPSettings* _settings):Render(pos,col,n,cli,_settings)
    {

        fbos.resize(1);
        glGenFramebuffersEXT(1,&fbos[0]);
        smoothing = BILATERAL_GAUSSIAN_SHADER;

        //particle_radius = 0.0125f*0.5f;
        particle_radius = 0.0125f*0.5f;

        createFramebufferTextures();

        cur_depth = "d1";
        glBindFramebufferEXT(GL_DRAW_FRAMEBUFFER_EXT,fbos[0]);
        glFramebufferTexture2DEXT(GL_DRAW_FRAMEBUFFER_EXT,GL_COLOR_ATTACHMENT0_EXT,GL_TEXTURE_2D,gl_framebuffer_texs["thickness"],0);
        glFramebufferTexture2DEXT(GL_DRAW_FRAMEBUFFER_EXT,GL_COLOR_ATTACHMENT1_EXT,GL_TEXTURE_2D,gl_framebuffer_texs["depthColor"],0);
        glFramebufferTexture2DEXT(GL_DRAW_FRAMEBUFFER_EXT,GL_COLOR_ATTACHMENT2_EXT,GL_TEXTURE_2D,gl_framebuffer_texs["normalColor"],0);
        glFramebufferTexture2DEXT(GL_DRAW_FRAMEBUFFER_EXT,GL_COLOR_ATTACHMENT3_EXT,GL_TEXTURE_2D,gl_framebuffer_texs["lightColor"],0);
        glFramebufferTexture2DEXT(GL_DRAW_FRAMEBUFFER_EXT,GL_COLOR_ATTACHMENT4_EXT,GL_TEXTURE_2D,gl_framebuffer_texs["Color"],0);
        glFramebufferTexture2DEXT(GL_DRAW_FRAMEBUFFER_EXT,GL_COLOR_ATTACHMENT5_EXT,GL_TEXTURE_2D,gl_framebuffer_texs["depthColorSmooth"],0);
        glFramebufferTexture2DEXT(GL_DRAW_FRAMEBUFFER_EXT,GL_DEPTH_ATTACHMENT_EXT,GL_TEXTURE_2D,gl_framebuffer_texs[cur_depth],0);
        glBindFramebufferEXT(GL_DRAW_FRAMEBUFFER_EXT,0);


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

        //TODO: All theses shader loads should be pushed to other sub classes or to some sort of shader class
        //string vert(GLSL_BIN_DIR);
        //string frag(GLSL_BIN_DIR);
        
        string vert = shader_source_dir;
        string frag = shader_source_dir;
        printf("vert: %s\n", shader_source_dir.c_str());
        vert+="/sphere_vert.glsl";

        frag+="/sphere_frag.glsl";
        glsl_program[SPHERE_SHADER] = compileShaders(vert.c_str(),frag.c_str());
        vert = shader_source_dir;
        frag = shader_source_dir;
        vert+="/depth_vert.glsl";
        frag+="/depth_frag.glsl";
        glsl_program[DEPTH_SHADER] = compileShaders(vert.c_str(),frag.c_str());
        vert = shader_source_dir;
        frag = shader_source_dir;
        vert+="/gaussian_blur_vert.glsl";
        frag+="/gaussian_blur_x_frag.glsl";
        glsl_program[GAUSSIAN_X_SHADER] = compileShaders(vert.c_str(),frag.c_str());
        frag = shader_source_dir;
        frag+="/gaussian_blur_y_frag.glsl";
        glsl_program[GAUSSIAN_Y_SHADER] = compileShaders(vert.c_str(),frag.c_str());
        frag = shader_source_dir;
        frag+="/gaussian_blur_frag.glsl";
        glsl_program[GAUSSIAN_SHADER] = compileShaders(vert.c_str(),frag.c_str());
        frag = shader_source_dir;
        frag+="/bilateral_blur_frag.glsl";
        glsl_program[BILATERAL_GAUSSIAN_SHADER] = compileShaders(vert.c_str(),frag.c_str());
        frag = shader_source_dir;
        frag+="/curvature_flow_frag.glsl";
        glsl_program[CURVATURE_FLOW_SHADER] = compileShaders(vert.c_str(),frag.c_str());
        vert = shader_source_dir;
        frag = shader_source_dir;
        vert+="/normal_vert.glsl";
        frag+="/normal_frag.glsl";
        glsl_program[NORMAL_SHADER] = compileShaders(vert.c_str(),frag.c_str()); 
        vert = shader_source_dir;
        frag = shader_source_dir;
        vert+="/copy_vert.glsl";
        frag+="/copy_frag.glsl";
        glsl_program[COPY_TO_FB] = compileShaders(vert.c_str(),frag.c_str()); 
    }
    void SSFRender::smoothDepth()
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
        else if (smoothing == GAUSSIAN_X_SHADER ||smoothing == GAUSSIAN_Y_SHADER)
        {
            glUseProgram(glsl_program[GAUSSIAN_X_SHADER]);
            glUniform1i( glGetUniformLocation(glsl_program[GAUSSIAN_X_SHADER], "depthTex"),0);
            glUniform1f( glGetUniformLocation(glsl_program[GAUSSIAN_X_SHADER], "del_x"),1.0f/window_width);
            glUniform1f( glGetUniformLocation(glsl_program[GAUSSIAN_X_SHADER], "sig"),settings->GetSettingAs<float>("blur_scale")); 
            fullscreenQuad();

            glFramebufferTexture2DEXT(GL_DRAW_FRAMEBUFFER_EXT,GL_DEPTH_ATTACHMENT_EXT,GL_TEXTURE_2D,gl_framebuffer_texs["d1"],0);
            glBindTexture(GL_TEXTURE_2D,gl_framebuffer_texs["d2"]);

glUseProgram(glsl_program[GAUSSIAN_Y_SHADER]);
            glUniform1i( glGetUniformLocation(glsl_program[GAUSSIAN_Y_SHADER], "depthTex"),0);
            glUniform1f( glGetUniformLocation(glsl_program[GAUSSIAN_Y_SHADER], "del_y"),1.0f/window_height);
            glUniform1f( glGetUniformLocation(glsl_program[GAUSSIAN_Y_SHADER], "sig"),settings->GetSettingAs<float>("blur_scale"));
            cur_depth = "d1";
            
        }
        else if (smoothing == GAUSSIAN_SHADER)
        {
            glUseProgram(glsl_program[GAUSSIAN_SHADER]);
            glUniform1i(glGetUniformLocation(glsl_program[GAUSSIAN_SHADER],"depthTex"),0);
            glUniform1f( glGetUniformLocation(glsl_program[GAUSSIAN_SHADER], "del_x"),1.0/((float)window_width));
            glUniform1f( glGetUniformLocation(glsl_program[GAUSSIAN_SHADER], "del_y"),1.0/((float)window_height));
            glUniform1f( glGetUniformLocation(glsl_program[GAUSSIAN_SHADER], "sig"),settings->GetSettingAs<float>("blur_scale"));
            cur_depth = "d2";
        }
        else if(smoothing == BILATERAL_GAUSSIAN_SHADER)
        {
            float xdir[] = {1.0f/window_height,0.0f};
            float ydir[] = {0.0f,1.0f/window_width};
            glUseProgram(glsl_program[BILATERAL_GAUSSIAN_SHADER]);
            glUniform1i( glGetUniformLocation(glsl_program[BILATERAL_GAUSSIAN_SHADER], "depthTex"),0);
            glUniform2fv( glGetUniformLocation(glsl_program[BILATERAL_GAUSSIAN_SHADER], "blurDir"),1,xdir);
            glUniform1f( glGetUniformLocation(glsl_program[BILATERAL_GAUSSIAN_SHADER], "blurDepthFallOff"),settings->GetSettingAs<float>("blur_depth_falloff"));
            glUniform1f( glGetUniformLocation(glsl_program[BILATERAL_GAUSSIAN_SHADER], "sig"),settings->GetSettingAs<float>("blur_scale"));
            fullscreenQuad();

            glFramebufferTexture2DEXT(GL_DRAW_FRAMEBUFFER_EXT,GL_DEPTH_ATTACHMENT_EXT,GL_TEXTURE_2D,gl_framebuffer_texs["d1"],0);
            glBindTexture(GL_TEXTURE_2D,gl_framebuffer_texs["d2"]);

            glUseProgram(glsl_program[BILATERAL_GAUSSIAN_SHADER]);
            glUniform1i( glGetUniformLocation(glsl_program[BILATERAL_GAUSSIAN_SHADER], "depthTex"),0);
            glUniform2fv( glGetUniformLocation(glsl_program[BILATERAL_GAUSSIAN_SHADER], "blurDir"),1,ydir);
            glUniform1f( glGetUniformLocation(glsl_program[BILATERAL_GAUSSIAN_SHADER], "blurDepthFallOff"),settings->GetSettingAs<float>("blur_depth_falloff"));
            glUniform1f( glGetUniformLocation(glsl_program[BILATERAL_GAUSSIAN_SHADER], "sig"),settings->GetSettingAs<float>("blur_scale"));
            cur_depth = "d1";
        }
        else if (smoothing == CURVATURE_FLOW_SHADER)
        {
            int iter = settings->GetSettingAs<int>("curvature_flow_iterations");
            float pmat[16];
            glGetFloatv(GL_PROJECTION_MATRIX,pmat);
            float focal_x = (pmat[0]*window_width)/2.0f;
            float focal_y = (pmat[5]*window_height)/2.0f;
            /*for(int i = 0;i<16;i++)
            {
                printf("pmat[%d] = %f\n",i,pmat[i]);
            }
            printf("focal_x = %f, focal_y = %f\n",focal_x,focal_y);*/
            glUseProgram(glsl_program[CURVATURE_FLOW_SHADER]);
            glUniform1i(glGetUniformLocation(glsl_program[CURVATURE_FLOW_SHADER],"depthTex"),0);
            glUniform1i(glGetUniformLocation(glsl_program[CURVATURE_FLOW_SHADER],"width"),window_width);
            glUniform1i(glGetUniformLocation(glsl_program[CURVATURE_FLOW_SHADER],"height"),window_height); 
            glUniform1f( glGetUniformLocation(glsl_program[CURVATURE_FLOW_SHADER], "del_x"),1.0/((float)window_width));
            glUniform1f( glGetUniformLocation(glsl_program[CURVATURE_FLOW_SHADER], "del_y"),1.0/((float)window_height));
            glUniform1f( glGetUniformLocation(glsl_program[CURVATURE_FLOW_SHADER], "focal_x"),focal_x);
            glUniform1f( glGetUniformLocation(glsl_program[CURVATURE_FLOW_SHADER], "focal_y"),focal_y);
            glUniform1f( glGetUniformLocation(glsl_program[CURVATURE_FLOW_SHADER], "dt"),settings->GetSettingAs<float>("curvature_flow_dt"));
            for(int i = 0; i < iter; i++)
            {
                fullscreenQuad();
                glFramebufferTexture2DEXT(GL_DRAW_FRAMEBUFFER_EXT,GL_DEPTH_ATTACHMENT_EXT,GL_TEXTURE_2D,gl_framebuffer_texs[cur_depth],0);
                if(cur_depth=="d1")
                    cur_depth="d2";
                else
                    cur_depth="d1";
                glBindTexture(GL_TEXTURE_2D,gl_framebuffer_texs[cur_depth]);
            }
            return;
        }
        fullscreenQuad();
    }

    void SSFRender::render()
    {

        glPushAttrib(GL_ALL_ATTRIB_BITS);
        glPushClientAttrib(GL_CLIENT_ALL_ATTRIB_BITS);

        //perserve original buffer
        GLint buffer;
        glGetIntegerv(GL_DRAW_BUFFER,&buffer);
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
        glViewport(0, 0, window_width, window_height);
        glScissor(0, 0, window_width, window_height);

        timers["render"]->start();

        
        if (blending)
        {
            //glDepthMask(GL_FALSE);
            glEnable(GL_BLEND);
            glBlendFunc(GL_ONE, GL_ONE);
        }

        
        //glViewport(0, 0, window_width-xywh[0], window_height-xywh[1]);



        glBindFramebufferEXT(GL_DRAW_FRAMEBUFFER_EXT,fbos[0]);
        //glDrawBuffers(2,buffers);
        glDisable(GL_DEPTH_TEST);
        glDepthMask(GL_FALSE);
        glDrawBuffer(GL_COLOR_ATTACHMENT0_EXT);
        glClearColor(0.0f,0.0f,0.0f,0.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        renderPointsAsSpheres();
        glEnable(GL_DEPTH_TEST);
        glDepthMask(GL_TRUE);

        if (blending)
        {
            //glDepthMask(GL_FALSE);
            glDisable(GL_BLEND);
            //glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        }
        //GLenum buffers[] = {GL_COLOR_ATTACHMENT4,GL_COLOR_ATTACHMENT5};
        //Render depth and thickness to a textures
        //glDrawBuffers(2,buffers);
        glDrawBuffer(GL_COLOR_ATTACHMENT4_EXT);
        glClearColor(0.0f,0.0f,0.0f,1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        renderPointsAsSpheres();
        //smoothDepth();


        //Smooth the depth texture to emulate a surface.
        //glDrawBuffer(GL_COLOR_ATTACHMENT1);
        //glBindFramebuffer(GL_DRAW_FRAMEBUFFER,0);
        glDrawBuffer(GL_COLOR_ATTACHMENT5_EXT);

        glFramebufferTexture2DEXT(GL_DRAW_FRAMEBUFFER_EXT,GL_DEPTH_ATTACHMENT_EXT,GL_TEXTURE_2D,gl_framebuffer_texs["d2"],0);
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);
        glActiveTexture(GL_TEXTURE0);
        cur_depth = "d1";
        glBindTexture(GL_TEXTURE_2D,gl_framebuffer_texs[cur_depth]);

        smoothDepth();
        glFramebufferTexture2DEXT(GL_DRAW_FRAMEBUFFER_EXT,GL_DEPTH_ATTACHMENT_EXT,GL_TEXTURE_2D,gl_framebuffer_texs["d1"],0);
        //If no shader was used to smooth then we need the original depth texture
        /*if (smoothing!=NO_SHADER && smoothing!=GAUSSIAN_X_SHADER && smoothing!=BILATERAL_GAUSSIAN_SHADER)
        {
            glBindTexture(GL_TEXTURE_2D,gl_framebuffer_texs["depth2"]);
        }
        else
        {*/
            glBindTexture(GL_TEXTURE_2D,gl_framebuffer_texs[cur_depth]);
        //}
        //glCopyTexSubImage2D(GL_TEXTURE_2D,0,0,0,0,0,800,600);


        glDisable(GL_DEPTH_TEST);
        glDepthMask(GL_FALSE);

        if (blending)
        {
            //glDepthMask(GL_FALSE);
            glEnable(GL_BLEND);
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        }
        //glBindFramebuffer(GL_DRAW_FRAMEBUFFER,fbos[0]);
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D,gl_framebuffer_texs["Color"]);
        //Render the normals for the new "surface".
        glDrawBuffer(GL_COLOR_ATTACHMENT2_EXT);
        glClearColor(0.0f,0.0f,0.0f,1.0f);
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

        glDepthMask(GL_TRUE);
        glEnable(GL_DEPTH_TEST);

        glBindFramebufferEXT(GL_DRAW_FRAMEBUFFER_EXT,0);
        glDrawBuffer(buffer);
        //glDrawBuffer(GL_BACK);

        glViewport(xywh[0],xywh[1],window_width,window_height);
        glScissor(xywh[0], xywh[1], window_width, window_height);



        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D,gl_framebuffer_texs["normalColor"]);
        glActiveTexture(GL_TEXTURE1);
        /*if (smoothing!=NO_SHADER && smoothing!=GAUSSIAN_X_SHADER && smoothing!=BILATERAL_GAUSSIAN_SHADER)
        {
            glBindTexture(GL_TEXTURE_2D,gl_framebuffer_texs["depth2"]);
        }
        else
        {*/
            glBindTexture(GL_TEXTURE_2D,gl_framebuffer_texs[cur_depth]);
        //}
        glUseProgram(glsl_program[COPY_TO_FB]);
        glUniform1i( glGetUniformLocation(glsl_program[COPY_TO_FB], "normalTex"),0);
        glUniform1i( glGetUniformLocation(glsl_program[COPY_TO_FB], "depthTex"),1);
        fullscreenQuad();


        glUseProgram(0);
        glBindTexture(GL_TEXTURE_2D,0);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D,0);
        //printf("done rendering, clean up\n");


        //glDisable(GL_POINT_SMOOTH);
        if (blending)
        {
            glDisable(GL_BLEND);
        }

        glPopClientAttrib();
        glPopAttrib();
        
        //glEnable(GL_LIGHTING);
        glBindBuffer(GL_ARRAY_BUFFER, 0);

        //make sure rendering timing is accurate
        glFinish();

        //printf("done rendering\n");
        timers["render"]->end();
        if (write_framebuffers)
        {
            writeFramebufferTextures();
            write_framebuffers = false;
        }
    }

    void SSFRender::deleteFramebufferTextures()
    {
        glDeleteTextures(1,&gl_framebuffer_texs["d1"]);
        glDeleteTextures(1,&gl_framebuffer_texs["d2"]);
        glDeleteTextures(1,&gl_framebuffer_texs["thickness"]);
        glDeleteTextures(1,&gl_framebuffer_texs["depthColor"]);
        glDeleteTextures(1,&gl_framebuffer_texs["depthColorSmooth"]);
        glDeleteTextures(1,&gl_framebuffer_texs["normalColor"]);
        glDeleteTextures(1,&gl_framebuffer_texs["lightColor"]);
        glDeleteTextures(1,&gl_framebuffer_texs["Color"]);
    }

    void SSFRender::createFramebufferTextures()
    {
        glGenTextures(1, &gl_framebuffer_texs["d1"]);
        glBindTexture(GL_TEXTURE_2D, gl_framebuffer_texs["d1"]);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexImage2D(GL_TEXTURE_2D,0,GL_DEPTH_COMPONENT32,window_width,window_height,0,GL_DEPTH_COMPONENT,GL_FLOAT,NULL);
        glGenTextures(1, &gl_framebuffer_texs["d2"]);
        glBindTexture(GL_TEXTURE_2D, gl_framebuffer_texs["d2"]);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexImage2D(GL_TEXTURE_2D,0,GL_DEPTH_COMPONENT32,window_width,window_height,0,GL_DEPTH_COMPONENT,GL_FLOAT,NULL);
        glGenTextures(1,&gl_framebuffer_texs["thickness"]);
        glBindTexture(GL_TEXTURE_2D, gl_framebuffer_texs["thickness"]);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA,window_width,window_height,0,GL_RGBA,GL_UNSIGNED_BYTE,NULL);
        //glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA32F,window_width,window_height,0,GL_RGBA,GL_FLOAT,NULL);
        glGenTextures(1,&gl_framebuffer_texs["depthColor"]);
        glBindTexture(GL_TEXTURE_2D, gl_framebuffer_texs["depthColor"]);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA,window_width,window_height,0,GL_RGBA,GL_UNSIGNED_BYTE,NULL);
        glGenTextures(1,&gl_framebuffer_texs["normalColor"]);
        glBindTexture(GL_TEXTURE_2D, gl_framebuffer_texs["normalColor"]);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA,window_width,window_height,0,GL_RGBA,GL_UNSIGNED_BYTE,NULL);
        //glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA32F,window_width,window_height,0,GL_RGBA,GL_FLOAT,NULL);
        glGenTextures(1,&gl_framebuffer_texs["lightColor"]);
        glBindTexture(GL_TEXTURE_2D, gl_framebuffer_texs["lightColor"]);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA,window_width,window_height,0,GL_RGBA,GL_UNSIGNED_BYTE,NULL);
        glGenTextures(1,&gl_framebuffer_texs["Color"]);
        glBindTexture(GL_TEXTURE_2D, gl_framebuffer_texs["Color"]);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA,window_width,window_height,0,GL_RGBA,GL_UNSIGNED_BYTE,NULL);



        glGenTextures(1,&gl_framebuffer_texs["depthColorSmooth"]);
        glBindTexture(GL_TEXTURE_2D, gl_framebuffer_texs["depthColorSmooth"]);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA,window_width,window_height,0,GL_RGBA,GL_UNSIGNED_BYTE,NULL);
        //glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA32F,window_width,window_height,0,GL_RGBA,GL_FLOAT,NULL);

    }

    void SSFRender::setWindowDimensions(GLuint width, GLuint height)
    {
        deleteFramebufferTextures();
        window_width = width;
        window_height = height; 
        createFramebufferTextures();
        glBindFramebufferEXT(GL_DRAW_FRAMEBUFFER_EXT,fbos[0]);
        glFramebufferTexture2DEXT(GL_DRAW_FRAMEBUFFER_EXT,GL_COLOR_ATTACHMENT0_EXT,GL_TEXTURE_2D,gl_framebuffer_texs["thickness"],0);
        glFramebufferTexture2DEXT(GL_DRAW_FRAMEBUFFER_EXT,GL_COLOR_ATTACHMENT1_EXT,GL_TEXTURE_2D,gl_framebuffer_texs["depthColor"],0);
        glFramebufferTexture2DEXT(GL_DRAW_FRAMEBUFFER_EXT,GL_COLOR_ATTACHMENT2_EXT,GL_TEXTURE_2D,gl_framebuffer_texs["normalColor"],0);
        glFramebufferTexture2DEXT(GL_DRAW_FRAMEBUFFER_EXT,GL_COLOR_ATTACHMENT3_EXT,GL_TEXTURE_2D,gl_framebuffer_texs["lightColor"],0);
        glFramebufferTexture2DEXT(GL_DRAW_FRAMEBUFFER_EXT,GL_COLOR_ATTACHMENT4_EXT,GL_TEXTURE_2D,gl_framebuffer_texs["Color"],0);
        glFramebufferTexture2DEXT(GL_DRAW_FRAMEBUFFER_EXT,GL_COLOR_ATTACHMENT5_EXT,GL_TEXTURE_2D,gl_framebuffer_texs["depthColorSmooth"],0);
        glFramebufferTexture2DEXT(GL_DRAW_FRAMEBUFFER_EXT,GL_DEPTH_ATTACHMENT_EXT,GL_TEXTURE_2D,gl_framebuffer_texs[cur_depth],0);
        glBindFramebufferEXT(GL_DRAW_FRAMEBUFFER_EXT,0);
    }
};

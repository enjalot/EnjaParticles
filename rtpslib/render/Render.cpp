/****************************************************************************************
* Real-Time Particle System - An OpenCL based Particle system developed to run on modern GPUs. Includes SPH fluid simulations.
* version 1.0, September 14th 2011
* 
* Copyright (C) 2011 Ian Johnson, Andrew Young, Gordon Erlebacher, Myrna Merced, Evan Bollig
* 
* This software is provided 'as-is', without any express or implied
* warranty.  In no event will the authors be held liable for any damages
* arising from the use of this software.
* 
* Permission is granted to anyone to use this software for any purpose,
* including commercial applications, and to alter it and redistribute it
* freely, subject to the following restrictions:
* 
* 1. The origin of this software must not be misrepresented; you must not
* claim that you wrote the original software. If you use this software
* in a product, an acknowledgment in the product documentation would be
* appreciated but is not required.
* 2. Altered source versions must be plainly marked as such, and must not be
* misrepresented as being the original software.
* 3. This notice may not be removed or altered from any source distribution.
****************************************************************************************/


#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include <GL/glew.h>

#include "Render.h"
#include "util.h"
#include "stb_image.h" 
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h" 

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
     
        shader_source_dir = settings->GetSettingAs<string>("rtps_path");
        shader_source_dir += "/shaders";
        printf("SHADER SOURCE DIR\n", shader_source_dir.c_str());

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
        GLubyte col1[] = {0,0,0,255};
        GLubyte col2[] = {255,255,255,255};

        generateCheckerBoardTex(col1,col2,8,640);
        printf("GL VERSION %s\n", glGetString(GL_VERSION));
        //blending = settings.GetSettingAs<bool>("Render: Blending");
        //blending = settings->getUseAlphaBlending();
        blending = settings->GetSettingAs<bool>("render_use_alpha");
        setupTimers();
    }

    //----------------------------------------------------------------------
    Render::~Render()
    {
        printf("Render destructor\n");
        for (map<ShaderType,GLuint>::iterator i = glsl_program.begin();i!=glsl_program.end();i++)
        {
            glDeleteProgram(i->second);
        }


        for (map<string,GLuint>::iterator i = gl_framebuffer_texs.begin();i!=gl_framebuffer_texs.end();i++)
        {
            glDeleteTextures(1,&(i->second));
        }
        //for(vector<GLuint>::iterator i=rbos.begin(); i!=rbos.end();i++)
        //{
        if (rbos.size())
        {
            glDeleteRenderbuffersEXT(rbos.size() ,&rbos[0]);
        }
        //}
        //for(vector<GLuint>::iterator i=fbos.begin(); i!=fbos.end();i++)
        //{
        if (fbos.size())
        {
            glDeleteFramebuffersEXT(fbos.size(),&fbos[0]);
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
        timers["render"]->start();

        glPushAttrib(GL_ALL_ATTRIB_BITS);
        glPushClientAttrib(GL_CLIENT_ALL_ATTRIB_BITS);

		//printf("*** before renderPointCloud\n");
        glDepthMask(GL_TRUE);
		glEnable(GL_LIGHTING);
#ifdef CLOUD_COLLISION
		renderPointCloud(); //GE
#endif
		glDisable(GL_LIGHTING);
        glDepthMask(GL_FALSE);


        if (blending)
        {
            //glDisable(GL_DEPTH_TEST);
            glDepthMask(GL_FALSE);
            glEnable(GL_BLEND);
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        }
        else
        {
            //glEnable(GL_DEPTH_TEST);
        }

        //glBlendFunc(GL_SRC_ALPHA, GL_ONE);

        // draws circles instead of squares
        glEnable(GL_POINT_SMOOTH); 
        //TODO make the point size a setting
        glPointSize(5.0f);

        drawArrays();
        //printf("done rendering, clean up\n");

        glDepthMask(GL_TRUE);

        glDisable(GL_LIGHTING);

        glPopClientAttrib();
        glPopAttrib();
        //glDisable(GL_POINT_SMOOTH);
        if (blending)
        {
            glDisable(GL_BLEND);
        }
        //glEnable(GL_LIGHTING);
        glBindBuffer(GL_ARRAY_BUFFER, 0);


        //printf("done rendering\n");
        timers["render"]->end();

        //make sure rendering timing is accurate
        glFinish();
    }

#ifdef CLOUD_COLLISION
	//----------------------------------------------------------------------
	void Render::renderPointCloud()
	{
        glPointSize(0.1f);
		int sz = cloud_positions->size();
#if 1
		glBegin(GL_POINTS);
        glColor3f(.5, .5, .5);
        for (int i=0; i < cloud_num; i++) 
        {
            float4& f = (*cloud_positions)[i];
            //printf("v: %f, %f, %f\n", (float) f.x, (float) f.y, (float) f.z);
            if (f.x < 0 || f.y < 0 || f.z < 0) continue;
            if (f.x > 5 || f.y > 5 || f.z > 5) continue;
            glVertex3f(f.x, f.y, f.z);
        }
		glEnd();
		//exit(0);
#endif

#if 1
// Something wrong with the faces when the arm is moving

    int nb_faces = cloud_faces->size();

    if (nb_faces > 0) 
    {
        //printf("nb_faces= %d\n", nb_faces);
        glBegin(GL_QUADS);
            for (int i=0; i < nb_faces; i++) {
                int4& vertices = (*cloud_faces)[i];
                int4& normals = (*cloud_faces_normals)[i];
                //vertices.print("v");
                //normals.print("n");
                //printf("ver: %d, %d, %d, %d\n", vertices.x, vertices.y, vertices.z, vertices.w);
                float4& v1 = (*cloud_positions)[vertices.x];
                float4& n1 = (*cloud_normals)[normals.x];
                //v1.print("v1");
                //n1.print("n1");
                glNormal3f(n1.x, n1.y, n1.z);
                glVertex3f(v1.x, v1.y, v1.z);

                float4& v2 = (*cloud_positions)[vertices.y];
                float4& n2 = (*cloud_normals)[normals.y];
                //v2.print("v2");
                //n2.print("n2");
                glNormal3f(n2.x, n2.y, n2.z);
                glVertex3f(v2.x, v2.y, v2.z);

                float4& v3 = (*cloud_positions)[vertices.z];
                float4& n3 = (*cloud_normals)[normals.z];
                //v3.print("v3");
                //n3.print("n3");
                glNormal3f(n3.x, n3.y, n3.z);
                glVertex3f(v3.x, v3.y, v3.z);

                float4& v4 = (*cloud_positions)[vertices.w];
                float4& n4 = (*cloud_normals)[normals.w];
                //v4.print("v4");
                //n4.print("n4");
                glNormal3f(n4.x, n4.y, n4.z);
                glVertex3f(v4.x, v4.y, v4.z);
            }
        glEnd();
	}
#endif
}
#endif  //CLOUD_COLLISION
	//----------------------------------------------------------------------

    void Render::writeBuffersToDisk()
    {
        write_framebuffers = true;
    }


    void Render::writeFramebufferTextures() 
    {
        for (map<string,GLuint>::iterator i = gl_framebuffer_texs.begin();i!=gl_framebuffer_texs.end();i++)
        {
            string s(i->first);
            s+=".png";
            writeTexture(i->second, s.c_str());
        }
    }

    void Render::convertDepthToRGB(const GLfloat* depth, GLuint size, GLubyte* rgba) const
    {
        GLfloat minimum = 1.0f;
        for (GLuint i = 0;i<size;i++)
        {
            if (minimum>depth[i])
            {
                minimum = depth[i];
            }
        }
        GLfloat one_minus_min = 1.f-minimum;
        for (GLuint i = 0;i<size;i++)
        {
            for (GLuint j = 0;j<3;j++)
            {
                rgba[(i*4)+j]=(GLubyte)(((depth[i]-minimum)/one_minus_min) *255U);
            }
            rgba[(i*4)+3] = 255U; //no transparency;
        }
    }

    int Render::writeTexture( GLuint tex, const char* filename) const
    {
        printf("writing %s texture to disc.\n",filename);
        glBindTexture(GL_TEXTURE_2D,tex);
        GLubyte* image = new GLubyte[window_width*window_height*4];
        if (!strcmp(filename,"depth.png") || !strcmp(filename,"depth2.png"))
        {
            GLfloat* fimg = new GLfloat[window_width*window_height];
            glGetTexImage(GL_TEXTURE_2D,0,GL_DEPTH_COMPONENT,GL_FLOAT,fimg);
            convertDepthToRGB(fimg,window_width*window_height,image);
	    delete[] fimg;
        }
        else
        {
            glGetTexImage(GL_TEXTURE_2D,0,GL_RGBA,GL_UNSIGNED_BYTE,image);
        }
        if (!stbi_write_png(filename,window_width,window_height,4,(void*)image,0))
        {
            printf("failed to write image %s",filename);
            return -1;
        }

        glBindTexture(GL_TEXTURE_2D,0);
        delete[] image;

        return 0;
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
        glBegin(GL_QUADS);
        glTexCoord2f(0.f,0.f);
        glVertex2f(0.f,0.f);

        glTexCoord2f(1.f,0.f);
        glVertex2f(1.f,0.f);

        glTexCoord2f(1.f,1.f);
        glVertex2f(1.f,1.f);

        glTexCoord2f(0.f,1.f);
        glVertex2f(0.f,1.f);
        glEnd();
        perspectiveProjection();
    }

	//----------------------------------------------------------------------
    void Render::renderPointsAsSpheres()
    {

        glEnable(GL_POINT_SPRITE);
        glTexEnvi(GL_POINT_SPRITE, GL_COORD_REPLACE, GL_TRUE);
        glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);

        glUseProgram(glsl_program[SPHERE_SHADER]);
        //float particle_radius = 0.125f * 0.5f;
        glUniform1f( glGetUniformLocation(glsl_program[SPHERE_SHADER], "pointScale"), ((float)window_width) / tanf(65. * (0.5f * 3.1415926535f/180.0f)));

        //GE PUT particle_radius in the panel (as a test)
        float radius_scale = settings->getRadiusScale(); //GE
        //glUniform1f( glGetUniformLocation(glsl_program[SPHERE_SHADER], "pointRadius"), particle_radius );
        glUniform1f( glGetUniformLocation(glsl_program[SPHERE_SHADER], "pointRadius"), particle_radius*radius_scale ); //GE
        glUniform1f( glGetUniformLocation(glsl_program[SPHERE_SHADER], "near"), near_depth );
        glUniform1f( glGetUniformLocation(glsl_program[SPHERE_SHADER], "far"), far_depth );

        //glColor3f(1., 1., 1.);

        drawArrays();

        glUseProgram(0);

        glDisable(GL_VERTEX_PROGRAM_POINT_SIZE);
        glDisable(GL_POINT_SPRITE);
    }
	//----------------------------------------------------------------------

    void Render::render_box(float4 min, float4 max)
    {

        glEnable(GL_DEPTH_TEST);
        glColor4f(.0f, 1.0f, .0f, 1.0f);
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
        glColor4f(0.0f, 0.4f, 0.0f, 1.0f);
        //glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER,GL_COLOR_ATTACHMENT2,GL_TEXTURE_2D,0,0);
        //glBindTexture(GL_TEXTURE_2D,gl_framebuffer_texs["normalColor"]);
        //glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER,GL_COLOR_ATTACHMENT2,GL_TEXTURE_2D,gl_framebuffer_texs["normalColor"],0);
        //glBindTexture(GL_TEXTURE_2D,gl_textures["checker_board"]);
        //glBegin(GL_TRIANGLE_STRIP);
        glBegin(GL_QUADS);
        float4 scale = float4((0.25f)*(max.x-min.x),(0.25f)*(max.y-min.y),(0.25f)*(max.z-min.z),0.0f);
        /*glTexCoord2f(0.f,0.f);
        glVertex3f(min.x-scale.x, min.y-scale.y, min.z);
        glTexCoord2f(1.f,0.f);
        glVertex3f(max.x+scale.x, min.y-scale.y, min.z);
        glTexCoord2f(1.f,1.f);
        glVertex3f(max.x+scale.x, max.y+scale.y, min.z);
        glTexCoord2f(0.f,1.f);
        glVertex3f(min.x-scale.x, max.y+scale.y, min.z);*/
        glTexCoord2f(0.f,0.f);
        glVertex3f(-10000., -10000., min.z);
        glTexCoord2f(1.f,0.f);
        glVertex3f(10000., -10000., min.z); 
        glTexCoord2f(1.f,1.f);
        glVertex3f(10000., 10000., min.z);
        glTexCoord2f(0.f,1.f);
        glVertex3f(-10000., 10000., min.z);
        glEnd();
        glBindTexture(GL_TEXTURE_2D,0);
        //glDisable(GL_DEPTH_TEST);
    }

	//----------------------------------------------------------------------
    int Render::generateCheckerBoardTex(GLubyte* color1,GLubyte* color2,int num_squares, int length)
    {
        unsigned int imageSize = length*length;
        GLubyte* image = new GLubyte[imageSize*4];
        memset(image,0,imageSize);
        int sq_size = length/num_squares;
        GLubyte* col;
        for (unsigned int i = 0; i<imageSize; i++)
        {
            if ((i/sq_size)%2 && (i/sq_size))
            {
                col = color1;
            }
            else
            {
                col = color2;
            }
            for(int j = 0; j<4; j++)
            {
                image[(i*4)+j] = col[j];
            }
        }

        glGenTextures(1, &gl_textures["checker_board"]);
        glBindTexture(GL_TEXTURE_2D, gl_textures["checker_board"]);

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);

        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, length, length, 0,
                     GL_RGBA, GL_UNSIGNED_BYTE, image);
		delete[] image;
        return 0; //success
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
        //int print_freq = 100; //one second
        int time_offset = 5;

        //timers[TI_RENDER]     = new GE::Time("render", time_offset, print_freq);
        timers["render"] = new EB::Timer("Render call", time_offset);
		return 0;
    }

    void Render::printTimers()
    {
        //timers[TI_RENDER]->print();
        timers.printAll();
    }


    int Render::generateCircleTexture(GLubyte r, GLubyte g, GLubyte b, GLubyte alpha, int diameter)
    {
        unsigned int imageSize = diameter*diameter*4;
        unsigned int radius = diameter/2;
        GLubyte* image = new GLubyte[imageSize];
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

        glGenTextures(1, &gl_textures["circle"]);
        glBindTexture(GL_TEXTURE_2D, gl_textures["circle"]);

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);

        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, diameter, diameter, 0,
                     GL_RGBA, GL_UNSIGNED_BYTE, image);
        delete[] image;
        return 0; //success
    }

    int Render::loadTexture(string texture_file, string texture_name)
    {

        //std::string path(GLSL_SOURCE_DIR);
        //path += "../../../sprites/boid.png";
        //path += "../../../sprites/enjalot.jpg";
        printf("LOAD TEXTURE!!!!!!!!!!!!!!\n");
        //printf("path: %s\n", path.c_str());

        //Load an image with stb_image
        int w,h,channels;
        int force_channels = 0;

        unsigned char *im = stbi_load( texture_file.c_str(), &w, &h, &channels, force_channels );
        printf("after load w: %d h: %d channels: %d\n", w, h, channels);
        printf("im looking for the image at %s\n", texture_file.c_str());
        if (im == NULL)
        {
            printf("fail!: %s\n", stbi_failure_reason());
            printf("WTF\n");
        }

        //load as gl texture
        glGenTextures(1, &gl_textures[texture_name]);
        glBindTexture(GL_TEXTURE_2D, gl_textures[texture_name]);

        glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        
        //better way to do this?
        if(channels == 3)
        {
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0,
                     GL_RGB, GL_UNSIGNED_BYTE, &im[0]);
        }
        else if (channels == 4)
        {
             glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0,
                  GL_RGBA, GL_UNSIGNED_BYTE, &im[0]);
        }

        glBindTexture(GL_TEXTURE_2D,0);
        free(im);
        return 0; //success
    }

    void Render::deleteFramebufferTextures()
    {

    }

    void Render::createFramebufferTextures()
    {

    }

    void Render::setWindowDimensions(GLuint width, GLuint height)
    {
        window_width = width;
        window_height = height;
    }

    void Render::setParticleRadius(float pradius)
    {
        particle_radius = pradius;
    }

}



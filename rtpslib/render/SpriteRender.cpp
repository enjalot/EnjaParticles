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



#include "GL/glew.h"

#include "SpriteRender.h"

using namespace std;
namespace rtps
{
    SpriteRender::SpriteRender(GLuint pos, GLuint col, int n, CL* cli, RTPSettings* _settings):Render(pos,col,n,cli,_settings)
    {
        string path(GLSL_SOURCE_DIR);
        string filename = settings->GetSettingAs<string>("render_texture");
        //path += "../../../sprites/fsu_seal.jpg";
        //path += "../../../sprites/firejet_blast.png";   //borrowed from http://homepage.mac.com/nephilim/sw3ddev/additive_blending.html
        path = "../../sprites/" + filename;
        //path += "../../../sprites/firejet_smoke.png";
        //path += "../../../sprites/tomek.jpg";
        //path += "../../../sprites/enjalot.jpg";
        printf("LOAD TEXTURE!!!!!!!!!!!!!!\n");
        printf("filename: %s\n", filename.c_str());
        printf("path: %s\n", path.c_str());
        settings->printSettings();
        loadTexture(path, "texture");
        //string vert(GLSL_BIN_DIR);
        //string frag(GLSL_BIN_DIR);
        //vert+="/sphere_vert.glsl";
        //frag+="/sphere_tex_frag.glsl";
        string vert = shader_source_dir + "/sprite_vert.glsl";
        string frag = shader_source_dir + "/" + settings->GetSettingAs<string>("render_frag_shader");

        glsl_program[SPHERE_SHADER] = compileShaders(vert.c_str(),frag.c_str());
    }
    void SpriteRender::render()
    {

        glPushAttrib(GL_ALL_ATTRIB_BITS);
        glPushClientAttrib(GL_CLIENT_ALL_ATTRIB_BITS);
        //printf("BLENDING: %d\n", blending);


        if (blending)
        {
            glDisable(GL_DEPTH_TEST);
            glDepthMask(GL_FALSE);
            glEnable(GL_BLEND);
            string afunc = settings->GetSettingAs<string>("render_alpha_function");
            if(afunc == "alpha")
                glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
            else if(afunc == "add")
                glBlendFunc(GL_SRC_ALPHA, GL_ONE);
            else if(afunc == "multiply")
            {
                glBlendFunc(GL_DST_COLOR, GL_ZERO);
                //glBlendFunc(GL_ZERO, GL_SRC_COLOR);
                //glBlendColor(.9, .9, .9, 1.);
                //glBlendFunc(GL_CONSTANT_COLOR, GL_ZERO);
                //glBlendFunc(GL_ONE_MINUS_SRC_ALPHA, GL_ONE);
            }
        }

        glDisable(GL_LIGHTING);
        //glBlendFunc(GL_SRC_ALPHA, GL_ONE);





        glBindTexture(GL_TEXTURE_2D,gl_textures["texture"]);
        renderPointsAsSpheres();
        glBindTexture(GL_TEXTURE_2D,0);


        //glDepthMask(GL_TRUE);

        //glDisable(GL_POINT_SMOOTH);
        if (blending)
        {
            glEnable(GL_DEPTH_TEST);
            glDisable(GL_BLEND);
        }
        glBindBuffer(GL_ARRAY_BUFFER, 0);
 
        glPopClientAttrib();
        glPopAttrib();
        
        glFinish();
 

    }

}

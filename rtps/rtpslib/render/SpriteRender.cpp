
#include "GL/glew.h"

#include "SpriteRender.h"

using namespace std;
namespace rtps
{
    SpriteRender::SpriteRender(GLuint pos, GLuint col, int n, CL* cli, RTPSettings* _settings):Render(pos,col,n,cli,_settings)
    {
        string path(GLSL_SOURCE_DIR);
        //path += "../../../sprites/fsu_seal.jpg";
        path += "../../../sprites/firejet_blast.png";   //borrowed from http://homepage.mac.com/nephilim/sw3ddev/additive_blending.html
        //path += "../../../sprites/firejet_smoke.png";
        //path += "../../../sprites/tomek.jpg";
        //path += "../../../sprites/enjalot.jpg";
        printf("LOAD TEXTURE!!!!!!!!!!!!!!\n");
        printf("path: %s\n", path.c_str());
        loadTexture(path, "texture");
        string vert(GLSL_BIN_DIR);
        string frag(GLSL_BIN_DIR);
        //vert+="/sphere_vert.glsl";
        //frag+="/sphere_tex_frag.glsl";
        vert+="/sprite_vert.glsl";
        frag+="/sprite_tex_frag.glsl";

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
            //glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
            glBlendFunc(GL_SRC_ALPHA, GL_ONE);
            //glBlendFunc(GL_DST_COLOR, GL_ZERO);
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

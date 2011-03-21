#include "SpriteRender.h"

using namespace std;
namespace rtps
{
    SpriteRender::SpriteRender(GLuint pos, GLuint col, int n, CL* cli, RTPSettings* _settings):Render(pos,col,n,cli,_settings)
    {
        string path(GLSL_SOURCE_DIR);
        path += "../../../sprites/tomek.jpg";
        //path += "../../../sprites/enjalot.jpg";
        printf("LOAD TEXTURE!!!!!!!!!!!!!!\n");
        printf("path: %s\n", path.c_str());
        loadTexture(path, "texture");
        string vert(GLSL_BIN_DIR);
        string frag(GLSL_BIN_DIR);
        vert+="/sphere_vert.glsl";
        frag+="/sphere_tex_frag.glsl";
        glsl_program[SPHERE_SHADER] = compileShaders(vert.c_str(),frag.c_str());
    }
    void SpriteRender::render()
    {
        glBindTexture(GL_TEXTURE_2D,gl_textures["texture"]);
        renderPointsAsSpheres();
        glBindTexture(GL_TEXTURE_2D,0);
    }
}

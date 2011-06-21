#include <GL/glew.h>

#include "MeshRender.h"

using namespace std;
namespace rtps
{
    MeshRender::MeshRender(GLuint pos, GLuint col, int n, CL* cli, RTPSettings* _settings):Render(pos,col,n,cli,_settings)
    {
        string path(GLSL_SOURCE_DIR);
        string vert = shader_source_dir + "/triangle_vert.glsl";
        string frag = shader_source_dir + "/triangle_frag.glsl"; 
        string geom = shader_source_dir + "/triangle_geom.glsl";
        GLenum geom_params[] = {GL_GEOMETRY_INPUT_TYPE_EXT, GL_GEOMETRY_OUTPUT_TYPE_EXT, GL_GEOMETRY_VERTICES_OUT_EXT};
        GLint geom_values[] = {GL_POINTS,GL_TRIANGLE_STRIP,3}; 
        glsl_program[MESH_TRIANGLES] = compileShaders(vert.c_str(),frag.c_str(),geom.c_str(),geom_params,geom_values,3);
    }

    void MeshRender::render()
    {
        glPushAttrib(GL_ALL_ATTRIB_BITS);
        glPushClientAttrib(GL_CLIENT_ALL_ATTRIB_BITS);
        //printf("BLENDING: %d\n", blending);

        glDisable(GL_LIGHTING);

        printf("here\n");
        glUseProgram(glsl_program[MESH_TRIANGLES]);
        drawArrays();
        glUseProgram(0);


        glBindBuffer(GL_ARRAY_BUFFER, 0);

        glPopClientAttrib();
        glPopAttrib();

        glFinish();
    }
}

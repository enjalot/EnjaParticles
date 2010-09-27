#ifndef RTPS_RENDER_H_INCLUDED
#define RTPS_RENDER_H_INCLUDED

#if defined __APPLE__ || defined(MACOSX)
    //OpenGL stuff
    #include <OpenGL/gl.h>
#else
    //OpenGL stuff
    #include <GL/gl.h>
#endif

#include "../structs.h"

namespace rtps{

class Render
{
public:
    Render(GLuint pos_vbo, GLuint vel_vbo, int num);
    ~Render();

    //decide which kind of rendering to use
    enum RenderType {POINTS, SPRITES};
    RenderType rtype;

    //number of particles
    int num;

    GLuint pos_vbo;
    GLuint col_vbo;

    void render();
    void drawArrays();

    void render_box(float4 min, float4 max);

    //void compileShaders();

};

}

#endif

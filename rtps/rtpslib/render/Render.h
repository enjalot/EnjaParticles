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
#include "../timege.h"

namespace rtps{

class Render
{
public:
    Render(GLuint pos_vbo, GLuint vel_vbo, int num);
    ~Render();

    //decide which kind of rendering to use
    enum RenderType {POINTS, SPRITES};

    void setNum(int nn){num = nn;};

    void render();
    void drawArrays();

    void render_box(float4 min, float4 max);

    enum {TI_RENDER=0, TI_GLSL 
          }; //2
    GE::Time* timers[2];
    int setupTimers();

    //void compileShaders();

private:
    //number of particles
    int num;

    RenderType rtype;
    bool glsl;
    bool mikep;
    GLuint glsl_program;    

    GLuint pos_vbo;
    GLuint col_vbo;

    GLuint compileShaders();
    GLuint mpShaders();

};


}

#endif

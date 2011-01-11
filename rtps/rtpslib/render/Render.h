#ifndef RTPS_RENDER_H_INCLUDED
#define RTPS_RENDER_H_INCLUDED

#include <map>

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

enum Shaders {SHADER_DEPTH=0,SHADER_CURVATURE_FLOW,SHADER_FRESNEL};

class Render
{
public:
    Render(GLuint pos_vbo, GLuint vel_vbo, int num);
    ~Render();

    //decide which kind of rendering to use
    enum RenderType {POINTS, SPRITES};
	enum ShaderType {SPHERE_SHADER, DEPTH_SHADER, MIKEP_SHADER};

    void setNum(int nn){num = nn;};

    void render();
    void drawArrays();

	void renderPointsAsSpheres();
	void orthoProjection();
	void perspectiveProjection();
	void fullscreenQuad();

    void render_box(float4 min, float4 max);
    void render_table(float4 min, float4 max);

    enum {TI_RENDER=0, TI_GLSL}; //2
    GE::Time* timers[2];
    int setupTimers();

    //void compileShaders();

private:
    //number of particles
    int num;

    RenderType rtype;
    bool glsl;
    bool mikep;
    bool blending;
    std::map<ShaderType,GLuint> glsl_program;    
    std::map<std::string,GLuint> gl_tex;

    GLuint pos_vbo;
    GLuint col_vbo;

    GLuint compileShaders(const char* vertex_file, const char* fragment_file, const char* geometry_file = NULL);
    GLuint mpShaders();
    int loadTexture();
	int generateCircleTexture(GLubyte r, GLubyte g, GLubyte b, GLubyte alpha, int diameter);
};	


}

#endif

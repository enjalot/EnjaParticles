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
#include "../opencl/CLL.h"
#include "../opencl/Kernel.h"
#include "../opencl/Buffer.h"

namespace rtps{

enum Shaders {SHADER_DEPTH=0,SHADER_CURVATURE_FLOW,SHADER_FRESNEL};

class Render
{
public:
    Render(GLuint pos_vbo, GLuint vel_vbo, int num, CL *cli);
    ~Render();

    //decide which kind of rendering to use
    enum RenderType {POINTS, SPRITES};
	enum ShaderType {NO_SHADER,SPHERE_SHADER, DEPTH_SHADER, GAUSSIAN_X_SHADER, GAUSSIAN_Y_SHADER, BILATERAL_GAUSSIAN_SHADER, NORMAL_SHADER, CURVATURE_FLOW_SHADER, MIKEP_SHADER, COPY_TO_FB};

    void setNum(int nn){num = nn;}
	void setDepthSmoothing(ShaderType shade){ smoothing = shade;}
	void setWindowDimensions(GLuint width,GLuint height);
	void setParticleRadius(float pradius);
	
    void render();
    void drawArrays();

	void renderPointsAsSpheres();
	void smoothDepth();

	void orthoProjection();
	void perspectiveProjection();
	void fullscreenQuad();

    void render_box(float4 min, float4 max);
    void render_table(float4 min, float4 max);

    enum {TI_RENDER=0, TI_GLSL}; //2
    GE::Time* timers[2];
    int setupTimers();
    void printTimers();

    //void compileShaders();

private:
    //number of particles
    int num;

    RenderType rtype;
    bool glsl;
    bool mikep;
    bool blending;
	ShaderType smoothing;
    std::map<ShaderType,GLuint> glsl_program;    
    std::map<std::string,GLuint> gl_tex;
	std::vector<GLuint> fbos;
	std::vector<GLuint> rbos;
	Buffer<float>	cl_depth;
	Kernel	k_curvature_flow;
	GLuint window_height,window_width;
	float particle_radius;
	float near_depth;
	float far_depth;

    GLuint pos_vbo;
    GLuint col_vbo;
	CL *cli;

    GLuint compileShaders(const char* vertex_file, const char* fragment_file, const char* geometry_file = NULL, GLenum* geom_param=NULL, GLint* geom_value=NULL, int geom_param_len=0);
    int loadTexture();
	int generateCircleTexture(GLubyte r, GLubyte g, GLubyte b, GLubyte alpha, int diameter);
	void deleteFramebufferTextures();
	void createFramebufferTextures();
};	


}

#endif

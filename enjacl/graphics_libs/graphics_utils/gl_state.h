////////////////////////////////////////////////////////////////////////////
//
// gl_state.h
//
// 2006 Gordon Erlebacher
// Meant to work with OpenGL 2.0
//
////////////////////////////////////////////////////////////////////////////

#ifndef _GL_STATE_H_
#define _GL_STATE_H_

//#include <Cg/cg.h>
//#include <Cg/cgGL.h>
#include <vector>
#include "glincludes.h"

class TextureOGL;
class TextureOGL1D;

namespace CG {

class Program;

// all classes in GL, could be static method inside PROGRAM

class GL {
private:
	static int nb_instances; // insure that only once instance is activated
	int maxPrograms; // maximum number of allowed shaders (set to 100)
	Program** shader;
	//std::vector<Program*> program; 

public:
	enum SHADER_TYPE {FRAG_SHADER=0, VERT_SHADER, BOTH_SHADERS};
	static GLuint shader_program;

	/// constructor. Progarm exists if GL is instantiated more than once. 
	GL(int maxPrograms);
	Program* getShader(int i) {
		return shader[i];
	}
	GLuint setupVertexShader(char* filename);
	GLuint setupFragmentShader(char* filename);
	GLuint setupShaderProgram();
	CG::Program& setupShaderProgram(char* name, GLuint *shader_id, int which=BOTH_SHADERS);

     static void init(bool use_ogl_arb = true);
     static void destroy();

     static void enable_vp();
     static void disable_vp();
     static void enable_fp();
     static void disable_fp();

     static void print_gl_info();

private:
	GL();
};

class Program
{
public:
	 GLuint shader_program;  // one can have multiple shader programs

private:
	 // in this model: only a single shader_program per file.
	 // only a single file associated with each shader program

	 //GLuint   s, f, v;  // vertex, fragment shader (s is either)

public:
	Program();
     void load_from_string(const char* str, bool vertex_prg,
			   const char* entry = 0);
     bool load_from_file  (const char* filename, bool vertex_prg,
			   const char* entry = 0);
     bool load            (const char* filename, bool vertex_prg,
			   const char* entry = 0)
	  { return load_from_file(filename, vertex_prg, entry); }

	 void addFragmentShader(char* filename); 
	 void addVertexShader(char* filename); 
	 void link(); // after all shaders are added

	float get_param1(const char* name);

     void set_param1(const char* name, int x);
     void set_param1(const char* name, float x);
     void set_param1(const char* name, const float* v);
     void set_param2(const char* name, float x, float y);
     void set_param2(const char* name, const float* v);
     void set_param3(const char* name, float x, float y, float z);
     void set_param3(const char* name, const float* v);
     void set_param4(const char* name, float x, float y, float z, float w);
     void set_param4(const char* name, const float* v);
     bool set_tex   (const char* name, TextureOGL1D& tex_obj, int texUnit=0);
     bool set_tex   (const char* name, TextureOGL& tex_obj, int texUnit=0);
     bool set_tex   (const char* name, GLuint tex_obj);
     bool set_tex   (const char* name, GLuint tex_id, int texUnit=0);
	bool set_tex_2D(const char* name, GLuint tex_obj, int texUnit=0);
	 bool set_tex1d(const char* name, unsigned int tex_obj, int texUnit=0);

	void set_binding_buffer(const char* name, GLuint buffer);

	 GLuint getId() {
	 	return shader_program; // a shader program contains one or multiple vertex/fragment shaders to be linked
	 }

     void valid(); // not used
	void checkError(char* msg);
};

}  // end namespace

#endif

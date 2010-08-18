// save the location of all variables in shaders in special 
// variables. Idially, I should create classes that wrap 
// the shader variales I am going to use, but seems like 
// overkill
////////////////////////////////////////////////////////////////////////////
//
// gl_state.cpp
//
// 2006 Gordon Erlebacher
// Meant to work with OpenGL 2.0
//
////////////////////////////////////////////////////////////////////////////

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
//#include <GL/glew.h>
#include "platform.h"
#include "gl_state.h"
#include "textfile.h"
#include "tex_ogl.h"
#include "tex_ogl_1d.h"

//using namespace GL;
using namespace CG;

static bool use_arb = true;
GLuint GL::shader_program = 1000;
int GL::nb_instances = 0;
//----------------------------------------------------------------------
GL::GL(int maxPrograms)
{
	if (nb_instances == 1) {
		printf("GL is a singleton: only one instance allowed!\n");
		exit(0); // need better method to exit
	}

	this->maxPrograms = maxPrograms;
	shader = new Program* [maxPrograms];

	for (int i=0; i < maxPrograms; i++) {
		shader[i] = 0;
	}

	nb_instances++;
}
//----------------------------------------------------------------------
GLuint GL::setupShaderProgram()
{	
	printf("0\n");
	Program* sh= new Program();
	printf("1\n");
	int id = sh->getId();
	printf("2\n");
	shader[id] = sh;
	printf("3\n");
	return id;
}
//----------------------------------------------------------------------
GLuint GL::setupVertexShader(char* filename)
{
// assumes single vertex shader in shader program

	//printf("----> vertex shader in file: %s\n", filename);
	Program* sh = new Program();
	sh->load_from_file(filename, true);
	int id = sh->getId();
	if (id < 1) {
		printf("cannot setup Vertex Shader: id = %d\n", id);
		exit(0);
	}
	if (id >= maxPrograms) {
		printf("Exceeded max of %d Shaders\n", maxPrograms);
		exit(0);
	}
	shader[id] = sh;
	sh->link();
	return id;
}
//----------------------------------------------------------------------
GLuint GL::setupFragmentShader(char* filename)
{
	printf("----> fragment shader in file: %s\n", filename);
	Program* sh = new Program();
	sh->load_from_file(filename, false);
	int id = sh->getId();
	if (id < 1) {
		printf("cannot setup Fragment Shader: id = %d\n", id);
		exit(0);
	}
	if (id >= maxPrograms) {
		printf("Exceeded max of %d Shaders\n", maxPrograms);
		exit(0);
	}
	shader[id] = sh;
	sh->link(); // why remove this? If htere, there are problems with uniform on macs)
	return id;
}
//----------------------------------------------------------------------
void gl_error_callback()
{
}
//----------------------------------------------------------------------
void GL::init(bool use_ogl_arb)
{
	 printf("inside init\n");
	 shader_program = glCreateProgram();
}
//----------------------------------------------------------------------
void print_target_info(GLenum target)
{
     GLint val;
	 printf("1, target= %d\n", target);
     glGetProgramivARB(target, GL_MAX_PROGRAM_LOCAL_PARAMETERS_ARB, &val);
	 printf("2\n");
     printf("GL_MAX_PROGRAM_LOCAL_PARAMETERS_ARB         %d\n", val);
     glGetProgramivARB(target, GL_MAX_PROGRAM_ENV_PARAMETERS_ARB, &val);
     printf("GL_MAX_PROGRAM_ENV_PARAMETERS_ARB           %d\n", val);
     printf("\n");
     glGetProgramivARB(target, GL_MAX_PROGRAM_INSTRUCTIONS_ARB, &val);
     printf("GL_MAX_PROGRAM_INSTRUCTIONS_ARB             %d\n", val);
     glGetProgramivARB(target, GL_MAX_PROGRAM_TEMPORARIES_ARB, &val);
     printf("GL_MAX_PROGRAM_TEMPORARIES_ARB              %d\n", val);
     glGetProgramivARB(target, GL_MAX_PROGRAM_PARAMETERS_ARB, &val);
     printf("GL_MAX_PROGRAM_PARAMETERS_ARB               %d\n", val);
     glGetProgramivARB(target, GL_MAX_PROGRAM_ATTRIBS_ARB, &val);
     printf("GL_MAX_PROGRAM_ATTRIBS_ARB                  %d\n", val);
     glGetProgramivARB(target, GL_MAX_PROGRAM_ALU_INSTRUCTIONS_ARB, &val);
     printf("GL_MAX_PROGRAM_ALU_INSTRUCTIONS_ARB         %d\n", val);
     glGetProgramivARB(target, GL_MAX_PROGRAM_TEX_INSTRUCTIONS_ARB, &val);
     printf("GL_MAX_PROGRAM_TEX_INSTRUCTIONS_ARB         %d\n", val);
     glGetProgramivARB(target, GL_MAX_PROGRAM_TEX_INDIRECTIONS_ARB, &val);
     printf("GL_MAX_PROGRAM_TEX_INDIRECTIONS_ARB         %d\n", val);
     printf("\n");
     glGetProgramivARB(target, GL_MAX_PROGRAM_NATIVE_INSTRUCTIONS_ARB, &val);
     printf("GL_MAX_PROGRAM_NATIVE_INSTRUCTIONS_ARB      %d\n", val);
     glGetProgramivARB(target, GL_MAX_PROGRAM_NATIVE_TEMPORARIES_ARB, &val);
     printf("GL_MAX_PROGRAM_NATIVE_TEMPORARIES_ARB       %d\n", val);
     glGetProgramivARB(target, GL_MAX_PROGRAM_NATIVE_PARAMETERS_ARB, &val);
     printf("GL_MAX_PROGRAM_NATIVE_PARAMETERS_ARB        %d\n", val);
     glGetProgramivARB(target, GL_MAX_PROGRAM_NATIVE_ATTRIBS_ARB, &val);
     printf("GL_MAX_PROGRAM_NATIVE_ATTRIBS_ARB           %d\n", val);
     glGetProgramivARB(target, GL_MAX_PROGRAM_NATIVE_ALU_INSTRUCTIONS_ARB, &val);
     printf("GL_MAX_PROGRAM_NATIVE_ALU_INSTRUCTIONS_ARB  %d\n", val);
     glGetProgramivARB(target, GL_MAX_PROGRAM_NATIVE_TEX_INSTRUCTIONS_ARB, &val);
     printf("GL_MAX_PROGRAM_NATIVE_TEX_INSTRUCTIONS_ARB  %d\n", val);
     glGetProgramivARB(target, GL_MAX_PROGRAM_NATIVE_TEX_INDIRECTIONS_ARB, &val);
     printf("GL_MAX_PROGRAM_NATIVE_TEX_INDIRECTIONS_ARB  %d\n", val);
}
//----------------------------------------------------------------------
void GL::print_gl_info()
{
     if (use_arb) {
	  printf("\n     cg info for GL_VERTEX_PROGRAM_ARB\n");
	  print_target_info(GL_VERTEX_PROGRAM_ARB);
	  printf("-----------------------------------------------\n\n");
	  printf("\n     cg info for GL_FRAGMENT_PROGRAM_ARB\n");
	  printf("-----------------------------------------------\n");
	  print_target_info(GL_FRAGMENT_PROGRAM_ARB);
	  printf("-----------------------------------------------\n\n");
     }
     else {
     }
}
//----------------------------------------------------------------------
CG::Program& GL::setupShaderProgram(char* name, GLuint *shader_id, int which)
{
	*shader_id = setupShaderProgram();
	CG::Program* pg = getShader(*shader_id);

	char frag_name[32];
	char vert_name[32];

	sprintf(frag_name, "%s.frag", name);
	sprintf(vert_name, "%s.vert", name);

	switch (which) {
	case BOTH_SHADERS:
		pg->addFragmentShader(frag_name);
		pg->addVertexShader(vert_name);
		printf("vert_name= %s\n", vert_name);
		break;
	case VERT_SHADER:
		pg->addVertexShader(vert_name);
		break;
	case FRAG_SHADER:
		pg->addFragmentShader(frag_name);
		break;
	}

	pg->link();
}
//----------------------------------------------------------------------
Program::Program()
{
	printf("program 0\n");  // ERROR ON NEXT LINE?
	shader_program = glCreateProgram();
	printf("program 1, shader_program= %d\n", shader_program);
}
//----------------------------------------------------------------------
void Program::load_from_string(const char* str, bool vertex_prg,
			       const char* entry)
{
}
//----------------------------------------------------------------------
void Program::addVertexShader(char* filename)
{
	GLuint s = glCreateShader(GL_VERTEX_SHADER);
	char* st = textFileRead(filename);

	const char* stp = st;
    glShaderSource(s,  1, &stp,  NULL);
    glCompileShader(s);

    GLint status;
    glGetShaderiv(s, GL_COMPILE_STATUS, &status);
    const char* stat = status == GL_TRUE ? "true" : "false";
    printf("vertex shader (%s) compiled correctly? %s\n", filename, stat);
    checkError("1b");

    if (1) {
        GLchar log[500];
        log[499] = '\0';
        GLsizei lg;
    checkError("1b.1");
        glGetShaderInfoLog(shader_program, 499, &lg, log);
		printf("lg= %d\n", lg);
    checkError("1b.2");
        if (strlen(log) != 0) printf("compile log= %s\n", log);
    }

	if (stat == "false") {
		exit(0);
	}

	checkError("1c");
    glAttachShader(shader_program, s);
	checkError("1d");
}
//----------------------------------------------------------------------
void Program::addFragmentShader(char* filename)
{
	GLuint s = glCreateShader(GL_FRAGMENT_SHADER);
	char* st = textFileRead(filename);

	const char* stp = st;
    glShaderSource(s,  1, &stp,  NULL);
    glCompileShader(s);

	GLint status;
 	glGetShaderiv(s, GL_COMPILE_STATUS, &status);
 	const char* stat = status == GL_TRUE ? "true" : "false";
 	printf("frag shader (%s) compiled correctly? %s\n", filename, stat);

    if (1) {
        char log[500];
        log[499] = '\0';
        GLsizei lg;
        glGetShaderInfoLog(s, 499, &lg, log);
		printf("fragment shader, lg= %d\n");
        if (strlen(log) != 0) printf("compile log= %s\n", log);
    }

	if (stat == "false") {
		exit(0);
	}
    glAttachShader(shader_program, s);
}
//----------------------------------------------------------------------
bool Program::load_from_file  (const char* filename, bool vertex_prg,
			       const char* entry)
{
	GLuint s, f, v;

	 //printf("load_from_file: shader_program = %d ..\n", shader_program);
	 if (vertex_prg) {
     	v = s = glCreateShader(GL_VERTEX_SHADER);
	 } else {
		printf("create Fragment Shader\n");
     	f = s = glCreateShader(GL_FRAGMENT_SHADER);
	 }

	char* st = textFileRead(filename);
	const char* stp = st;
	//printf("st= %s\n", st);
	//printf("shader source: s=%d, text= \n%s\n", s, st);

    glShaderSource(s,  1, &stp,  NULL);
	glCompileShader(s);
	if (1) {
		char log[500];
		log[499] = '\0';
		GLsizei lg;
		glGetShaderInfoLog(s, 500, &lg, log);
		if (strlen(log) != 0) printf("compile log= %s\n", log);
	}
	glAttachShader(shader_program, s); // why is this required? 
	//glLinkProgram(shader_program);

	//printf("created shader: %d from file %s\n", shader_program, filename);
	return true;
}
//----------------------------------------------------------------------
void Program::set_binding_buffer(const char* name, GLuint buffer)
{
#ifdef SHADER4
	int loc = glGetUniformLocation(shader_program, name);
	glUniformBufferEXT(shader_program, loc, buffer);
#endif
}
//----------------------------------------------------------------------
void Program::set_param1(const char* name, float x)
{
	 // loc = -1 if variable is not active (i.e., not used in shader)
	 checkError("param1: a");
	 int loc = glGetUniformLocation(shader_program, name);
	 checkError("param1: b");
	 glUniform1f(loc, x);
	 checkError("param1: c");
}
//----------------------------------------------------------------------
void Program::set_param1(const char* name, int x)
{
	 int loc = glGetUniformLocation(shader_program, name);
	 glUniform1i(loc, x);
}
//----------------------------------------------------------------------
float Program::get_param1(const char* name)
{
	int loc = glGetUniformLocation(shader_program, name);
	GLfloat v[10];
	//printf("get_param1: var: %s, loc= %d\n", name, loc);
	glGetUniformfv(shader_program, loc, v);
	return v[0];
}
//----------------------------------------------------------------------
void Program::set_param1(const char* name, const float* v)
{
}
//----------------------------------------------------------------------
void Program::set_param2(const char* name, float x, float y)
{
	 int loc = glGetUniformLocation(shader_program, name);
	 glUniform2f(loc, x, y);
}
//----------------------------------------------------------------------
void Program::set_param2(const char* name, const float* v)
{
}
//----------------------------------------------------------------------
void Program::set_param3(const char* name, float x, float y, float z)
{
}
//----------------------------------------------------------------------
void Program::set_param3(const char* name, const float* v)
{
}
//----------------------------------------------------------------------
void Program::set_param4(const char* name, float x, float y, float z, float w)
{
	 int loc = glGetUniformLocation(shader_program, name);
	 glUniform4f(loc, x,y,z,w);
}
//----------------------------------------------------------------------
void Program::set_param4(const char* name, const float* v)
{
}
//----------------------------------------------------------------------
bool Program::set_tex(const char* name, TexOGL1D& texture, int texUnit)
{
	int sampler_uniform_location = glGetUniformLocation(shader_program, name); // ERROR
	// loc = -1 if variable is not active (i.e., not used in shader)
	//printf("set_tex %s, unit: %d: loc= %d\n", name, texUnit, sampler_uniform_location);

	// IMPORTANT: put glActiveTexture before texture.bind()
	// OTHERWISE DOES NOT WORK (BUG?). Correct. 
 	glActiveTexture(GL_TEXTURE0 + texUnit);
	glEnable(texture.getTarget());
	texture.bind();
	glUniform1i(sampler_uniform_location, texUnit);
	return true;
}
//----------------------------------------------------------------------
bool Program::set_tex(const char* name, TexOGL& texture, int texUnit)
{
	int sampler_uniform_location = glGetUniformLocation(shader_program, name); // ERROR
	// loc = -1 if variable is not active (i.e., not used in shader)
	//printf("set_tex %s, unit: %d: loc= %d\n", name, texUnit, sampler_uniform_location);

	// IMPORTANT: put glActiveTexture before texture.bind()
	// OTHERWISE DOES NOT WORK (BUG?). Correct. 
 	glActiveTexture(GL_TEXTURE0 + texUnit);
	glEnable(texture.getTarget());
	texture.bind();
	glUniform1i(sampler_uniform_location, texUnit);
	return true;
}
//----------------------------------------------------------------------
bool Program::set_tex(const char* name, GLuint tex_obj, int texUnit)
{
	int my_sampler_uniform_location = glGetUniformLocation(shader_program, name);
	//printf("set_tex 2: loc= %d\n", my_sampler_uniform_location);
 	glActiveTexture(GL_TEXTURE0 + texUnit);
	glEnable(TARGET);
	exit(0);
	glBindTexture(TARGET, tex_obj);
	glUniform1i(my_sampler_uniform_location, texUnit);
	return true;
}
//----------------------------------------------------------------------
bool Program::set_tex_2D(const char* name, GLuint tex_obj, int texUnit)
{
	int my_sampler_uniform_location = glGetUniformLocation(shader_program, name);
	//printf("set_tex 2: loc= %d\n", my_sampler_uniform_location);
 	glActiveTexture(GL_TEXTURE0 + texUnit);
	glEnable(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D, tex_obj);
	glUniform1i(my_sampler_uniform_location, texUnit);
	return true;
}
//----------------------------------------------------------------------
bool Program::set_tex1d(const char* name, unsigned int tex_obj, int texUnit)
{
	int my_sampler_uniform_location = glGetUniformLocation(shader_program, name);
	//printf("set_tex 3: loc= %d\n", my_sampler_uniform_location);
 	glActiveTexture(GL_TEXTURE0 + texUnit);
	glEnable(GL_TEXTURE_1D);
	glBindTexture(GL_TEXTURE_1D, tex_obj);
	glUniform1i(my_sampler_uniform_location, texUnit);
	return true;
}
//----------------------------------------------------------------------
void Program::valid()
{
}
//----------------------------------------------------------------------
void Program::link()
{
	int err = 0;
	GLint progLinkSuccess = 0;

	//printf("link programs\n");
	glLinkProgram(shader_program); // returns void

	GLint status;
 	glGetProgramiv(shader_program, GL_LINK_STATUS, &status);
 	const char* stat = status == GL_TRUE ? "true" : "false";
 	//printf("link status: %s\n", stat);

	//glGetObjectParameterivARB(shader_program, 
		//GL_OBJECT_LINK_STATUS_ARB, &progLinkSuccess);
	//if (!progLinkSuccess) {
		//err = 1;
		//fprintf(stderr, "Shader could not be linked\n");
	//}

	if (1) {
		char log[500];
		log[499] = '\0';
		GLsizei lg;
		glGetProgramInfoLog(shader_program, 500, &lg, log);
		if (strlen(log) != 0) printf("link log= %s\n", log);
	}

	glUseProgram(shader_program);  // ERROR
	if (glGetError() != 0) {
		char log[500];
		log[499] = '\0';
		GLsizei lg;
		glGetProgramInfoLog(shader_program, 500, &lg, log);
		if (strlen(log) != 0) printf("use_program log= %s\n", log);
	}

	if (err == 1) {
		exit(1);
	}
	//printf("Link stage successful!\n");
}
//----------------------------------------------------------------------
void Program::checkError(char* msg)
{
#if 0
	printf("glerror: %s,  %s\n", msg, gluErrorString(glGetError())); // error
#endif
}
//----------------------------------------------------------------------

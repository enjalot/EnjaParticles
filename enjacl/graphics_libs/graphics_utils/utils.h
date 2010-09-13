#ifndef __UTILS_H__
#define __UTILS_H__

#include "glincludes.h"

#include "platform.h"
#include "tex_ogl.h"
#include "gl_state.h"
//#include "Framework.h"

//#include "include.h"
#define WANT_STREAM
#include "newran.h" // from random/

#include <fstream>
#include <istream>
#include <iostream>

#include <vector>
//using namespace std;

class InternalFormat;
class Format;
class DataType;
class Target;
class TexMeta;
class TextureOGL;

typedef std::vector<InternalFormat> IFORMATS;
typedef std::vector<Format> FORMATS;
typedef std::vector<DataType> DATA_TYPES;
typedef std::vector<Target> TARGETS;
typedef std::vector<TexMeta*> TEXMETAS;


class InternalFormat {
public:
	const char* name;
	GLint format;
	InternalFormat(const char* name, GLint format) {
		this->name = name;
		this->format = format;
	}
};

class Format {
public:
	const char* name;
	GLenum format;
	Format(const char* name, GLenum format) {
		this->name = name;
		this->format = format;
	}
};


class DataType {
public:
	const char* name;
	GLenum type;
	DataType(const char* name, GLenum type) {
		this->name = name;
		this->type = type;
	}
};

class Target {
public:
	const char* name;
	GLenum target;
	Target(const char* name, GLenum target) {
		this->name = name;
		this->target = target;
	}
};

class TexMeta {
public:
	Target* tg;
	InternalFormat* ifmt;
	Format* fmt;
	DataType* dt;
	TexMeta(Target* tg, InternalFormat* ifmt, Format* fmt, DataType* dt) {
		this->tg = tg;
		this->ifmt = ifmt;
		this->fmt = fmt;
		this->dt = dt;
	}
};
	
#if 0
     int          sz;
	 int		  height;
     GLenum       target;
	 GLint        internal_format;
	 GLenum       format;
	 GLenum       data_type;
#endif

class Utils {
public: 
// not really a good idea, but this is similar to being global
// these variables should really be static
	//Framework& fmwk;
	IFORMATS internalFormats;
	FORMATS  formats;
	DATA_TYPES dataTypes;
	TARGETS targets;
	TEXMETAS texmetas;
	Uniform uniform; // from random/

public:
	//utils(Framework& fmk);
	Utils();
	~Utils() {;}
	//void draw_quad(float width, float height);

	GLuint gen_quad();
	GLuint gen_quad(TextureOGL& tex0);
	GLuint gen_quad_multi(TextureOGL& tex0);
	GLuint gen_quad_multi(TextureOGL& tex0, TextureOGL& tex1);
	GLuint gen_quad_multi(TextureOGL& tex0, TextureOGL& tex1, TextureOGL& tex2);
	GLuint gen_quad_multi(TextureOGL& tex0, TextureOGL& tex1, TextureOGL& tex2, TextureOGL& tex3);

	void draw_quad();
	void draw_quad(TextureOGL& tex1);
	void draw_quad_multi(TextureOGL& tex1);
	void draw_quad_multi(TextureOGL& tex1, TextureOGL& tex2);
	void draw_quad_multi(TextureOGL& tex1, TextureOGL& tex2, TextureOGL& tex3);
	void draw_quad_multi(TextureOGL& tex1, TextureOGL& tex2, TextureOGL& tex3, TextureOGL& tex4);
	void draw_quad_multi(float w0=1., float h0=1., float w1=1., float h1=1., 
   		float w2=1., float h2=1., float w3=1., float h3=1.);
	//void draw_quad_multi(float width, float height);
	void draw_quad(float w);
	void draw_quad_multi();

	// not usable at this point
	void setTexFormat(TextureOGL& tex, InternalFormat& ifmt, Format& fmt, DataType& type);
	void setTexTarget(TextureOGL& tex, Target& target);

	void setTexFormat(TextureOGL&, GLint iformat, GLenum format, GLenum dataType);
	void setTexTarget(TextureOGL&, GLenum target);

	float* readPixels();
	void readPixels(GLenum buffer);
	void readPixels(float* pixels, int nx, int ny);
	void readPixels(GLenum buffer, float* pixels, int nx, int ny);
	void printBuffer(GLenum buffer, int xo, int yo, int w, int h);
	void reshape(int size);

	void disableTextures(int i); // Disable texture units 0 -> i
	void enableTextures(int i); // Enable texture units 0 -> i

	//void printFBO(int curBuf, int lower_left, int upper_right, int w, int h, const char* msg);
	//void printFBO2(int curBuf, int lower_left, int upper_right, int w, int h, const char* msg);
	void checkError(char* msg);

	float rand_float();
	int rand_int(int a, int b);
	double rand_float(double a, double b);

	void initializeData();
	double getSeed();
	int nano_sleep(double sleep_time);
	unsigned int nextPow2( unsigned int x );
};

// helper class for portability
//class utils : public Utils
//{
//}

#endif

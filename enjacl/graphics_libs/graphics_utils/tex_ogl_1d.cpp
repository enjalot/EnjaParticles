//////////////////////////////////////////////////////////////////////////
//
// tex_ogl.cpp
//
// 2003 Patrick Crawley
//
//////////////////////////////////////////////////////////////////////////

#include <stdio.h>
#include <stdlib.h>
#include "platform.h"
#include "tex_ogl_1d.h"

//////////////////////////////////////////////////////////////////////////
//

TextureOGL1D::~TextureOGL1D()
{
	//printf("inside destructure\n");
}
//----------------------------------------------------------------------
void TextureOGL1D::init_targ(int width, GLenum targ)
{
// HARDCODED for rect. IN FACT, SHOULD DEPEND ON ARGUMENT
     sz     = width;
     target = targ; 
     glGenTextures(1, (GLuint*) &obj);
}
//----------------------------------------------------------------------
void TextureOGL1D::init(int width, bool rect_tex)
{
// HARDCODED for rect. IN FACT, SHOULD DEPEND ON ARGUMENT
	 return;
     target = (rect_tex) ? GL_TEXTURE_RECTANGLE_NV : GL_TEXTURE_2D;
     sz     = width;
     glGenTextures(1, (GLuint*) &obj);
	 printf("inside TextureOGL1D::init, shoudl not happen\n");
	 exit(0);
}
//----------------------------------------------------------------------
void TextureOGL1D::init_explicit_targ(unsigned int tex_obj, int width, GLenum targ)
{
     sz     = width;
     obj    = tex_obj;
     target = targ;
}
//----------------------------------------------------------------------
void TextureOGL1D::init_explicit(unsigned int tex_obj, int width, bool rect_tex)
{
     sz     = width;
     obj    = tex_obj;
     target = (rect_tex) ? GL_TEXTURE_RECTANGLE_NV : GL_TEXTURE_2D;
	 exit(0);
}
//----------------------------------------------------------------------
void TextureOGL1D::load_image(GLint internal_format, GLenum format, 
			    GLenum data_type, void* data)
{
	static int count;

	this->data_type = data_type;
	this->format = format;
	this->internal_format = internal_format;

     bind();
	 //GE glTexImage2D takes up most of the time
     glTexImage1D(target, 0, internal_format, sz, 
		  0, format, data_type, data);
     clamp_to_edge();
     linear();
	 //printf("nb calls to load_image= %d\n", count++);
}

//----------------------------------------------------------------------
void TextureOGL1D::kill()
{
     glDeleteTextures(1, (GLuint*) &obj);
}
//----------------------------------------------------------------------
bool TextureOGL1D::is_rect()
{
     return (target == GL_TEXTURE_RECTANGLE_NV) ? true : false;
}
//----------------------------------------------------------------------
void TextureOGL1D::clamp()
{
     glTexParameteri(target, GL_TEXTURE_WRAP_S, GL_CLAMP);
}
//----------------------------------------------------------------------
void TextureOGL1D::clamp_to_edge()
{
     glTexParameteri(target, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
}
//----------------------------------------------------------------------
void TextureOGL1D::repeat()
{
     if (target == GL_TEXTURE_RECTANGLE_NV) {
	  clamp_to_edge();
     }
     else {
	  glTexParameteri(target, GL_TEXTURE_WRAP_S, GL_REPEAT);
     }
}
//----------------------------------------------------------------------
void TextureOGL1D::point_sampling()
{
     glTexParameteri(target, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
     glTexParameteri(target, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
}
//----------------------------------------------------------------------
void TextureOGL1D::linear_sampling()
{
     glTexParameteri(target, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
     glTexParameteri(target, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
}
//----------------------------------------------------------------------

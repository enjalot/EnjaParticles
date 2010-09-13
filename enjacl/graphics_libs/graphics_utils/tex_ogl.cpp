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
#include "tex_ogl.h"

//////////////////////////////////////////////////////////////////////////
//

TextureOGL::~TextureOGL()
{
	printf("inside ~TextureOGL\n");
	printf("tex_ogl destructor: obj= %d\n", obj);
	//glDeleteTextures(1, &obj);
	//printf("inside destructure\n");
}
//----------------------------------------------------------------------
void TextureOGL::init_targ(int width, int rheight, GLenum targ)
{
// HARDCODED for rect. IN FACT, SHOULD DEPEND ON ARGUMENT
     sz     = width;
     height = rheight;
     target = targ; 
     glGenTextures(1, (GLuint*) &obj);
}
//----------------------------------------------------------------------
void TextureOGL::init(int width, int rheight, bool rect_tex)
{
// HARDCODED for rect. IN FACT, SHOULD DEPEND ON ARGUMENT
	 return;
     target = (rect_tex) ? GL_TEXTURE_RECTANGLE_NV : GL_TEXTURE_2D;
     sz     = width;
     height = rheight;
     glGenTextures(1, (GLuint*) &obj);
	 printf("inside TextureOGL::init, should not happen\n");
	 exit(0);
}
//----------------------------------------------------------------------
void TextureOGL::init_explicit_targ(unsigned int tex_obj, int width, int rheight, GLenum targ)
{
     sz     = width;
     obj    = tex_obj;
     height = rheight;
     target = targ;
}
//----------------------------------------------------------------------
void TextureOGL::init_explicit(unsigned int tex_obj, int width, int rheight, bool rect_tex)
{
     sz     = width;
     obj    = tex_obj;
     height = rheight;
     target = (rect_tex) ? GL_TEXTURE_RECTANGLE_NV : GL_TEXTURE_2D;
	 exit(0);
}
//----------------------------------------------------------------------
void TextureOGL::load_image(GLint internal_format, GLenum format, 
			    GLenum data_type, void* data)
{
	static int count;

	this->data_type = data_type;
	this->format = format;
	this->internal_format = internal_format;
	setInternalData();

     bind();
	 //GE glTexImage2D takes up most of the time
     //glTexImage2D(target, 0, internal_format, sz, height,
     glTexImage2D(target, 0, internal_format, sz, height, //GE
		  0, format, data_type, data);
     clamp_to_edge();
     linear();
	 //printf("nb calls to load_image= %d\n", count++);
	 printf("tex_ogl: load: glTExImage2D, w,h= %d, %d\n", sz, height);
}

//----------------------------------------------------------------------
void TextureOGL::kill()
{
     glDeleteTextures(1, (GLuint*) &obj);
}
//----------------------------------------------------------------------
bool TextureOGL::is_rect()
{
     return (target == GL_TEXTURE_RECTANGLE_NV) ? true : false;
}
//----------------------------------------------------------------------
void TextureOGL::read_framebuffer()
{
     bind();
     glCopyTexSubImage2D(target, 0, 0, 0, 0, 0, sz, height);
}
//----------------------------------------------------------------------
void TextureOGL::clamp()
{
     glTexParameteri(target, GL_TEXTURE_WRAP_S, GL_CLAMP);
     glTexParameteri(target, GL_TEXTURE_WRAP_T, GL_CLAMP);
}
//----------------------------------------------------------------------
void TextureOGL::clamp_to_edge()
{
     glTexParameteri(target, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
     glTexParameteri(target, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
}
//----------------------------------------------------------------------
void TextureOGL::repeat()
{
     if (target == GL_TEXTURE_RECTANGLE_NV) {
	  clamp_to_edge();
     }
     else {
	  glTexParameteri(target, GL_TEXTURE_WRAP_S, GL_REPEAT);
	  glTexParameteri(target, GL_TEXTURE_WRAP_T, GL_REPEAT);
     }
}
//----------------------------------------------------------------------
void TextureOGL::point_sampling()
{
     glTexParameteri(target, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
     glTexParameteri(target, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
}
//----------------------------------------------------------------------
void TextureOGL::linear_sampling()
{
     glTexParameteri(target, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
     glTexParameteri(target, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
}
//----------------------------------------------------------------------
void DepthTextureOGL::init(int size)
{
     TextureOGL::init(size,size, 0);
     TextureOGL::bind();

     float* depth = new float[sz*sz*1];
     glTexImage2D(target, 0, GL_DEPTH_COMPONENT,
                  sz,sz, 0, GL_DEPTH_COMPONENT,
                  GL_FLOAT, (void*)depth);
     glTexParameteri(target, GL_DEPTH_TEXTURE_MODE, GL_LUMINANCE);
     clamp_to_edge();
     point();

     delete [] depth;
}
//----------------------------------------------------------------------
void DepthTextureOGL::load_depth()
{
     bind();
     glCopyTexSubImage2D(target, 0, 0, 0, 0, 0, sz, sz);
}
//----------------------------------------------------------------------
void TextureOGL::setInternalData()
{
	switch (internal_format) {
	//case FLOAT_BUFFER:  // same value as FLOAT32_4
	case FLOAT32_4:
		nb_internal_channels = 4;
		nb_bytes_per_channel = 4;
		break;
	case FLOAT32_3:
		nb_internal_channels = 3;
		nb_bytes_per_channel = 4;
		break;
	case FLOAT32_2:
		nb_internal_channels = 2;
		nb_bytes_per_channel = 4;
		break;
	case FLOAT32_1:
		nb_internal_channels = 1;
		nb_bytes_per_channel = 4;
		break;
	case FLOAT16_4:
		nb_internal_channels = 4;
		nb_bytes_per_channel = 2;
		break;
	case FLOAT16_3:
		nb_internal_channels = 3;
		nb_bytes_per_channel = 2;
		break;
	case FLOAT16_2:
		nb_internal_channels = 2;
		nb_bytes_per_channel = 2;
		break;
	case FLOAT16_1:
		nb_internal_channels = 1;
		nb_bytes_per_channel = 2;
		break;
	}
}
//----------------------------------------------------------------------
void TextureOGL::setBorder(int b)
{
	border = b; 
	inner_width  = getWidth() - border;
	inner_height = getHeight() - border;
}
//----------------------------------------------------------------------

//////////////////////////////////////////////////////////////////////////
//
// tex_ogl.h
//
// 2003 Patrick Crawley
//
//////////////////////////////////////////////////////////////////////////

#ifndef _TEX_OGL_H_
#define _TEX_OGL_H_

#include "platform.h"
#include "glincludes.h"
//#include <GL/gl.h>
//#include <GL/glext.h>

//----------------------------------------------------------------------
class TextureOGL
{
protected:
     unsigned int obj;
     int          sz;
	 int		  height;
     GLenum       target;
	 GLint        internal_format;
	 GLenum       format;
	 GLenum       data_type;
	 int		  nb_internal_channels; 
	 int 		  nb_bytes_per_channel;

	 // keep track of border for rectangular textures (have no OpenGL border option)
	 // for 2D textures, useful way to keep track of borders
	 int		  border; // not used
	 int		  inner_width; // without border
	 int		  inner_height; // without border

	 // ideally, each texture should have an associated texture unit. 
	 // The problem is that two texture units can  correspond to the same
	 // texture (is this true?)

public:
     TextureOGL() : obj(0), sz(0), height(0), border(0), target(TARGET) {}
     //TextureOGL() : obj(0), sz(0), height(0), border(0), target(GL_TEXTURE_2D) {}
     ~TextureOGL();

	 // TextureOGL(const TextureOGL&);  // copy constructor: should be implemented

	 void init_targ(int width, int rheight, GLenum target);
	 void init(int width, int rheight, bool rect_tex);
	 void init_explicit_targ(unsigned int tex_obj, int width, int rheight, GLenum target);
	 void init_explicit(unsigned int tex_obj, int width, int rheight, bool rect_tex);
     void load_image(GLint internal_format, GLenum format, 
		     GLenum data_type, void* data);
     void load(GLint internal_format, GLenum format, 
               GLenum data_type, void* data)
          { load_image(internal_format, format, data_type, data); }
	 // load data using pre-existing formats
	 void load(void* data) 
	 	  { load(internal_format, format, data_type, data); }
     void kill();

     void read_framebuffer();

     void clamp();
     void clamp_to_edge();
     void repeat();
     void point()  { point_sampling(); }
     void linear() { linear_sampling(); }
     void point_sampling();
     void linear_sampling();
	 GLenum getTarget() {return target;}

     void bind()   { glBindTexture(target, obj); }

     bool is_rect();
     int  size()    { return sz; }
     int  getWidth()    { return sz; }
     int  getHeight()  { return height; }
     unsigned int tex_obj() { return obj; }

     void enable()  { glEnable (target); }
     void disable() { glDisable(target); }
	 void setTarget(GLenum target) {
		bind();
	 	this->target = target;
	 }
	 void setFormat(GLint iformat, GLenum format, GLenum dataType) {
	 	bind();
		this->internal_format = iformat;
		this->format = format;
		this->data_type = dataType;
	}
	GLint getIFormat() { return internal_format; }
	GLenum getFormat() { return format; }
	GLenum getDataType() { return data_type; }

	int getNbInternalChannels() { return nb_internal_channels; }
	int getNbBytesPerChannel() { return nb_bytes_per_channel; }
	int getNbBytesPerTexel() { return nb_bytes_per_channel*nb_internal_channels; }
	void setInternalData();

	void setBorder(int b);
	int getBorder() { return border; }
	int getInnerWidth() { return inner_width; }
	int getInnerHeight() { return inner_height; }
};
//----------------------------------------------------------------------

class DepthTextureOGL : public TextureOGL
{
public:
     void init(int size);
     void load_depth();
};
//----------------------------------------------------------------------

typedef TextureOGL      TexOGL;
typedef DepthTextureOGL DepthTexOGL;
//----------------------------------------------------------------------

#endif


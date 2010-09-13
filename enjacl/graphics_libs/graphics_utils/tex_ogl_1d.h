//////////////////////////////////////////////////////////////////////////
//
// tex_ogl_1d.h
// 
// Feb. 1, 2008,Gordon Erlebacher
// Based on original 2D version of Crawley first started in 2003
//
//////////////////////////////////////////////////////////////////////////

#ifndef _TEX_OGL1D_H_
#define _TEX_OGL1D_H_

#include "platform.h"
#include "glincludes.h"

//----------------------------------------------------------------------
class TextureOGL1D
{
protected:
     unsigned int obj;
     int          sz;
     GLenum       target;
	 GLint        internal_format;
	 GLenum       format;
	 GLenum       data_type;

	 // ideally, each texture should have an associated texture unit. 
	 // The problem is that two texture units can  correspond to the same
	 // texture (is this true?)

public:
     TextureOGL1D() : obj(0), sz(0), target(TARGET) {}
     ~TextureOGL1D();

	 // TextureOGL1D(const TextureOGL1D&);  // copy constructor: should be implemented

	 void init_targ(int width, GLenum target);
	 void init(int width, bool rect_tex);
	 void init_explicit_targ(unsigned int tex_obj, int width, GLenum target);
	 void init_explicit(unsigned int tex_obj, int width, bool rect_tex);
     void load_image(GLint internal_format, GLenum format, 
		     GLenum data_type, void* data);
     void load(GLint internal_format, GLenum format, 
               GLenum data_type, void* data)
          { load_image(internal_format, format, data_type, data); }
	 // load data using pre-existing formats
	 void load(void* data) 
	 	  { load(internal_format, format, data_type, data); }
     void kill();

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
};
//----------------------------------------------------------------------

typedef TextureOGL1D      TexOGL1D;
//----------------------------------------------------------------------

#endif


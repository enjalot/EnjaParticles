#ifndef _ABSTRACT__TEXTURES_H_
#define _ABSTRACT__TEXTURES_H_

#include "utils.h"

// Create new textures 

class AbstractTextures
{
protected:
	 GLint	 	internal_format;
	 GLenum     format;
	 GLenum 	data_type;
	 GLenum		target;
	 int 		nx, ny;
	 Utils 		u;
	 int 		nb_internal_channels; // RGB(3), RGBA(4)
	 int 		nb_bytes_per_channel; // GL_FLOAT16 (2), GL_FLOAT32 (4)

public:
	AbstractTextures();
	AbstractTextures(int sz);
	AbstractTextures(int szx, int szy);
	virtual ~AbstractTextures();

	GLint getInternalFormat() { return internal_format; }
	GLenum getFormat(GLenum fmt) { return format; }
	GLenum getDataType(GLenum type) { return data_type; }
	void setInternalFormat(GLint i_fmt);
	void setFormat(GLenum fmt);
	void setDataType(GLenum type);
	void setFormat(GLint i_fmt, GLenum fmt, GLenum type);
	void setSize(int nx, int ny);
	void setBorder(int b);  // fixed border width in all directions
	void setTarget(GLenum target);

// Texture generation
// Allocate memory within the method (it is self contained)

	virtual TexOGL* createBWNoise() = 0;
	virtual TexOGL* createTwoColorNoise(float r1, float g1, float b1, float r2, float g2, float b2) = 0;
	virtual TexOGL* createOneColor(float r, float g, float b, float a=1.0) = 0;
	virtual TexOGL* createTwoColorHorizontal(float r1, float g1, float b1, float r2, float g2, float b2) = 0;
	virtual TexOGL* createTwoColorVertical(float r1, float g1, float b1, float r2, float g2, float b2) = 0;
	virtual TexOGL* createCheckerBoard(float r1, float g1, float b1, float r2, float g2, float b2, int mx=5, int my=5) = 0;
	//virtual TexOGL* createCircular() = 0;
};

#endif

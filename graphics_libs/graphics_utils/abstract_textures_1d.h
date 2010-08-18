#ifndef _ABSTRACT__TEXTURES_1D_H_
#define _ABSTRACT__TEXTURES_1D_H_

#include "utils.h"
#include "tex_ogl_1d.h"

// Create new textures 

class AbstractTextures1D
{
protected:
	 GLint	 	internal_format;
	 GLenum     format;
	 GLenum 	data_type;
	 GLenum		target;
	 int 		nx;
	 Utils 		u;

public:
	AbstractTextures1D();
	AbstractTextures1D(int sz);
	void setFormat(GLint i_fmt, GLenum fmt, GLenum type);
	void setSize(int nx);
	void setTarget(GLenum target);

// Texture generation
// Allocate memory within the method (it is self contained)

	virtual TexOGL1D* createBWNoise() = 0;
	virtual TexOGL1D* createTwoColorNoise(float r1, float g1, float b1, float r2, float g2, float b2) = 0;
	virtual TexOGL1D* createOneColor(float r, float g, float b) = 0;
	//virtual TexOGL1D* createTwoColorHorizontal(float r1, float g1, float b1, float r2, float g2, float b2) = 0;
};

#endif

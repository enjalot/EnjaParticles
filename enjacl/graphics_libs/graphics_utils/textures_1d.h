#ifndef _TEXTURES_1D_H_
#define _TEXTURES_1D_H_

#include "tex_ogl_1d.h"
#include "abstract_textures_1d.h"

// Create new textures 
// These textures have 4 components (whether 8 bit, 32 bit, or whatever)
// The data type is always GL_FLOAT (code arrays that store texture in program)

class Array3D;

class Textures1D : public AbstractTextures1D
{

public:
	Textures1D();
	/// initialize from existing texture (internal format, format, datatype, target)
	Textures1D(TexOGL1D& tex);
	Textures1D(int sz);

// Texture generation
// Allocate memory within the method (it is self contained)

	TexOGL1D* createBWNoise();
	TexOGL1D* createGrayNoise();
	TexOGL1D* createOneColor(float r, float g, float b);
	TexOGL1D* createTwoColor(float r1, float g1, float b1, float r2, float g2, float b2);
	TexOGL1D* createTwoColorNoise(float r1, float g1, float b1, float r2, float g2, float b2);
	TexOGL1D* createGaussian(float rms);
	TexOGL1D* createGaussianBW(float rms);
	TexOGL1D* createGrayNoiseRGBA();

	TexOGL1D* userDefined(Array3D& data); // only use index (i,0,0)
	TexOGL1D* createEmpty();
};

#endif

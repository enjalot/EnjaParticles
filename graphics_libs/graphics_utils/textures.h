#ifndef _TEXTURES_H_
#define _TEXTURES_H_

#include "tex_ogl.h"
#include "abstract_textures.h"

// Create new textures 
// These textures have 4 components (whether 8 bit, 32 bit, or whatever)
// The data type is always GL_FLOAT (code arrays that store texture in program)

class Array3D;

class Textures : public AbstractTextures
{

public:
	Textures();
	/// initialize from existing texture (internal format, format, datatype, target)
	Textures(TexOGL& tex);
	Textures(int sz);
	Textures(int szx, int szy);

// Texture generation
// Allocate memory within the method (it is self contained)

	TexOGL* createBWNoise();
	TexOGL* createGrayNoise();
	TexOGL* createTwoColorNoise(float r1, float g1, float b1, float r2, float g2, float b2);
	TexOGL* createOneColor(float r, float g, float b, float a=1.0);
	TexOGL* createTwoColorHorizontal(float r1, float g1, float b1, float r2, float g2, float b2);
	TexOGL* createTwoColorVertical(float r1, float g1, float b1, float r2, float g2, float b2);
	TexOGL* createCheckerBoard(float r1, float g1, float b1, float r2, float g2, float b2, int mx=5, int my=5);
	TexOGL* createFloatCheck(float base, float incr);
	TexOGL* createGaussian(float rms);
	TexOGL* createGaussianBW(float rms);
	TexOGL* createTestFloat();

	TexOGL* createGrayNoiseRGBA();

// Test velocity fields

	//TexOGL* createCircular() { return createCircular(1., 1.); }
	//TexOGL* createRadial() { return createRadial(1., 1.); }

	TexOGL* createCircular(float a=1., float b=1.);
	TexOGL* createRadial(float a=1., float b=1.);

	TexOGL* userDefined(Array3D& data);
	TexOGL* createEmpty();
};

#endif

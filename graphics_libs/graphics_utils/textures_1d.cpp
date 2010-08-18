
#include <stdio.h>
#include <stdlib.h>
#include "textures_1d.h"
#include "Array3D.h"
#include <math.h>
#include <sys/time.h>

//----------------------------------------------------------------------
Textures1D::Textures1D() : AbstractTextures1D()
{
	internal_format = FLOAT_BUFFER;
	format 		= GL_RGBA;
	data_type 	= GL_FLOAT;
	target 		= TARGET;
}
//----------------------------------------------------------------------
Textures1D::Textures1D(int sz) : AbstractTextures1D(sz)
{
	internal_format = FLOAT_BUFFER;
	format 		= GL_RGBA;
	data_type 	= GL_FLOAT;
	target 		= TARGET;
	setSize(sz);
}
//----------------------------------------------------------------------
Textures1D::Textures1D(TexOGL1D& tex) : AbstractTextures1D()
{
	internal_format = tex.getIFormat();
	format 		= tex.getFormat();
	data_type 	= tex.getDataType();
	target 		= tex.getTarget();
	setSize(tex.getWidth());
}
//----------------------------------------------------------------------
// Texture generation
// Allocate memory within the method (it is self contained)

TexOGL1D* Textures1D::createBWNoise()
{
// ONLY FOR GL_FLOAT datatype

	if (data_type != GL_FLOAT) {
		printf("createBWNoise: data_type should be GL_FLOAT\n");
		return 0;
	}

    float val, p;
	TexOGL1D* t = new TexOGL1D();

	TexOGL1D& input = *t;

	Array3D tex(4, nx, 1);

    	for (int i = 0; i < nx; i++) {
            p = u.rand_float();
            val = (p < 0.5) ? 0 : 1.;
            tex(0, i, 0) = val;
            tex(1, i, 0) = val;
            tex(2, i, 0) = val;
            tex(3, i, 0) = 1.0;
        }
    input.init_targ(nx, target); 
    input.load(internal_format, format, data_type, tex.getDataPtr()); 
	input.clamp();
	input.point();

	return t;
}
//----------------------------------------------------------------------
// Texture generation
// Allocate memory within the method (it is self contained)

TexOGL1D* Textures1D::createGrayNoiseRGBA()
{
// ONLY FOR GL_FLOAT datatype

    float val, p;
	TexOGL1D* t = new TexOGL1D();

	TexOGL1D& input = *t;

	Array3D tex(4, nx, 1);

    	for (int i = 0; i < nx; i++) {
            tex(0, i, 0) = u.rand_float();
            tex(1, i, 0) = u.rand_float();
            tex(2, i, 0) = u.rand_float();
            tex(3, i, 0) = 1.0;
			if (i == 0) {
				printf("... RG= %f, %f\n", tex(0,i,0), tex(1,i,0));
			}
        }
    input.init_targ(nx, target); 
    input.load(internal_format, format, data_type, tex.getDataPtr()); 
	input.clamp();
	input.point();

	return t;
}
//----------------------------------------------------------------------
// Texture generation
// Allocate memory within the method (it is self contained)

TexOGL1D* Textures1D::createGrayNoise()
{
// ONLY FOR GL_FLOAT datatype

	//struct timeval *tp;
	//struct timezone *tzp;
	//int timeday = gettimeofday(tp, tzp);

#ifndef LINUX
	sranddev();
#else 
	srand(100);
#endif

	if (internal_format != GL_RGBA) {
		printf("createGrayNoise: internal  format should be GL_RGBA\n");
		//exit(0);
	}
	if (data_type != GL_FLOAT) {
		printf("createBWNoise: data_type should be GL_FLOAT\n");
		return 0;
	}

    float val, p;
	TexOGL1D* t = new TexOGL1D();

	TexOGL1D& input = *t;
	Array3D tex(4, nx, 1);

    for (int i = 0; i < nx; i++) {
			float p = u.rand_float();
            tex(0, i, 0) = p;
            tex(1, i, 0) = p;
            tex(2, i, 0) = p;
            tex(3, i, 0) = 1.0;
    }
    input.init_targ(nx, target); 
    input.load(internal_format, format, data_type, tex.getDataPtr()); 
	input.repeat();
	input.point();

	return t;
}
//----------------------------------------------------------------------
TexOGL1D* Textures1D::createTwoColorNoise(float r1, float g1, float b1, float r2, float g2, float b2)
{
	if (data_type != GL_FLOAT) {
		printf("createBWNoise: data_type should be GL_FLOAT\n");
		return 0;
	}

    float val, p;
	TexOGL1D* t = new TexOGL1D();

	TexOGL1D& input = *t;

	Array3D tex(4, nx, 1);

    	for (int i = 0; i < nx; i++) {
            p = u.rand_float();
			if (p < 0.5) {
            	tex(0, i, 0) = r1;
            	tex(1, i, 0) = g1;
            	tex(2, i, 0) = b1;
			} else {
            	tex(0, i, 0) = r2;
            	tex(1, i, 0) = g2;
            	tex(2, i, 0) = b2;
			}
            tex(3, i, 0) = 1.0;
        }
    input.init_targ(nx, target); 
    input.load(internal_format, format, data_type, tex.getDataPtr()); 
	input.clamp();
	input.point();

	return t;
}
//----------------------------------------------------------------------
TexOGL1D* Textures1D::createOneColor(float r, float g, float b)
{
	if (data_type != GL_FLOAT) {
		printf("createBWNoise: data_type should be GL_FLOAT\n");
		return 0;
	}

    float val, p;
	TexOGL1D* t = new TexOGL1D();
	TexOGL1D& input = *t;

	Array3D tex(4, nx);

    	for (int i = 0; i < nx; i++) {
            tex(0, i, 0) = r;
            tex(1, i, 0) = g;
            tex(2, i, 0) = b;
            tex(3, i, 0) = 1.0;
        }
    input.init_targ(nx, target); 
    input.load(internal_format, format, data_type, tex.getDataPtr()); 
	input.clamp();
	input.point();

	return t;
}
//----------------------------------------------------------------------
TexOGL1D* Textures1D::createTwoColor(float r1, float g1, float b1, float r2, float g2, float b2)
{
	if (data_type != GL_FLOAT) {
		printf("createBWNoise: data_type should be GL_FLOAT\n");
		return 0;
	}

	TexOGL1D* t = new TexOGL1D();
	TexOGL1D& input = *t;

	Array3D tex(4, nx, 1);

    	for (int i = 0; i < nx; i++) {
			if (i < nx/2) {
            	tex(0, i, 0) = r1;
            	tex(1, i, 0) = g1;
            	tex(2, i, 0) = b1;
			} else {
            	tex(0, i, 0) = r2;
            	tex(1, i, 0) = g2;
            	tex(2, i, 0) = b2;
			}
            tex(3, i, 0) = 1.0;
        }

    input.init_targ(nx, target); 
    input.load(internal_format, format, data_type, tex.getDataPtr()); 
	input.clamp();
	input.point();

	return t;
}
//----------------------------------------------------------------------
TexOGL1D* Textures1D::createGaussian(float rms)
// grayscale gaussian
{
	if (data_type != GL_FLOAT) {
		printf("createGaussian: data_type should be GL_FLOAT\n");
		return 0;
	}

    float val, p;
	TexOGL1D* t = new TexOGL1D();
	TexOGL1D& input = *t;

	Array3D tex(4, nx);
	float dx = 2./(nx-1.);

    	for (int i = 0; i < nx; i++) {
			float x = -1.+dx*i;
			float f = exp(-(x*x)*rms);
            tex(0, i, 0) = 1.;// f;//f;
           	tex(1, i, 0) = 1.;// f;//f;
           	tex(2, i, 0) = 1.;// f;//f;
            tex(3, i, 0) = f;
        }

    input.init_targ(nx, target); 
    input.load(internal_format, format, data_type, tex.getDataPtr()); 
	input.clamp();
	input.point();

	return t;
}
//----------------------------------------------------------------------
TexOGL1D* Textures1D::createGaussianBW(float rms)
// BW gaussian
{
	if (data_type != GL_FLOAT) {
		printf("createGaussian: data_type should be GL_FLOAT\n");
		return 0;
	}

    float val, p;
	TexOGL1D* t = new TexOGL1D();
	TexOGL1D& input = *t;

	Array3D tex(4, nx);
	float dx = 2./(nx-1.);

	// WRONG: in this implementation, each pixel has different gray value. 

    	for (int i = 0; i < nx; i++) {
			float x = -1.+dx*i;
			float f = exp(-x*x*rms);
			float c = u.rand_float();
			//c = (c < 0.5) ? 0. : 1.;
            tex(0, i, 0) = c;// f;//f;
           	tex(1, i, 0) = c;// f;//f;
           	tex(2, i, 0) = c;// f;//f;
            tex(3, i, 0) = f;
            //tex(3, i, j) = 1.0;
        }

    input.init_targ(nx, target); 
    input.load(internal_format, format, data_type, tex.getDataPtr()); 
	input.clamp();
	input.point();

	return t;
}
//----------------------------------------------------------------------
TexOGL1D* Textures1D::userDefined(Array3D& data)
{
	if (data_type != GL_FLOAT) {
		printf("userDefined: data_type should be GL_FLOAT\n");
		exit(0);
		return 0;
	}

	TexOGL1D* t = new TexOGL1D();
	TexOGL1D& input = *t;

	int* dims = data.getDims();
	if (dims[0] != nx) {
		printf("Textures1D::userDefined, inconsistent dimensions: nx != data dims[0]\n");
		exit(0);
	}
	setSize(dims[1]); // dims[0] is RGBA

	input.init_targ(nx, target);
    input.load(internal_format, format, data_type, data.getDataPtr()); 
	//input.repeat();
	input.clamp();
	input.point();
	return t;
}
//----------------------------------------------------------------------
TexOGL1D* Textures1D::createEmpty()
{
	TexOGL1D* t = new TexOGL1D();
	TexOGL1D& input = *t;

    input.init_targ(nx, target); 
    input.load(internal_format, format, data_type, 0);
	input.clamp();
	input.point();
	return t;
}
//----------------------------------------------------------------------


#include <stdio.h>
#include <stdlib.h>
#include "textures.h"
#include "Array3D.h"
#include <math.h>
#include <sys/time.h>

//----------------------------------------------------------------------
Textures::Textures() : AbstractTextures()
{
	internal_format = FLOAT_BUFFER;
	format 		= GL_RGBA;
	data_type 	= GL_FLOAT;
	target 		= TARGET;
	nb_internal_channels = 4;
	nb_bytes_per_channel = 4; // float buffer
}
//----------------------------------------------------------------------
Textures::Textures(int sz) : AbstractTextures(sz)
{
	internal_format = FLOAT_BUFFER;
	format 		= GL_RGBA;
	data_type 	= GL_FLOAT;
	target 		= TARGET;
	nb_internal_channels = 4;
	nb_bytes_per_channel = 4; // float buffer
	setSize(sz, sz);
}
//----------------------------------------------------------------------
Textures::Textures(int szx, int szy) : AbstractTextures(szx, szy)
{
	internal_format = FLOAT_BUFFER;
	format 		= GL_RGBA;
	data_type 	= GL_FLOAT;
	target 		= TARGET;
	nb_internal_channels = 4;
	nb_bytes_per_channel = 4; // float buffer
	setSize(szx, szy);
}
//----------------------------------------------------------------------
Textures::Textures(TexOGL& tex) : AbstractTextures()
{
	internal_format = tex.getIFormat();
	format 		= tex.getFormat();
	data_type 	= tex.getDataType();
	target 		= tex.getTarget();
	nb_internal_channels = tex.getNbInternalChannels();
	nb_bytes_per_channel = tex.getNbBytesPerChannel(); // float buffer
	setSize(tex.getWidth(), tex.getHeight());
}
//----------------------------------------------------------------------
// Texture generation
// Allocate memory within the method (it is self contained)

TexOGL* Textures::createBWNoise()
{
// ONLY FOR GL_FLOAT datatype

	if (data_type != GL_FLOAT) {
		printf("createBWNoise: data_type should be GL_FLOAT\n");
		return 0;
	}

    float val, p;
	TexOGL* t = new TexOGL();

	TexOGL& input = *t;

	Array3D tex(nb_internal_channels, nx, ny);

		for (int j = 0; j < ny; j++) {
   		for (int i = 0; i < nx; i++) {
            p = u.rand_float();
            val = (p < 0.5) ? 0 : 1.;
			for (int k = 0;  k < nb_internal_channels; k++) {
            	tex(k, i, j) = val;
			}
			if (nb_internal_channels == 4) {
				tex(3, i, j) = 1.0;
			}
    	}}

    input.init_targ(nx, ny, target); 
    input.load(internal_format, format, data_type, tex.getDataPtr()); 
	input.clamp();
	input.point();

	return t;
}
//----------------------------------------------------------------------
// Texture generation
// Allocate memory within the method (it is self contained)

TexOGL* Textures::createGrayNoiseRGBA()
{
// ONLY FOR GL_FLOAT datatype

    float val, p;
	TexOGL* t = new TexOGL();

	TexOGL& input = *t;

	Array3D tex(nb_internal_channels, nx, ny);

	for (int j = 0; j < ny; j++) {
    for (int i = 0; i < nx; i++) {
		for (int k=0; k < nb_internal_channels; k++) {
            tex(k, i, j) = u.rand_float();
        }
		if (nb_internal_channels == 4) {
			tex(3, i, j) = 1.0;
		}
	}}

    input.init_targ(nx, ny, target); 
    input.load(internal_format, format, data_type, tex.getDataPtr()); 
	input.clamp();
	input.point();

	return t;
}
//----------------------------------------------------------------------
// Texture generation
// Allocate memory within the method (it is self contained)

TexOGL* Textures::createGrayNoise()
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
	TexOGL* t = new TexOGL();

	TexOGL& input = *t;
	Array3D tex(nb_internal_channels, nx, ny);

    for (int i = 0; i < nx; i++) {
	for (int j = 0; j < ny; j++) {
			float p = u.rand_float();
			for (int k=0; k < nb_internal_channels; k++) {
            	tex(k, i, j) = p;
			}
			if (nb_internal_channels == 4) {
				tex(3, i, j) = 1.0;
			}
    }}
    input.init_targ(nx, ny, target); 
    input.load(internal_format, format, data_type, tex.getDataPtr()); 
	input.repeat();
	input.point();

	return t;
}
//----------------------------------------------------------------------
TexOGL* Textures::createTwoColorNoise(float r1, float g1, float b1, float r2, float g2, float b2)
{
	if (data_type != GL_FLOAT) {
		printf("createBWNoise: data_type should be GL_FLOAT\n");
		return 0;
	}

	if (nb_internal_channels != 4) {
		printf("Textures::createTwoColorNoise, only works for 4-channel textures\n");
		exit(0);
	}

    float val, p;
	TexOGL* t = new TexOGL();

	TexOGL& input = *t;

	Array3D tex(4, nx, ny);

	for (int j = 0; j < ny; j++) {
    	for (int i = 0; i < nx; i++) {
            p = u.rand_float();
			if (p < 0.5) {
            	tex(0, i, j) = r1;
            	tex(1, i, j) = g1;
            	tex(2, i, j) = b1;
			} else {
            	tex(0, i, j) = r2;
            	tex(1, i, j) = g2;
            	tex(2, i, j) = b2;
			}
            tex(3, i, j) = 1.0;
        }
    }
    input.init_targ(nx, ny, target); 
    input.load(internal_format, format, data_type, tex.getDataPtr()); 
	input.clamp();
	input.point();

	return t;
}
//----------------------------------------------------------------------
TexOGL* Textures::createOneColor(float r, float g, float b, float a)
{
	if (data_type != GL_FLOAT) {
		printf("createBWNoise: data_type should be GL_FLOAT\n");
		return 0;
	}

	if (nb_internal_channels != 4) {
		printf("Textures::createOneColorNoise, only works for 4-channel textures\n");
		exit(0);
	}


    float val, p;
	TexOGL* t = new TexOGL();
	TexOGL& input = *t;

	Array3D tex(4, nx, ny);

	for (int j = 0; j < ny; j++) {
    	for (int i = 0; i < nx; i++) {
            tex(0, i, j) = r;
            tex(1, i, j) = g;
            tex(2, i, j) = b;
            tex(3, i, j) = a;
        }
    }
    input.init_targ(nx, ny, target); 
    input.load(internal_format, format, data_type, tex.getDataPtr()); 
	input.clamp();
	input.point();

	return t;
}
//----------------------------------------------------------------------
TexOGL* Textures::createTwoColorHorizontal(float r1, float g1, float b1, float r2, float g2, float b2)
// Two horizontal strips, bottom color 1, top: color 2. Texture evenly split
{
	if (data_type != GL_FLOAT) {
		printf("createBWNoise: data_type should be GL_FLOAT\n");
		return 0;
	}

	if (nb_internal_channels != 4) {
		printf("Textures::createTwoColorHorizontal, only works for 4-channel textures\n");
		exit(0);
	}

	TexOGL* t = new TexOGL();
	TexOGL& input = *t;

	Array3D tex(4, nx, ny);

	for (int j = 0; j < ny; j++) {
    	for (int i = 0; i < nx; i++) {
			if (j < ny/2) {
            	tex(0, i, j) = r1;
            	tex(1, i, j) = g1;
            	tex(2, i, j) = b1;
			} else {
            	tex(0, i, j) = r2;
            	tex(1, i, j) = g2;
            	tex(2, i, j) = b2;
			}
            tex(3, i, j) = 1.0;
        }
    }
    input.init_targ(nx, ny, target); 
    input.load(internal_format, format, data_type, tex.getDataPtr()); 
	input.clamp();
	input.point();

	return t;
}
//----------------------------------------------------------------------
TexOGL* Textures::createTwoColorVertical(float r1, float g1, float b1, float r2, float g2, float b2)
// Two vertical strips, left color 1, right: color 2. Texture evenly split
{
	if (data_type != GL_FLOAT) {
		printf("createBWNoise: data_type should be GL_FLOAT\n");
		return 0;
	}

	if (nb_internal_channels != 4) {
		printf("Textures::createTwoColorVertical, only works for 4-channel textures\n");
		exit(0);
	}

	TexOGL* t = new TexOGL();
	TexOGL& input = *t;

	Array3D tex(4, nx, ny);

	for (int j = 0; j < ny; j++) {
    	for (int i = 0; i < nx; i++) {
			if (i < nx/2) {
            	tex(0, i, j) = r1;
            	tex(1, i, j) = g1;
            	tex(2, i, j) = b1;
			} else {
            	tex(0, i, j) = r2;
            	tex(1, i, j) = g2;
            	tex(2, i, j) = b2;
			}
            tex(3, i, j) = 1.0;
        }
    }
    input.init_targ(nx, ny, target); 
    input.load(internal_format, format, data_type, tex.getDataPtr()); 
	input.clamp();
	input.point();

	return t;
}
//----------------------------------------------------------------------
TexOGL* Textures::createCheckerBoard(float r1, float g1, float b1, float r2, float g2, float b2, int mx, int my)
// Each internal square has width mx and height my, horizontal and vertical offset are zero
{
	if (data_type != GL_FLOAT) {
		printf("createBWNoise: data_type should be GL_FLOAT\n");
		return 0;
	}

	if (nb_internal_channels != 4) {
		printf("Textures::createCheckBoard, only works for 4-channel textures\n");
		exit(0);
	}

	TexOGL* t = new TexOGL();
	TexOGL& input = *t;

	Array3D tex(4, nx, ny);

	for (int j = 0; j < ny; j++) {
		int jm = j/my;
		int rj = jm % 2;
    	for (int i = 0; i < nx; i++) {
			int im = i/mx;
			int ri = im % 2;
			if ((ri - rj) == 0) {
            	tex(0, i, j) = r1;
            	tex(1, i, j) = g1;
            	tex(2, i, j) = b1;
			} else {
            	tex(0, i, j) = r2;
            	tex(1, i, j) = g2;
            	tex(2, i, j) = b2;
			}
            tex(3, i, j) = 1.0;
        }
    }

	printf("nx,ny= %d, %d\n", nx, ny);
    input.init_targ(nx, ny, target); 
    input.load(internal_format, format, data_type, tex.getDataPtr()); 
	input.clamp();
	input.point();

	return t;
}
//----------------------------------------------------------------------
TexOGL* Textures::createRadial(float a, float b)
// Physical domain is [-1,1]
{
	if (data_type != GL_FLOAT) {
		printf("createBWNoise: data_type should be GL_FLOAT\n");
		return 0;
	}

	if (nb_internal_channels != 4) {
		printf("Textures::createRadial, only works for 4-channel textures\n");
		exit(0);
	}

	TexOGL* t = new TexOGL();
	TexOGL& input = *t;

	Array3D tex(4, nx, ny);
	float dx = 2./(nx-1.);
	float dy = 2./(ny-1.);

	for (int j = 0; j < ny; j++) {
		float y = -1.+dy*j;
    	for (int i = 0; i < nx; i++) {
			float x = -1.+dx*i;
            tex(0, i, j) = a*x;
           	tex(1, i, j) = b*y;
           	tex(2, i, j) = 0.;
            tex(3, i, j) = 1.0;
        }
    }

    input.init_targ(nx, ny, target); 
    input.load(internal_format, format, data_type, tex.getDataPtr()); 
	input.clamp();
	input.point();

	return t;
}
//----------------------------------------------------------------------
TexOGL* Textures::createCircular(float a, float b)
// Physical domain is [-1,1]
{
	if (data_type != GL_FLOAT) {
		printf("createBWNoise: data_type should be GL_FLOAT\n");
		return 0;
	}

	if (nb_internal_channels != 4) {
		printf("Textures::createCircular, only works for 4-channel textures\n");
		exit(0);
	}

	TexOGL* t = new TexOGL();
	TexOGL& input = *t;

	Array3D tex(4, nx, ny);
	float dx = 2./(nx-1.);
	float dy = 2./(ny-1.);

	for (int j = 0; j < ny; j++) {
		float y = -1.+dy*j;
    	for (int i = 0; i < nx; i++) {
			float x = -1.+dx*i;
            tex(0, i, j) = -a*y;
           	tex(1, i, j) =  b*x;
           	tex(2, i, j) = 0.;
            tex(3, i, j) = 1.0;
        }
    }

    input.init_targ(nx, ny, target); 
    input.load(internal_format, format, data_type, tex.getDataPtr()); 
	input.clamp();
	input.point();

	return t;
}
//----------------------------------------------------------------------
TexOGL* Textures::createFloatCheck(float base, float incr) 
{
	if (data_type != GL_FLOAT) {
		printf("createBWNoise: data_type should be GL_FLOAT\n");
		return 0;
	}

	if (nb_internal_channels != 4) {
		printf("Textures::createFloatCheck, only works for 4-channel textures\n");
		exit(0);
	}

	TexOGL* t = new TexOGL();
	TexOGL& input = *t;

	Array3D tex(4, nx, ny);

	for (int j = 0; j < ny; j++) {
    	for (int i = 0; i < nx; i++) {
            tex(0, i, j) = base + i*incr;
           	tex(1, i, j) = base + j*incr;
           	tex(2, i, j) = 0.;
            tex(3, i, j) = base + i*incr;
        }
    }

    input.init_targ(nx, ny, target); 
    input.load(internal_format, format, data_type, tex.getDataPtr()); 
	input.clamp();
	input.point();

	return t;
}
//----------------------------------------------------------------------
TexOGL* Textures::createGaussian(float rms)
// grayscale gaussian
{
	if (data_type != GL_FLOAT) {
		printf("createGaussian: data_type should be GL_FLOAT\n");
		return 0;
	}

	if (nb_internal_channels != 4) {
		printf("Textures::createGaussian, only works for 4-channel textures\n");
		exit(0);
	}

    float val, p;
	TexOGL* t = new TexOGL();
	TexOGL& input = *t;

	Array3D tex(4, nx, ny);
	float dx = 2./(nx-1.);
	float dy = 2./(ny-1.);

	for (int j = 0; j < ny; j++) {
		float y = -1.+dy*j;
    	for (int i = 0; i < nx; i++) {
			float x = -1.+dx*i;
			float f = exp(-(x*x+y*y)*rms);
            tex(0, i, j) = 1.;// f;//f;
           	tex(1, i, j) = 1.;// f;//f;
           	tex(2, i, j) = 1.;// f;//f;
            tex(3, i, j) = f;
            //tex(3, i, j) = 1.0;
        }
    }

    input.init_targ(nx, ny, target); 
    input.load(internal_format, format, data_type, tex.getDataPtr()); 
	input.clamp();
	input.point();

	return t;
}
//----------------------------------------------------------------------
TexOGL* Textures::createGaussianBW(float rms)
// BW gaussian
{
	if (data_type != GL_FLOAT) {
		printf("createGaussian: data_type should be GL_FLOAT\n");
		return 0;
	}

	if (nb_internal_channels != 4) {
		printf("Textures::createGaussianBW, only works for 4-channel textures\n");
		exit(0);
	}

    float val, p;
	TexOGL* t = new TexOGL();
	TexOGL& input = *t;

	Array3D tex(4, nx, ny);
	float dx = 2./(nx-1.);
	float dy = 2./(ny-1.);

	// WRONG: in this implementation, each pixel has different gray value. 

	for (int j = 0; j < ny; j++) {
		float y = -1.+dy*j;
    	for (int i = 0; i < nx; i++) {
			float x = -1.+dx*i;
			float f = exp(-(x*x+y*y)*rms);
			float c = u.rand_float();
			//c = (c < 0.5) ? 0. : 1.;
            tex(0, i, j) = c;// f;//f;
           	tex(1, i, j) = c;// f;//f;
           	tex(2, i, j) = c;// f;//f;
            tex(3, i, j) = f;
            //tex(3, i, j) = 1.0;
        }
    }

    input.init_targ(nx, ny, target); 
    input.load(internal_format, format, data_type, tex.getDataPtr()); 
	input.clamp();
	input.point();

	return t;
}
//----------------------------------------------------------------------
TexOGL* Textures::userDefined(Array3D& data)
{
	if (data_type != GL_FLOAT) {
		printf("userDefined: data_type should be GL_FLOAT\n");
		exit(0);
		return 0;
	}

	if (nb_internal_channels != 4) {
		printf("Textures::createUserDefined, only works for 4-channel textures\n");
		exit(0);
	}

	TexOGL* t = new TexOGL();
	TexOGL& input = *t;

	int* dims = data.getDims();
	setSize(dims[1], dims[2]); // dims[0] is RGBA

printf("user defined: size: dims[]: %d, %d, %d\n", dims[0], dims[1], dims[2]);
//printf("nx, ny= %d, %d\n", nx, ny);
//exit(0);
	input.init_targ(nx, ny, target);
    input.load(internal_format, format, data_type, data.getDataPtr()); 
	//input.repeat();
	input.clamp();
	input.point();
	return t;
}
//----------------------------------------------------------------------
TexOGL* Textures::createTestFloat()
{
// special texture to test floating point operations and linear interpolation

	if (data_type != GL_FLOAT) {
		printf("testFloat: data_type should be GL_FLOAT\n");
		exit(0);
		return 0;
	}

	if (nb_internal_channels != 4) {
		printf("Textures::createTestFloat, only works for 4-channel textures\n");
		exit(0);
	}

    float val, p;
	TexOGL* t = new TexOGL();
	TexOGL& input = *t;

	Array3D tex(4, nx, ny);

	// red: -0.7 0.7 (first two texels)
	// green: -2.7 2.7 (first two texels);
	// blue: 0. 1.  (first two texels
	// alpha = 1.0;
	// 
	// green: -0.7

	tex.setTo((GE_FLOAT) 2.7);
	#if 1
	tex(0, 0,0) = -0.7;
	tex(0, 1,0) =  0.7;
	tex(1, 0,0) = -2.7;
	tex(1, 1,0) =  2.7;
	tex(2, 0,0) =  0.;
	tex(2, 1,0) =  1.;
	#endif

    input.init_targ(nx, ny, target); 
    input.load(internal_format, format, data_type, tex.getDataPtr()); 
	input.clamp();
	//input.linear();
	input.point();
	return t;
}
//----------------------------------------------------------------------
TexOGL* Textures::createEmpty()
{
	TexOGL* t = new TexOGL();
	TexOGL& input = *t;

	// to be filled by 

    input.init_targ(nx, ny, target); 
    input.load(internal_format, format, data_type, 0);
	input.clamp();
	input.point();
	return t;
}
//----------------------------------------------------------------------

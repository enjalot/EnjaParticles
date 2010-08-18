
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <sys/time.h>
#include <errno.h>
#include <time.h>
#include "platform.h"
#include "utils.h"
#include "tex_ogl.h"

//----------------------------------------------------------------------
//Utils::Utils(Framework& fmk) : fmwk(fmk)
Utils::Utils()
{
	// need a random number between 0 and 1
	// get nb of sec since beginning of time :-)
	double seed = getSeed();
	Random::Set(seed);
}
//----------------------------------------------------------------------
GLuint Utils::gen_quad()
{
	GLuint quad = glGenLists(1);
	glNewList(quad, GL_COMPILE);

	draw_quad();

	glEndList();
	return quad;
}
//----------------------------------------------------------------------
GLuint Utils::gen_quad(TexOGL& tex0)
{
	GLuint quad = glGenLists(1);
	glNewList(quad, GL_COMPILE);

	draw_quad(tex0);

	glEndList();
	return quad;
}
//----------------------------------------------------------------------
GLuint Utils::gen_quad_multi(TexOGL& tex0)
{
	GLuint quad = glGenLists(1);
	glNewList(quad, GL_COMPILE);

	draw_quad_multi(tex0);

	glEndList();
	return quad;
}
//----------------------------------------------------------------------
GLuint Utils::gen_quad_multi(TexOGL& tex0, TexOGL& tex1)
{
	GLuint quad = glGenLists(1);
	glNewList(quad, GL_COMPILE);

	draw_quad_multi(tex0, tex1);

	glEndList();
	return quad;
}
//----------------------------------------------------------------------
GLuint Utils::gen_quad_multi(TexOGL& tex0, TexOGL& tex1, TexOGL& tex2)
{
	GLuint quad = glGenLists(1);
	glNewList(quad, GL_COMPILE);

	draw_quad_multi(tex0, tex1, tex2);

	glEndList();
	return quad;
}
//----------------------------------------------------------------------
GLuint Utils::gen_quad_multi(TexOGL& tex0, TexOGL& tex1, TexOGL& tex2, TexOGL& tex3)
{
	GLuint quad = glGenLists(1);
	glNewList(quad, GL_COMPILE);

	draw_quad_multi(tex0, tex1, tex2, tex3);

	glEndList();
	return quad;
}
//----------------------------------------------------------------------
// textures as arguments helps handle 2D versus RECT textures
void Utils::draw_quad_multi(TexOGL& tex0)
{
// quad for the entire screen in range [-1,1] x [-1,1]

	 int w0 = tex0.getTarget() == GL_TEXTURE_2D ? 1. : tex0.getWidth();
	 int h0 = tex0.getTarget() == GL_TEXTURE_2D ? 1. : tex0.getHeight();
	 //printf("w0, h0= %d, %d\n", w0, h0);

     glBegin(GL_QUADS);
     glMultiTexCoord2f(GL_TEXTURE0, 0., 0.);
     glVertex3f(-1., -1., 0.);
     glMultiTexCoord2f(GL_TEXTURE0, w0, 0.);
     glVertex3f( 1.,-1., 0.);
     glMultiTexCoord2f(GL_TEXTURE0, w0, h0);
     glVertex3f( 1., 1., 0.);
     glMultiTexCoord2f(GL_TEXTURE0, 0., h0);
     glVertex3f(-1., 1., 0.);
     glEnd();
}
//----------------------------------------------------------------------
// textures as arguments helps handle 2D versus RECT textures
void Utils::draw_quad_multi(TexOGL& tex0, TexOGL& tex1)
{
// quad for the entire screen in range [-1,1] x [-1,1]

	 int w0 = tex0.getTarget() == GL_TEXTURE_2D ? 1. : tex0.getWidth();
	 int w1 = tex1.getTarget() == GL_TEXTURE_2D ? 1. : tex1.getWidth();
	 int h0 = tex0.getTarget() == GL_TEXTURE_2D ? 1. : tex0.getHeight();
	 int h1 = tex1.getTarget() == GL_TEXTURE_2D ? 1. : tex1.getHeight();

     glBegin(GL_QUADS);
     glMultiTexCoord2f(GL_TEXTURE0, 0., 0.);
     glMultiTexCoord2f(GL_TEXTURE1, 0., 0.);
     glVertex3f(-1., -1., 0.);
     glMultiTexCoord2f(GL_TEXTURE0, w0, 0.);
     glMultiTexCoord2f(GL_TEXTURE1, w1, 0.);
     glVertex3f( 1.,-1., 0.);
     glMultiTexCoord2f(GL_TEXTURE0, w0, h0);
     glMultiTexCoord2f(GL_TEXTURE1, w1, h1);
     glVertex3f( 1., 1., 0.);
     glMultiTexCoord2f(GL_TEXTURE0, 0., h0);
     glMultiTexCoord2f(GL_TEXTURE1, 0., h1);
     glVertex3f(-1., 1., 0.);
     glEnd();
}
//----------------------------------------------------------------------
// textures as arguments helps handle 2D versus RECT textures
void Utils::draw_quad_multi(TexOGL& tex0, TexOGL& tex1, TexOGL& tex2)
{
// quad for the entire screen in range [-1,1] x [-1,1]

	 int w0 = tex0.getTarget() == GL_TEXTURE_2D ? 1. : tex0.getWidth();
	 int w1 = tex1.getTarget() == GL_TEXTURE_2D ? 1. : tex1.getWidth();
	 int w2 = tex2.getTarget() == GL_TEXTURE_2D ? 1. : tex2.getWidth();
	 int h0 = tex0.getTarget() == GL_TEXTURE_2D ? 1. : tex0.getHeight();
	 int h1 = tex1.getTarget() == GL_TEXTURE_2D ? 1. : tex1.getHeight();
	 int h2 = tex2.getTarget() == GL_TEXTURE_2D ? 1. : tex2.getHeight();

     glBegin(GL_QUADS);
     glMultiTexCoord2f(GL_TEXTURE0, 0., 0.);
     glMultiTexCoord2f(GL_TEXTURE1, 0., 0.);
     glMultiTexCoord2f(GL_TEXTURE2, 0., 0.);
     glVertex3f(-1., -1., 0.);
     glMultiTexCoord2f(GL_TEXTURE0, w0, 0.);
     glMultiTexCoord2f(GL_TEXTURE1, w1, 0.);
     glMultiTexCoord2f(GL_TEXTURE2, w2, 0.);
     glVertex3f( 1.,-1., 0.);
     glMultiTexCoord2f(GL_TEXTURE0, w0, h0);
     glMultiTexCoord2f(GL_TEXTURE1, w1, h1);
     glMultiTexCoord2f(GL_TEXTURE2, w2, h2);
     glVertex3f( 1., 1., 0.);
     glMultiTexCoord2f(GL_TEXTURE0, 0., h0);
     glMultiTexCoord2f(GL_TEXTURE1, 0., h1);
     glMultiTexCoord2f(GL_TEXTURE2, 0., h2);
     glVertex3f(-1., 1., 0.);
     glEnd();
}
//----------------------------------------------------------------------
void Utils::draw_quad_multi(TexOGL& tex0, TexOGL& tex1, TexOGL& tex2, TexOGL& tex3)
{
// quad for the entire screen in range [-1,1] x [-1,1]

	 int w0 = tex0.getTarget() == GL_TEXTURE_2D ? 1. : tex0.getWidth();
	 int w1 = tex1.getTarget() == GL_TEXTURE_2D ? 1. : tex1.getWidth();
	 int w2 = tex2.getTarget() == GL_TEXTURE_2D ? 1. : tex2.getWidth();
	 int w3 = tex3.getTarget() == GL_TEXTURE_2D ? 1. : tex3.getWidth();
	 int h0 = tex0.getTarget() == GL_TEXTURE_2D ? 1. : tex0.getHeight();
	 int h1 = tex1.getTarget() == GL_TEXTURE_2D ? 1. : tex1.getHeight();
	 int h2 = tex2.getTarget() == GL_TEXTURE_2D ? 1. : tex2.getHeight();
	 int h3 = tex3.getTarget() == GL_TEXTURE_2D ? 1. : tex3.getHeight();

     glBegin(GL_QUADS);
     glMultiTexCoord2f(GL_TEXTURE0, 0., 0.);
     glMultiTexCoord2f(GL_TEXTURE1, 0., 0.);
     glMultiTexCoord2f(GL_TEXTURE2, 0., 0.);
     glMultiTexCoord2f(GL_TEXTURE3, 0., 0.);
     glVertex3f(-1., -1., 0.);
     glMultiTexCoord2f(GL_TEXTURE0, w0, 0.);
     glMultiTexCoord2f(GL_TEXTURE1, w1, 0.);
     glMultiTexCoord2f(GL_TEXTURE2, w2, 0.);
     glMultiTexCoord2f(GL_TEXTURE3, w3, 0.);
     glVertex3f( 1.,-1., 0.);
     glMultiTexCoord2f(GL_TEXTURE0, w0, h0);
     glMultiTexCoord2f(GL_TEXTURE1, w1, h1);
     glMultiTexCoord2f(GL_TEXTURE2, w2, h2);
     glMultiTexCoord2f(GL_TEXTURE3, w3, h3);
     glVertex3f( 1., 1., 0.);
     glMultiTexCoord2f(GL_TEXTURE0, 0., h0);
     glMultiTexCoord2f(GL_TEXTURE1, 0., h1);
     glMultiTexCoord2f(GL_TEXTURE2, 0., h2);
     glMultiTexCoord2f(GL_TEXTURE3, 0., h3);
     glVertex3f(-1., 1., 0.);
     glEnd();
}
//----------------------------------------------------------------------
void Utils::draw_quad_multi(float w0, float h0, float w1, float h1, 
   float w2, float h2, float w3, float h3) 
{
     glBegin(GL_QUADS);
     glMultiTexCoord2f(GL_TEXTURE0, 0.,0.);
     glMultiTexCoord2f(GL_TEXTURE1, 0.,0.);
     glMultiTexCoord2f(GL_TEXTURE2, 0.,0.);
     glMultiTexCoord2f(GL_TEXTURE3, 0.,0.);
     glVertex3f(-1., -1., 0.);
     glMultiTexCoord2f(GL_TEXTURE0, w0,0.);
     glMultiTexCoord2f(GL_TEXTURE1, w1,0.);
     glMultiTexCoord2f(GL_TEXTURE2, w2,0.);
     glMultiTexCoord2f(GL_TEXTURE3, w3,0.);
     glVertex3f( 1.,-1., 0.);
     glMultiTexCoord2f(GL_TEXTURE0, w0, h0);
     glMultiTexCoord2f(GL_TEXTURE1, w1, h1);
     glMultiTexCoord2f(GL_TEXTURE2, w2, h2);
     glMultiTexCoord2f(GL_TEXTURE3, w3, h3);
     glVertex3f( 1., 1., 0.);
     glMultiTexCoord2f(GL_TEXTURE0, 0., h0);
     glMultiTexCoord2f(GL_TEXTURE1, 0., h1);
     glMultiTexCoord2f(GL_TEXTURE2, 0., h2);
     glMultiTexCoord2f(GL_TEXTURE3, 0., h3);
 	 glVertex3f( -1., 1., 0.);
	 glEnd();
}
//----------------------------------------------------------------------
/*
void Utils::draw_quad_multi(float width, float height)
{
	 //printf("draw_quad_multi: w,h= %f, %f\n", width, height);
     glBegin(GL_QUADS);
     glMultiTexCoord2f(GL_TEXTURE0, 0,0);
     glMultiTexCoord2f(GL_TEXTURE1, 0,0);
     glMultiTexCoord2f(GL_TEXTURE2, 0,0);
     glMultiTexCoord2f(GL_TEXTURE3, 0,0);
     glVertex3f(-1., -1., 0.);
     glMultiTexCoord2f(GL_TEXTURE0, width,0);
     glMultiTexCoord2f(GL_TEXTURE1, width,0);
     glMultiTexCoord2f(GL_TEXTURE2, width,0);
     glMultiTexCoord2f(GL_TEXTURE3, width,0);
     glVertex3f( 1.,-1., 0.);
     glMultiTexCoord2f(GL_TEXTURE0, width, height);
     glMultiTexCoord2f(GL_TEXTURE1, width, height);
     glMultiTexCoord2f(GL_TEXTURE2, width, height);
     glMultiTexCoord2f(GL_TEXTURE3, width, height);
     glVertex3f( 1., 1., 0.);
     glMultiTexCoord2f(GL_TEXTURE0, 0, height);
     glMultiTexCoord2f(GL_TEXTURE1, 0, height);
     glMultiTexCoord2f(GL_TEXTURE2, 0, height);
     glMultiTexCoord2f(GL_TEXTURE3, 0, height);
     glVertex3f(-1., 1., 0.);
     glEnd();
}
*/
//----------------------------------------------------------------------
void Utils::draw_quad_multi()
{
	float width = 1.0;
	float height = 1.0;
	//printf("draw_quad_multi(): should be not called\n");
	exit(0);
     glBegin(GL_QUADS);
     glMultiTexCoord2f(GL_TEXTURE0, 0,0);
     glMultiTexCoord2f(GL_TEXTURE1, 0,0);
     glMultiTexCoord2f(GL_TEXTURE2, 0,0);
     glMultiTexCoord2f(GL_TEXTURE3, 0,0);
     glVertex3f(-1., -1., 0.);
     glMultiTexCoord2f(GL_TEXTURE0, width,0);
     glMultiTexCoord2f(GL_TEXTURE1, width,0);
     glMultiTexCoord2f(GL_TEXTURE2, width,0);
     glMultiTexCoord2f(GL_TEXTURE3, width,0);
     glVertex3f( 1.,-1., 0.);
     glMultiTexCoord2f(GL_TEXTURE0, width,height);
     glMultiTexCoord2f(GL_TEXTURE1, width,height);
     glMultiTexCoord2f(GL_TEXTURE2, width,height);
     glMultiTexCoord2f(GL_TEXTURE3, width,height);
     glVertex3f( 1., 1., 0.);
     glMultiTexCoord2f(GL_TEXTURE0, 0,height);
     glMultiTexCoord2f(GL_TEXTURE1, 0,height);
     glMultiTexCoord2f(GL_TEXTURE2, 0,height);
     glMultiTexCoord2f(GL_TEXTURE3, 0,height);
     glVertex3f(-1., 1., 0.);
     glEnd();
}
//----------------------------------------------------------------------
void Utils::draw_quad()
{
	glBegin(GL_QUADS);
          glVertex3f( -1.0, -1.0,  0.0 );
          glVertex3f(  1.0, -1.0,  0.0 );
          glVertex3f(  1.0,  1.0,  0.0 );
          glVertex3f( -1.0,  1.0,  0.0 );
	glEnd();
}
//----------------------------------------------------------------------
void Utils::draw_quad(TexOGL& tex)
{
	int w = tex.getTarget() == GL_TEXTURE_2D ? 1. : tex.getWidth();
	int h = tex.getTarget() == GL_TEXTURE_2D ? 1. : tex.getHeight();
	//printf("draw_quad, w= %f\n", w);

	glBegin(GL_QUADS);
          glTexCoord2f( 0., 0.); glVertex3f( -1.0, -1.0,  0.0 );
          glTexCoord2f( w,  0.); glVertex3f(  1.0, -1.0,  0.0 );
          glTexCoord2f( w,  h);  glVertex3f(  1.0,  1.0,  0.0 );
          glTexCoord2f( 0., h);  glVertex3f( -1.0,  1.0,  0.0 );
	glEnd();
}
//----------------------------------------------------------------------
void Utils::draw_quad(float w)
{
	//printf("draw_quad, w= %f\n", w);
	glBegin(GL_QUADS);
          glTexCoord2f( 0., .0); glVertex3f( -1.0, -1.0,  0.0 );
          glTexCoord2f( w, 0.);  glVertex3f(  1.0, -1.0,  0.0 );
          glTexCoord2f( w, w);   glVertex3f(  1.0,  1.0,  0.0 );
          glTexCoord2f( 0., w);  glVertex3f( -1.0,  1.0,  0.0 );
	glEnd();
}
//----------------------------------------------------------------------
#if 0
// two component texture attached to FBO
void Utils::printFBO2(int curBuf, int lower_left, int upper_right, int w, int h, const char* msg)
{
	float* pixels = new float [w*h*2];

	printf("printFBO: %s\n", msg);

	glReadBuffer(fmwk.fboBuf[curBuf]);
    glReadPixels(lower_left, upper_right, w, h, GL_LUMINANCE_ALPHA, GL_FLOAT, pixels);

	for (int j=0; j < h; j++) {
	for (int i=0; i < w; i++) {
		int ii = 2*(i+w*j);
		float* p = pixels;
		//printf("i,j,pix= %d,%d, %f,%f\n", i, j, p[ii], p[ii+1]);
	}}
	printf("----- printFBO2 DONE -----\n");

	delete [] pixels;
}
#endif
//----------------------------------------------------------------------
#if 0
void Utils::printFBO(int curBuf, int lower_left, int upper_right, int w, int h, const char* msg)
{
	float* pixels = new float [w*h*4];

	glReadBuffer(fmwk.fboBuf[curBuf]);
    glReadPixels(lower_left, upper_right, w, h, GL_RGBA, GL_FLOAT, pixels);

	for (int j=0; j < h; j++) {
	for (int i=0; i < w; i++) {
		int ii = 4*(i+w*j);
		float* p = pixels;
		//printf("i,j,pix= %d,%d, %f,%f, %f,%f\n", i, j, p[ii], p[ii+1], p[ii+2], p[ii+3]);
	}}
	printf("----- printFBO DONE -----\n");

	delete [] pixels;
}
#endif
//----------------------------------------------------------------------
float* Utils::readPixels()
{
    float* pixels = new float [20*20*4];
    glReadBuffer(GL_FRONT);
    glReadPixels(0, 0, 20, 20, GL_RGBA, GL_FLOAT, pixels);
    for (int i=0; i < 20; i++) {
    for (int j=0; j < 20; j++) {
		int ix = i + j*20;
        printf("pixels lower left RGBA (%d,%d): %f, %f, %f, %f\n", i, j, 
			pixels[4*ix], pixels[4*ix+1], pixels[4*ix+2], pixels[4*ix+3]);
    }}
    return pixels;
}
//----------------------------------------------------------------------
void Utils::readPixels(GLenum buffer)
{
    float* pixels = new float [20*20*4];
    glReadBuffer(buffer);
    glReadPixels(1, 1, 20, 20, GL_RGBA, GL_FLOAT, pixels);
    for (int i=0; i < 20; i++) {
    for (int j=0; j < 20; j++) {
		int ix = i + j*20;
        printf("pixels lower left RGBA (%d,%d): %f, %f, %f, %f\n", i, j, 
			pixels[4*ix], pixels[4*ix+1], pixels[4*ix+2], pixels[4*ix+3]);
    }}
    delete [] pixels;
}
//----------------------------------------------------------------------
void Utils::printBuffer(GLenum buffer, int xo, int yo, int w, int h)
{
// assume float4

    float* pixels = new float [w*h*4];
    glReadBuffer(buffer);
    glReadPixels(xo, yo, w, h, GL_RGBA, GL_FLOAT, pixels);
    for (int j=0; j < h; j++) {
    for (int i=0; i < w; i++) {
		int ix = i + j*w;
		float* p = pixels + 4*ix;
        printf("pixels lower left RGBA (%d,%d): %f, %f, %f, %f\n", i+xo, j+yo, 
			p[0], p[1], p[2], p[3]);
    }}
    delete [] pixels;
}
//----------------------------------------------------------------------
void Utils::readPixels(float* pixels, int nx, int ny)
// assumes floating point buffer
{
    glReadBuffer(GL_FRONT);
    glReadPixels(0, 0, nx, ny, GL_RGBA, GL_FLOAT, pixels);
}
//----------------------------------------------------------------------
void Utils::readPixels(GLenum buffer, float* pixels, int nx, int ny)
// assumes floating point buffer
{
    glReadBuffer(buffer);
    glReadPixels(0, 0, nx, ny, GL_RGBA, GL_FLOAT, pixels);
}
//----------------------------------------------------------------------
void Utils::reshape(int tex_size)
{
    // matrix setup
    glViewport(0, 0, tex_size, tex_size);
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glOrtho(-1, 1, -1, 1, 1, 10); // Macs
    //glOrtho(-1, 1, -1, 1, 1, 10); // PC's and linux
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();
    glTranslatef(0.f, 0.f, -2.f);
}
//----------------------------------------------------------------------
void Utils::disableTextures(int i) 
{
	switch (i) {
	case 3:
		glActiveTexture(GL_TEXTURE3);
		glBindTexture(TARGET, 0);
		glDisable(TARGET);
	case 2:
		glActiveTexture(GL_TEXTURE2);
		glBindTexture(TARGET, 0);
		glDisable(TARGET);
	case 1:
		glActiveTexture(GL_TEXTURE1);
		glBindTexture(TARGET, 0);
		glDisable(TARGET);
	case 0:
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(TARGET, 0);
		glDisable(TARGET);
		break;
	}
}
//----------------------------------------------------------------------
void Utils::enableTextures(int i)
{
	switch (i) {
	case 3:
		glActiveTexture(GL_TEXTURE3);
		glEnable(TARGET);
		break;
	case 2:
		glActiveTexture(GL_TEXTURE2);
		glEnable(TARGET);
		break;
	case 1:
		glActiveTexture(GL_TEXTURE1);
		glEnable(TARGET);
		break;
	case 0:
		glActiveTexture(GL_TEXTURE0);
		glEnable(TARGET);
		break;
	}
}
//----------------------------------------------------------------------
void Utils::checkError(char* msg)
{
#if 0
	printf("glerror: %s,  %s\n", msg, gluErrorString(glGetError())); // error
#endif
}
//----------------------------------------------------------------------
float Utils::rand_float()
{
	//return ((float)rand() / (float) RAND_MAX);
	return uniform.Next();
}
//----------------------------------------------------------------------
int Utils::rand_int(int a, int b)
{
	return (int) (a + (b-a)*rand_float());
}
//----------------------------------------------------------------------
double Utils::rand_float(double a, double b)
{
	return a + (b-a)*rand_float();
}
//----------------------------------------------------------------------
void Utils::initializeData()
{
	internalFormats.push_back(InternalFormat("GL_RGBA16F_ARB", GL_RGBA16F_ARB));
	internalFormats.push_back(InternalFormat("GL_RGBA32F_ARB", GL_RGBA32F_ARB));
	internalFormats.push_back(InternalFormat("GL_RGBA", GL_RGBA));
	internalFormats.push_back(InternalFormat("GL_RGBA8", GL_RGBA8));
	internalFormats.push_back(InternalFormat("GL_RGBA12", GL_RGBA12));
	internalFormats.push_back(InternalFormat("GL_RGBA16", GL_RGBA16));

	formats.push_back(Format("GL_RGBA", GL_RGBA));

	targets.push_back(Target("GL_TEXTURE_RECTANGLE_ARB", GL_TEXTURE_RECTANGLE_ARB)); // on ATI
	//targets.push_back(Target("GL_TEXTURE_2D", GL_TEXTURE_2D));
	//targets.push_back(Target("GL_TEXTURE_RECTANGLE_NV", GL_TEXTURE_RECTANGLE_NV));   // on Nvidia
	dataTypes.push_back(DataType("GL_FLOAT", GL_FLOAT));

	for (int i=0; i < targets.size(); i++) {
	for (int j=0; j < internalFormats.size(); j++) {
		//printf("i,j,target: %d, %d, %s\n", i,j, targets[i].name);
		texmetas.push_back(new TexMeta(&targets[i], &internalFormats[j], &formats[0], &dataTypes[0]));
	}}

	for (int i=0; i < internalFormats.size(); i++) {
		printf("ifmt name: %s, id= 0x%x\n", internalFormats[i].name, internalFormats[i].format);
	}
	for (int i=0; i < formats.size(); i++) {
		printf("fmt name: %s, id= 0x%x\n", formats[i].name, formats[i].format);
	}
	for (int i=0; i < targets.size(); i++) {
		printf("tg name: %s, id= 0x%x\n", targets[i].name, targets[i].target);
	}
	for (int i=0; i < dataTypes.size(); i++) {
		printf("dtype name: %s, id= 0x%x\n", dataTypes[i].name, dataTypes[i].type);
	}

	for (int i=0; i < texmetas.size(); i++) {
		TexMeta* tm = texmetas[i];
		printf("textmeta[%d], ifmt= %s, fmt=%s, type=%s, tg= %s\n", i, 
			tm->ifmt->name, tm->fmt->name, tm->dt->name, tm->tg->name);
		printf("textmeta[%d], ifmt= 0x%x, fmt=0x%x, type=0x%x, tg= 0x%x\n", i, 
			tm->ifmt->format, tm->fmt->format, tm->dt->type, tm->tg->target);
	}
}
//----------------------------------------------------------------------
void Utils::setTexFormat(TexOGL& tex, InternalFormat& ifmt, Format& fmt, DataType& type)
{
	tex.setFormat(ifmt.format, fmt.format, type.type);
}
//----------------------------------------------------------------------
void Utils::setTexTarget(TexOGL& tex, Target& target)
{
	tex.setTarget(target.target);
}
//----------------------------------------------------------------------
void Utils::setTexFormat(TexOGL& tex, GLint iformat, GLenum format, GLenum dataType)
{
	tex.setFormat(internalFormats[iformat].format, formats[format].format, dataTypes[dataType].type);
}
//----------------------------------------------------------------------
void Utils::setTexTarget(TexOGL& tex, GLenum target)
{
	tex.setTarget(targets[target].target);
}
//----------------------------------------------------------------------
double Utils::getSeed() 
{
	struct timeval tp;
	struct timezone tzp;
	gettimeofday(&tp, &tzp);
	long nbMicroSeconds = tp.tv_usec;
	//printf("nbMicroSeconds= %ld\n", nbMicroSeconds);
	double seed= (nbMicroSeconds / 1000.);
	double integ;
	double frac = modf(seed, &integ);
	return frac;
}
//----------------------------------------------------------------------
int Utils::nano_sleep(double sleep_time)
{
 struct timespec tv;
 /* Construct the timespec from the number of whole seconds... */
 tv.tv_sec = (time_t) sleep_time;
 /* ... and the remainder in nanoseconds. */
 tv.tv_nsec = (long) ((sleep_time - tv.tv_sec) * 1e+9);

 while (1)
 {
  /* Sleep for the time specified in tv. If interrupted by a
    signal, place the remaining time left to sleep back into tv. */
  int rval = nanosleep (&tv, &tv);
  if (rval == 0)
   /* Completed the entire sleep time; all done. */
   return 0;
  else if (errno == EINTR)
   /* Interrupted by a signal. Try again. */
   continue;
  else 
   /* Some other error; bail out. */
   return rval;
 }
 return 0;
}
//----------------------------------------------------------------------
unsigned int Utils::nextPow2( unsigned int x )
{
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return ++x;
}
//----------------------------------------------------------------------

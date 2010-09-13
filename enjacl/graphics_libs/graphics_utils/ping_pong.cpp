
#include <stdlib.h> // exit(0)
#include "ping_pong.h"
#include "framebufferObject.h"
#include "textures.h"

using namespace std;

/// Create a Pingpong buffer
PingPong::PingPong(int nx, int ny)
{
	clock1 = new GE::Time("pingpong::begin");
	clock2 = new GE::Time("pingpong::end");

	printf("Pingpong constructor\n");

	if (ny == 0) ny = nx;
	curBuf = 0;
	curTex = 1-curBuf;
	fbo = new FramebufferObject();

	szx = nx;
	szy = ny;

	// Assume GL_TEXTURE_2D, floating point FBO, GL_RGBA
	// (the most inefficient)
	// create two textures
	// specify internal format, data type, format
	// create framebufferObject

	Textures tx(szx, szy);
	//internal_format = GL_RGBA;
	internal_format = FLOAT_BUFFER;
	format 		= GL_RGBA;
	data_type 	= GL_FLOAT;
	target = TARGET; // GL_TEXTURE_2D; // set via method

	tx.setTarget(target);
	tx.setFormat(internal_format, format, data_type);
	//GLuint target = GL_TEXTURE_RECTANGLE_NV;

	// For textures to be independent, I must do a glTextImage2D at some point

	// How does empty texture work? Is space reserved?
	tex[0] = tx.createEmpty();
	tex[1] = tx.createEmpty();

	// textures and fbo must already be defined
	setupFbo(); 
}
//----------------------------------------------------------------------
PingPong::PingPong(TexOGL* init_tex, TexOGL* init_tex1)
/// Initialize PingPong with a given texture
{
	clock1 = new GE::Time("pingpong::begin");
	clock2 = new GE::Time("pingpong::end");

	szx = init_tex->getWidth();
	szy = init_tex->getHeight();

	curBuf = 0;
	curTex = 1-curBuf;
	fbo = new FramebufferObject();

	internal_format = init_tex->getIFormat();
	format 			= init_tex->getFormat();
	data_type 		= init_tex->getDataType();
	target 			= init_tex->getTarget();

	// How does empty texture work? Is space reserved?
	tex[0] = init_tex;  // copy texture (not sure whether data is copied)
	tex[1] = init_tex1; // default copy constructor (I should create one)

	// textures and fbo must already be defined
	setupFbo(); 
}
//----------------------------------------------------------------------
PingPong::PingPong(TexOGL* init_tex_)
/// Initialize PingPong with a given texture
{
	clock1 = new GE::Time("pingpong::begin");
	clock2 = new GE::Time("pingpong::end");

	TexOGL& init_tex = *init_tex_;
	szx = init_tex.getWidth();
	szy = init_tex.getHeight();

	curBuf = 0;
	curTex = 1-curBuf;
	fbo = new FramebufferObject();

	internal_format = init_tex.getIFormat();
	format 			= init_tex.getFormat();
	data_type 		= init_tex.getDataType();
	target 			= init_tex.getTarget();

	// How does empty texture work? Is space reserved?
	tex[curTex] = init_tex_;  // simple pointer

	Textures tx(init_tex);
	tex[curBuf] = tx.createEmpty();

	// textures and fbo must already be defined
	setupFbo(); 
}
//----------------------------------------------------------------------
PingPong::~PingPong()
{
	// I should be able to delete twice without incurring an error. Not possible so far
	delete tex[0];
	delete tex[1];
	delete clock1;
	delete clock2;
	delete fbo;
}
//----------------------------------------------------------------------
string PingPong::info()
{
	/// print information about this object
	switch (internal_format) {
	case FLOAT_BUFFER: 
		internal_format_s = "FLOAT_BUFFER";
		break;
	case GL_RGBA: 
		internal_format_s = "GL_RGBA";
		break;
	default: 
		internal_format_s = "not defined";
		break;
	}

	switch (format) {
	case GL_RGBA:
		format_s = "GL_RGBA";
		break;
	default: 
		format_s = "not defined";
		break;
	}

	switch (target) {
	case GL_TEXTURE_2D:
		target_s = "GL_TEXTURE_2D";
		break;
	case GL_TEXTURE_RECTANGLE_NV:
	//case GL_TEXTURE_RECTANGLE_ARB:
		target_s = "GL_TEXTURE_RECTANGLE_NV";
		break;
	default: 
		target_s = "not defined";
		break;
	}

	string comma = ", ";

	string result =  target_s + ", " + internal_format_s + comma + format_s;
	return result; 
}
//----------------------------------------------------------------------
void PingPong::printInfo(const char* msg)
{
	if (msg) {
		printf("%s: ", msg);
	}
	printf("%d x %d, %s\n", tex[0]->getWidth(), tex[0]->getHeight(),   info().c_str());
}
//----------------------------------------------------------------------
TexOGL& PingPong::getBuffer()
{
	return *tex[curBuf];
}
//----------------------------------------------------------------------

/// get current Texture (pointer)
TexOGL& PingPong::getTexture()
{
	return *tex[curTex];
}
//----------------------------------------------------------------------

/// copy current Texture (allocate memory as well)
TexOGL& PingPong::copyTexture()
{
	TexOGL* texNew = new TexOGL(*tex[curTex]);
	return *texNew;
}
//----------------------------------------------------------------------
void PingPong::enable()
{
	fbo->Bind();
}
//----------------------------------------------------------------------
void PingPong::disable()
{
	FramebufferObject::Disable();
}
//----------------------------------------------------------------------

/// Set buffers to draw into this framebuffer object
void PingPong::begin(bool enableFBO)
{
	// I should not enable texture. The user might not wish to 
	// use textures
	//glEnable(GL_TEXTURE_2D); // or tex[0]->target()
	//tex[curTex]->bind(); // read from this texture


	clock1->begin();
	if (enableFBO)  {
		fbo->Bind();  // perhaps should not be done each time
	}

	glPushAttrib(GL_VIEWPORT_BIT | GL_COLOR_BUFFER_BIT); // .001ms
	glViewport(0, 0, szx, szy); // .001ms
	printf("pingpong begin: szx,szy= %d, %d\n", szx, szy);

	glDrawBuffer(fboBuf[curBuf]); // write to this buffer //.001ms
	clock1->end();

	//clock1->print();
	clock1->reset();
}
//----------------------------------------------------------------------

/// Disable this framebuffer object, prevent drawing into 
/// these buffers
void PingPong::end(bool disableFBO)
{
	clock2->begin();

	glPopAttrib();
	if (disableFBO) {
		FramebufferObject::Disable();
	}
	swap();

	clock2->end();

	//clock2->print();
	clock2->reset();
}
//----------------------------------------------------------------------
void PingPong::setupFbo()
// DO NOT FORGET TO INITIALIZE THE TEXTURES
{
	//printf("max color attachments: %d\n", fbo->GetMaxColorAttachments());
	fboBuf[0] = GL_COLOR_ATTACHMENT0_EXT;
	fboBuf[1] = GL_COLOR_ATTACHMENT1_EXT;
	fbo->Bind();

    fbo->AttachTexture(fboBuf[0], tex[0]->getTarget(), tex[0]->tex_obj());
    fbo->AttachTexture(fboBuf[1], tex[1]->getTarget(), tex[1]->tex_obj());

	if (fbo->IsValid() != 1) {
		printf("setupFBO(tex1, tex2) *** valid: %d\n", fbo->IsValid());
		exit(0);
	}
	FramebufferObject::Disable();
	return;
}
//----------------------------------------------------------------------
void PingPong::toBackBuffer()
{
	drawFBOtoScreen_b(GL_BACK, getTexture());
}
//----------------------------------------------------------------------
void PingPong::checkError(char* msg)   //   also sin superquadric
{
	printf("glerror: %s,  %s\n", msg, gluErrorString(glGetError())); // error
}
//----------------------------------------------------------------------
void PingPong::drawFBOtoScreen_b(GLenum screen, TexOGL& texture)
// unfortunately, the image on the screen appears to be 8 bit. 
// simply draw the texture to the screen. The second argument is not necessary.
{
	// Perhaps I should remember the active shader and reactivate it at the end?
	// This method is called because most people will forget it. 
	glUseProgram(0);

	if (screen != GL_FRONT && screen != GL_BACK) {
		printf("drawFBOtoScreen: screen should be GL_FRONT or GL_BACK\n");
		exit(0);
	}

	FramebufferObject::Disable(); // must be done BEFORE glDrawBuffer()
	glDrawBuffer(screen); // front or back

	texture.bind();
	glEnable(texture.getTarget());

	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();
	float rng;
	glOrtho(0., 1., 0., 1., -1., -10.); // for Equalizer
	rng = 2.;
	//gluOrtho2D(0., 1., 0., 1.);  // for local machine
	//rng = 0.;
	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();

	//if (texture.getHeight() != texture.getWidth()) {
		//printf("drawFBOToScreen: texture must be square\n");
		//exit(0);
	//}
	//float sz = texture.getTarget() == GL_TEXTURE_2D ? 1.0 : texture.getHeight();
	float sz;

	if (texture.getTarget() == GL_TEXTURE_2D) {
		szx = 1.;
	} else {
		szx = texture.getWidth();
		szy = texture.getHeight();
	}

	// should not clear the screen here. It should be cleared at the beginning of the 
	// main display function
	//glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);
	//


	glBegin(GL_QUADS);
	  glTexCoord2f(0., 0.);
	  glVertex3f(0., 0., rng);

	  glTexCoord2f(szx, 0.);
	  glVertex3f(1., 0., rng);

	  glTexCoord2f(szx, szy);
	  glVertex3f(1., 1., rng);

	  glTexCoord2f(0, szy);
	  glVertex3f(0., 1., rng);
	glEnd();

	glMatrixMode(GL_PROJECTION);
	glPopMatrix();
	glMatrixMode(GL_MODELVIEW);
	glPopMatrix();

	glDisable(texture.getTarget());
}
//----------------------------------------------------------------------
void PingPong::drawTexture(TexOGL& tex, int border)
{
//printf("inside drawTexture with border\n");
//printf("pingpong size: %d\n", getTexture().getWidth());
//printf("tex size: %d\n", tex.getWidth());

// Write into the pingpong buffer the texture tex
// Assume that both textures have the SAME border with the 
// same width.
// Later, this could be generalized

	glDisable(getTexture().getTarget());
	glDisable(GL_TEXTURE_2D);

	// tex: coarse
	// this: fine
	int wp = getTexture().getWidth(); // includes border
	int hp = getTexture().getHeight(); // fine
	int wt = tex.getWidth(); // includes border
	int ht = tex.getHeight();
	//printf("wp,hp= %d, %d\n", wp, hp); //1026
	//printf("wt,ht= %d, %d\n", wt, ht); //514

	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();
	gluOrtho2D(0., wp, 0., hp);
	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();

	tex.bind();
	glEnable(tex.getTarget());

	glBegin(GL_QUADS);
	  glTexCoord2i(border, border);
	  glVertex2i(border, border);

	  glTexCoord2i(wt-border, border);
	  glVertex2i(wp-border, border);

	  glTexCoord2i(wt, ht-border);
	  glVertex2i(wp-border, hp-border);

	  glTexCoord2i(border, ht-border);
	  glVertex2i(border, hp-border);
	glEnd();

	glMatrixMode(GL_PROJECTION);
	glPopMatrix();
	glMatrixMode(GL_MODELVIEW);
	glPopMatrix();

	glDisable(tex.getTarget());
	u.disableTextures(0);
}
//----------------------------------------------------------------------
void PingPong::drawTexture(TexOGL& tex)
{
	glDisable(getTexture().getTarget());
	glDisable(GL_TEXTURE_2D);
	int sz = tex.getTarget() == GL_TEXTURE_2D ? 1.0 : tex.getHeight();

	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();
	gluOrtho2D(0., 1., 0., 1.);
	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();

	tex.bind();
	glEnable(tex.getTarget());

	glBegin(GL_QUADS);
	  glTexCoord2f(0., 0.);
	  glVertex2f(0., 0.);

	  glTexCoord2f(sz, 0.);
	  glVertex2f(1., 0.);

	  glTexCoord2f(sz, sz);
	  glVertex2f(1., 1.);

	  glTexCoord2f(0, sz);
	  glVertex2f(0., 1.);
	glEnd();

	glMatrixMode(GL_PROJECTION);
	glPopMatrix();
	glMatrixMode(GL_MODELVIEW);
	glPopMatrix();

	glDisable(tex.getTarget());
	u.disableTextures(0);
}
//----------------------------------------------------------------------
void PingPong::swap()
{
	curTex =     curBuf; // texture written to buffer, which will be read from
	curBuf = 1 - curBuf; // buffer to write into when next called
}
//----------------------------------------------------------------------
void PingPong::undoSwap()
{
	swap();
}
//----------------------------------------------------------------------
void PingPong::print(int i1, int j1, int w, int h) 
{
	glDisable(getTexture().getTarget());
	fbo->Bind();  // perhaps should not be done each time
	//FramebufferObject::Disable();
	glFinish();
    float* pixels = new float [w*h*4];
    glReadBuffer(fboBuf[1-curBuf]); // buffer just written to 
    //glReadBuffer(GL_BACK); // buffer just written to 
    glReadPixels(i1, j1, w, h, GL_RGBA, GL_FLOAT, pixels);

    for (int j=0; j < h; j++) {
    for (int i=0; i < w; i++) {
		int ix = w*j + i;
        printf("pixels[%d,%d]: %f, %f, %f, %f\n", i+i1,j+j1, pixels[4*ix],
               pixels[4*ix+1], pixels[4*ix+2], pixels[4*ix+3]);
    }}
    delete [] pixels;
	FramebufferObject::Disable();
}
//----------------------------------------------------------------------
void PingPong::getData(float* pixels)
{
	glDisable(getTexture().getTarget());
	//printf("szx= %d, szy= %d\n", szx, szy);
	fbo->Bind();  // perhaps should not be done each time
	glFinish();
    glReadBuffer(fboBuf[1-curBuf]); // buffer just written to 
    glReadPixels(0, 0, szx, szy, GL_RGBA, GL_FLOAT, pixels);

	#if 0
	printf("---- inside getData -----\n");
    for (int j=0; j < getHeight(); j++) {
    for (int i=0; i < getWidth(); i++) {
		int ix = getWidth()*(j+0) + (i+0);
        printf("pixels[%d,%d]: %f, %f, %f, %f\n", i,j, pixels[4*ix],
               pixels[4*ix+1], pixels[4*ix+2], pixels[4*ix+3]);
    }}
	printf("---- end inside getData -----\n");
	#endif

	FramebufferObject::Disable();
}
//----------------------------------------------------------------------
#if 0
void PingPong::getData(float* pixels)
{
	glDisable(getTexture().getTarget());
	printf("szx= %d, szy= %d\n", szx, szy);
	fbo->Bind();  // perhaps should not be done each time
	glFinish();
    glReadBuffer(fboBuf[1-curBuf]); // buffer just written to 
    glReadPixels(0, 0, szx, szy, GL_RGBA, GL_FLOAT, pixels);
	FramebufferObject::Disable();
}
#endif
//----------------------------------------------------------------------
void PingPong::unbind()
{
	FramebufferObject::Disable();
}
//----------------------------------------------------------------------
void PingPong::bind()
{
	fbo->Bind();
}
//----------------------------------------------------------------------
// copy contents of PBO into texture of PingPong buffer (not the buffer)
void PingPong::setSubTexture(GLuint pbo, int xoff, int yoff, int width, int height)
{
	//No effect. Perhaps the rows should be aligned on 8 bytes on 64-bit machine?
	//glPixelStorei(GL_UNPACK_ALIGNMENT, 4); // no effect


#ifndef GL_VERSION_2_1
	printf("GL_PIXEL_UNPACK_BUFFER not defined on GL versions < 2_1\n");
	exit(0);

#else
	
	glClampColorARB(GL_CLAMP_FRAGMENT_COLOR_ARB, GL_FALSE);

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
	TexOGL& tex = getTexture();
    tex.bind();
	int level = 0;
	// The textures are float RGBA
    glTexSubImage2D(target, level, xoff, yoff, 
             //width/2, height/2, format, data_type, NULL);
             width, height, format, data_type, NULL);  // WHY CANNOT UPDATE FULL TEXTURE?
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
	checkError("---after glTexSubImage2D"); // ERROR
#endif
}
//----------------------------------------------------------------------

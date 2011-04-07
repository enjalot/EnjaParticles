#ifndef _PING_PONG_H_
#define _PING_PONG_H_

#include <string>
#include "tex_ogl.h"
#include "utils.h"
#include "timege.h"

// the two textures are swapped by default when doing end()
// can undo the swap with undoSwap()

//class string;
//class TexOGL;
class FramebufferObject;

class PingPong
{
private:
	 /// format for the underlying textures
     GLenum       target;
	 GLuint       internal_format;
	 GLenum       format;
	 GLenum       data_type;

	 GE::Time* clock1;
	 GE::Time* clock2;

	 int szx, szy;
	
	 Utils u;

	 std::string target_s;
	 std::string internal_format_s;
	 std::string format_s;
	 std::string data_type_s;

    FramebufferObject* fbo;
	GLenum fboBuf[2];  // used for fbo (not usually required)
    TexOGL* tex[2];
    int curBuf; // internal management of FBO texture
	int curTex;

public: 
    /// Create a Pingpong buffer
	/// square texture if ny=0
    PingPong(int nx, int ny=0);
	PingPong(TexOGL* init_tex); // const texOGL& does not work! WHY?
	//PingPong(TexOGL& init_tex, TexOGL& init_tex1);
	PingPong(TexOGL* init_tex, TexOGL* init_tex1);

    ~PingPong();

	int getWidth() { return szx; }
	int getHeight() { return szy; }

	/// generate an information string about this object
	std::string info();

	/// internal, type, type
	void setFormat();

	/// get the texture attached to the current drawing buffer
    TexOGL& getBuffer();

    /// get current Texture (pointer)
    TexOGL& getTexture();

    /// copy current Texture (allocate memory as well)
    TexOGL& copyTexture();

    /// Set buffers to draw into this framebuffer object
    void begin(bool enableFBO=true); //  enable fbo by default (for convenience)
	//
    /// Disable this framebuffer object, prevent drawing into 
    /// these buffers
    void end(bool disableFBO=true); // diable fbo by default (for convenience)


	void enable(); // enable fbo
	void disable(); // disable fbo

	/// draw last result drawn to the texture into the backbuffer
	/// Outstanding Shaders are disabled. The last shader used is not retained. 
	void toBackBuffer();

	/// OpenGL error checking
	void checkError(char* msg);   

	/// print information string
	void printInfo(const char* msg=0);

	/// Draw an entire 2D texture into the buffer
	/// Assumes that the FBO is already initialized
	/// We are already within begin() and end()
	void drawTexture(TexOGL& tex);

	// do not draw the border (border: width of border)
	void drawTexture(TexOGL& tex, int border);

	/// set underlying textures to point()
	void point() {
		tex[0]->bind();
		tex[0]->point();
		tex[1]->bind();
		tex[1]->point();
	}

	/// set underlying textures to linear
	void linear() {
		tex[0]->bind();
		tex[0]->linear();
		tex[1]->bind();
		tex[1]->linear();
	}

	void swap();
	void undoSwap(); // simply executes swap

	// print contents of buffer just written to
	void print(int i1, int j1, int w, int h);

	void bind();
	void unbind();

	GLenum getTexFBOid() { return fboBuf[curTex]; } 
	GLenum getBufFBOid() { return fboBuf[curBuf]; }

	void setSubTexture(GLuint pbo, int xoff, int yoff, int width, int height);

	// get the data stored in the pingpong buffer
	void getData(float* pixels);

private:
/// set up the framebuffer object
	void setupFbo();

	void drawFBOtoScreen_b(GLenum screen, TexOGL& texture);
};

#endif

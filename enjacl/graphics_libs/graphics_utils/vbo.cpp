#include "vbo.h"

using namespace std;

//----------------------------------------------------------------------
template <class P, class C> 
VBO<P,C>::VBO() 
{
	first = 0;
}
//----------------------------------------------------------------------
template<class P, class C> 
VBO<P,C>::~VBO()
{}
//----------------------------------------------------------------------
template<class P, class C> 
void VBO<P,C>::create(const vector<P>* vertex, const vector<C>* color)
{
	printf("vertex= %d\n", vertex);
	this->pts = vertex;
	this->color = color;
    // vertex info
	nbBytes = pts->size() * sizeof(P);

    glGenBuffers(2, &vbo_v);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_v); // leave empty to write into it
    glBufferData(GL_ARRAY_BUFFER, nbBytes, &(*pts)[0], GL_STREAM_COPY);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

	exit(0);
    
    // color information
    nbBytes = color->size() * sizeof(C); // usually 4 color channels
    
    glGenBuffers(1, &vbo_c);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_c); // leave empty to write into it
    glBufferData(GL_ARRAY_BUFFER, nbBytes, &(*color)[0], GL_STREAM_COPY);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}
//----------------------------------------------------------------------
template<class P, class C> 
void VBO<P,C>::draw(GLenum mode, int count)
{
	glBindBuffer(GL_ARRAY_BUFFER, vbo_v); 
		glVertexPointer (3, GL_FLOAT, 0, 0);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_c); 
	    glColorPointer(3, GL_FLOAT, 0, 0);

	glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_COLOR_ARRAY);

	glDrawArrays(GL_POINTS, first, count); // error
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glDisableClientState(GL_VERTEX_ARRAY);
	glDisableClientState(GL_COLOR_ARRAY);
}
//----------------------------------------------------------------------


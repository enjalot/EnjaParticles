template<class P, class T>
class VBO_T      // Texture
{
private:
	const std::vector<P>* pts;   /// vertex info (2,3,4 components)
	const std::vector<T>* tex; 
	GLuint vbo_v;  /// vertex id
	GLuint vbo_t;  /// texture id
	int nbPrimitives; /// number of primitives to display
	int nbBytes;     /// number of bytes used by the objects
	int first; /// offset
	Utils u;

public:
	VBO_T();
	~VBO_T();
	void create(const std::vector<P>* vertex, const std::vector<T>* tex);
	/// mode: GL_POINTS, GL_QUADS, etc.
	/// count: number of objects to draw
	// At this time, each point has 3 floats. MUST BE IMPROVED/GENERALIZED
	void draw(GLenum mode, int count);

private:
	VBO_T(const VBO_T&);
};

//======================================================================
template <class P, class T> 
VBO_T<P,T>::VBO_T() 
{
	first = 0;
}
//----------------------------------------------------------------------
template<class P, class T> 
VBO_T<P,T>::~VBO_T()
{}
//----------------------------------------------------------------------
template<class P, class T> 
void VBO_T<P,T>::create(const vector<P>* vertex, const vector<T>* tex)
{
	this->pts = vertex;
	this->tex = tex;
    // vertex info
	nbBytes = pts->size() * sizeof(P);

    glGenBuffers(1, &vbo_v);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_v); // leave empty to write into it
    //glBufferData(GL_ARRAY_BUFFER, nbBytes, &(*pts)[0], GL_STREAM_COPY);
    glBufferData(GL_ARRAY_BUFFER, nbBytes, &(*pts)[0], GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    
    nbBytes = tex->size() * sizeof(T); // 1,2,3,4 channels
    
    glGenBuffers(1, &vbo_t);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_t); // leave empty to write into it
    glBufferData(GL_ARRAY_BUFFER, nbBytes, &(*tex)[0], GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}
//----------------------------------------------------------------------
template<class P, class T> 
void VBO_T<P,T>::draw(GLenum mode, int count)
{
	//u.checkError("draw: 1\n");
	glBindBuffer(GL_ARRAY_BUFFER, vbo_v); 
		glVertexPointer (3, GL_FLOAT, 0, 0);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_t); 
	    glTexCoordPointer(3, GL_FLOAT, 0, 0);
	//u.checkError("draw: 2\n");

	glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_TEXTURE_COORD_ARRAY);

	glDrawArrays(mode, first, count); // error
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glDisableClientState(GL_VERTEX_ARRAY);
	glDisableClientState(GL_TEXTURE_COORD_ARRAY);
}
//----------------------------------------------------------------------


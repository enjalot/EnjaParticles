template<class P, class C>
class VBO
{
public:
	GLuint vbo_v;  /// vertex id
	GLuint vbo_c;  /// color id
private:
	const std::vector<P>* pts;   /// vertex info (2,3,4 components)
	const std::vector<C>* color; /// color info (3 or 4 components)
	int nbPrimitives; /// number of primitives to display
	int nbBytes;     /// number of bytes used by the objects
	int first; /// offset

public:
	VBO();
	~VBO();
	void create(const std::vector<P>* vertex, const std::vector<C>* color);
	// update existing vbo with new data (not necessarily efficently)
	void update(const std::vector<P>* vertex, const std::vector<C>* color);
	/// mode: GL_POINTS, GL_QUADS, etc.
	/// count: number of objects to draw
	void draw(GLenum mode, int count);

private:
	VBO(const VBO&);
};

//======================================================================
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
	this->pts = vertex;
	this->color = color;
    // vertex info
	nbBytes = pts->size() * sizeof(P);


    glGenBuffers(1, &vbo_v);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_v); // leave empty to write into it
    //glBufferData(GL_ARRAY_BUFFER, nbBytes, &(*pts)[0], GL_STREAM_COPY);
    glBufferData(GL_ARRAY_BUFFER, nbBytes, &(*pts)[0], GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    
    // color information
    nbBytes = color->size() * sizeof(C); // usually 4 color channels

	printf("VBO::create, nbBytes= %d\n", nbBytes);

	#if 0
	printf("pts size: %d\n"< pts->size());
	for (int i=0; i < (int) pts->size(); i++) {
		printf("x,y,z= %f, %f, %f\n", (*pts)[i].x, (*pts)[i].y, (*pts)[i].z);
		printf("x,y,z= %f, %f, %f\n", (*color)[i].x, (*color)[i].y, (*color)[i].z);
	}
	#endif

    
    glGenBuffers(1, &vbo_c);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_c); // leave empty to write into it
    //glBufferData(GL_ARRAY_BUFFER, nbBytes, &(*color)[0], GL_STREAM_COPY);
    glBufferData(GL_ARRAY_BUFFER, nbBytes, &(*color)[0], GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

	printf("nbBytes= %d\n", nbBytes);
	//printf("create: vbo_v= %d\n", (int) vbo_v);
	//printf("create: vbo_c= %d\n", (int) vbo_c);
}
//----------------------------------------------------------------------
template<class P, class C> 
void VBO<P,C>::update(const vector<P>* vertex, const vector<C>* color)
{
	this->pts = vertex;
	this->color = color;
    // vertex info
	nbBytes = pts->size() * sizeof(P);

    glGenBuffers(1, &vbo_v);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_v); // leave empty to write into it
    //glBufferData(GL_ARRAY_BUFFER, nbBytes, &(*pts)[0], GL_STREAM_COPY);
    glBufferData(GL_ARRAY_BUFFER, nbBytes, &(*pts)[0], GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    
    // color information
    nbBytes = color->size() * sizeof(C); // usually 4 color channels

	//printf("VBO::create, nbBytes= %d\n", nbBytes);
	//for (int i=0; i < (int) pts->size(); i++) {
		//printf("x,y,z= %f, %f, %f\n", (*pts)[i].x, (*pts)[i].y, (*pts)[i].z);
		//printf("x,y,z= %f, %f, %f\n", (*color)[i].x, (*color)[i].y, (*color)[i].z);
	//}

    
    glGenBuffers(1, &vbo_c);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_c); // leave empty to write into it
    //glBufferData(GL_ARRAY_BUFFER, nbBytes, &(*color)[0], GL_STREAM_COPY);
    glBufferData(GL_ARRAY_BUFFER, nbBytes, &(*color)[0], GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}
//----------------------------------------------------------------------
template<class P, class C> 
void VBO<P,C>::draw(GLenum mode, int count)
{
	// First argument is not used\n");
	printf("draw\n");
	//printf("draw, vbo_v= %d\n", (int) vbo_v);
	// what about if int, etc. ?
	int nbCh_p = sizeof(P) / sizeof(float); 
	int nbCh_c = sizeof(C) / sizeof(float); 

	//int nbCh_p = 3;
	//int nbCh_c = 4;

	printf("nbCh_p = %d\n", nbCh_p);
	printf("nbCh_c = %d\n", nbCh_c);

	printf("nbCh_p= %d, nbCh_c= %d\n", nbCh_p, nbCh_c);

	//glBindBuffer(GL_ARRAY_BUFFER, vbo_v); 
	glBindBuffer(GL_ARRAY_BUFFER, 1); 
		glVertexPointer (nbCh_p, GL_FLOAT, 0, 0);
	//glBindBuffer(GL_ARRAY_BUFFER, vbo_c); 
	glBindBuffer(GL_ARRAY_BUFFER, 2); 
	    glColorPointer(nbCh_c, GL_FLOAT, 0, 0);

	glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_COLOR_ARRAY);

	//glDrawArrays(mode, first, count); // error
	// first gives "cannot accsess memory"
	glDrawArrays(mode, 0, count); // error
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glDisableClientState(GL_VERTEX_ARRAY);
	glDisableClientState(GL_COLOR_ARRAY);
}
//----------------------------------------------------------------------


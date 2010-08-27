const GLfloat light_ambient[]  = { 0.4f, 0.4f, 0.04, 1.0f };
const GLfloat light_diffuse[]  = { 1.0f, 1.0f, 1.0f, 1.0f };
const GLfloat light_specular[] = { 1.0f, 1.0f, 1.0f, 1.0f };
const GLfloat light_position[] = { 2.0f, 5.0f, 5.0f, 0.0f };

const GLfloat mat_ambient[]    = { 0.2f, 0.2f, 1.0f, 1.0f };
const GLfloat mat_diffuse[]    = { 0.2f, 0.8f, 1.0f, 1.0f };
const GLfloat mat_specular[]   = { 1.0f, 1.0f, 1.0f, 1.0f };
const GLfloat high_shininess[] = { 100.0f };

const GLfloat mat_ambient_back[]    = { 0.5f, 0.2f, 0.2f, 1.0f };
const GLfloat mat_diffuse_back[]    = { 1.0f, 0.2f, 0.2f, 1.0f };

// Program entry point 


void define_lights_and_materials()
{

	// Ambient: color in the absence of light
    glLightfv(GL_LIGHT0, GL_AMBIENT,  light_ambient);
    glLightfv(GL_LIGHT0, GL_DIFFUSE,  light_diffuse);
    glLightfv(GL_LIGHT0, GL_SPECULAR, light_specular);
    glLightfv(GL_LIGHT0, GL_POSITION, light_position);

    glMaterialfv(GL_FRONT, GL_AMBIENT,   mat_ambient);
    glMaterialfv(GL_FRONT, GL_DIFFUSE,   mat_diffuse);
    glMaterialfv(GL_FRONT, GL_SPECULAR,  mat_specular);
    glMaterialfv(GL_FRONT, GL_SHININESS, high_shininess);

	glMaterialfv(GL_BACK, GL_AMBIENT,   mat_ambient_back);
    glMaterialfv(GL_BACK, GL_DIFFUSE,   mat_diffuse_back);
    glMaterialfv(GL_BACK, GL_SPECULAR,  mat_specular);
    glMaterialfv(GL_BACK, GL_SHININESS, high_shininess);

	// lighting only has effect if lighting is on
	// color of vertices have no effect without material 
	//   if phone is one (GL_SMOOTH)

    glEnable(GL_LIGHT0);
	//glEnable(GL_LIGHT_MODEL_TWO_SIDE);

	// I don't really undersand GL_COLOR_MATERIAL
    glEnable(GL_COLOR_MATERIAL);
    //glEnable(GL_LIGHTING);
    glEnable(GL_NORMALIZE); // makes sure normal remain unit length

		// Blending is only active when blending mode is ON
	// Page 129 in GLUT course
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
}
//------------------------------------------------------


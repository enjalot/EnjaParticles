#ifndef GLUEEEEEE_H
#define GLUEEEEEE_H

#include "importgl.h"

#ifdef __cplusplus
extern "C" {
#endif


void gluLookAt(GLfloat eyex, GLfloat eyey, GLfloat eyez,
          GLfloat centerx, GLfloat centery, GLfloat centerz,
          GLfloat upx, GLfloat upy, GLfloat upz);

void gluLookAtf(float eyex, float eyey, float eyez,
          float centerx, float centery, float centerz,
          float upx, float upy, float upz);


#ifdef __cplusplus
}
#endif



#endif

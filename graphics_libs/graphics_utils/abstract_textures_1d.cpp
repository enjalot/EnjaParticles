
#include <stdio.h>
#include <stdlib.h>
#include "abstract_textures_1d.h"
#include "Array3D.h"

//----------------------------------------------------------------------
AbstractTextures1D::AbstractTextures1D()
{
	internal_format = 0;
	format 		= 0;
	data_type 	= 0;
	target 		= 0;
}
//----------------------------------------------------------------------
AbstractTextures1D::AbstractTextures1D(int sz)
{
	internal_format = 0;
	format 		= 0;
	data_type 	= 0;
	target 		= 0;
	setSize(sz);
}
//----------------------------------------------------------------------
void AbstractTextures1D::setFormat(GLint i_fmt, GLenum fmt, GLenum type)
{
	internal_format = i_fmt;
	format 			= fmt;
	data_type 		= type;
}
//----------------------------------------------------------------------
void AbstractTextures1D::setSize(int nx)
{
	this->nx = nx;
}
//----------------------------------------------------------------------
void AbstractTextures1D::setTarget(GLenum target)
{
	this->target = target;
}
//----------------------------------------------------------------------

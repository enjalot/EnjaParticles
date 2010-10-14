
#include <stdio.h>
#include <stdlib.h>
#include "abstract_textures.h"
#include "Array3D.h"

//----------------------------------------------------------------------
AbstractTextures::AbstractTextures()
{
	internal_format = 0;
	format 		= 0;
	data_type 	= 0;
	target 		= 0;
}
//----------------------------------------------------------------------
AbstractTextures::AbstractTextures(int sz)
{
	internal_format = 0;
	format 		= 0;
	data_type 	= 0;
	target 		= 0;
	setSize(sz, sz);
}
//----------------------------------------------------------------------
AbstractTextures::AbstractTextures(int szx, int szy)
{
	internal_format = 0;
	format 		= 0;
	data_type 	= 0;
	target 		= 0;
	setSize(szx, szy);
}
//----------------------------------------------------------------------
AbstractTextures::~AbstractTextures()
{}
//----------------------------------------------------------------------
void AbstractTextures::setFormat(GLint i_fmt, GLenum fmt, GLenum type)
{
	internal_format = i_fmt;
	format 			= fmt;
	data_type 		= type;
}
//----------------------------------------------------------------------
void AbstractTextures::setInternalFormat(GLint i_fmt)
{
	internal_format = i_fmt;
}
//----------------------------------------------------------------------
void AbstractTextures::setFormat(GLenum fmt)
{
	format = fmt;
}
//----------------------------------------------------------------------
void AbstractTextures::setDataType(GLenum type)
{
	data_type = type;
}
//----------------------------------------------------------------------
void AbstractTextures::setSize(int nx, int ny)
{
	this->nx = nx;
	this->ny = ny;
}
//----------------------------------------------------------------------
void AbstractTextures::setTarget(GLenum target)
{
	this->target = target;
}
//----------------------------------------------------------------------

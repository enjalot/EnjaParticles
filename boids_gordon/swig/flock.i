/* File : flock.i */
%module flock
%{
/* Put headers and other declarations here */
#define SWIG_FILE_WITH_INIT
#include "boids.h"
#include "structs.h"
#include "domain/IV.h"
%}

%include "typemaps.i"
%include "std_vector.i"
%include "cstring.i"

/** Import decl details for float4 
 * These are necessary because we cant map
 *  "friend operator" routines to python's 
 * __add__, __mul__, etc.
 * using automatic aspects of swig. We manually
 * extend the type to have those operators completed
 * In this file: 
 */ 
%include "float4.i"

/** Swig cant provide std::vector of any type. We need to
    declare the types we want available */
namespace std
{
    %template(intvec) vector<int>; 
    %template(float4vec) vector<rtps::float4>; 
    %template(vecvec) vector<vector<int> >; 
}

%include "domain/IV.h"
%include "boids.h"

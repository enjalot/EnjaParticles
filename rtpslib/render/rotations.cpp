#ifndef _ROTATE_GE_H_
#define _ROTATE_GE_H_

// Derived from: (no use of quaternions)
// http://www.mactech.com/articles/mactech/Vol.15/15.03/NaturalObjectRotation/index.html

//----------------------------------------------------------------------
class Rotate
{
	Rotate();
	void Vectorize();
	void ZeroHysteresisRotation();
	void FreeRotateWithMouse();
};
//----------------------------------------------------------------------

#endif

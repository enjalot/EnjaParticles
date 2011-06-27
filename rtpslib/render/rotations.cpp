
#include "rotations.h"

//----------------------------------------------------------------------
Rotate::Rotate()
{}
//----------------------------------------------------------------------
//Vectorize()
// Create a 3D position vector on a unit sphere based on
// mouse hitPt, sphere's screen origin, and sphere radius in
// pixels. Return zero on success or non-zero if hit point 
// was outside the sphere. This routine uses Quickdraw[TM] 3D
// and assumes that the relevant QD3D headers and library
// files have been included. Also need to #include <math.h>.

#define BOUNDS_ERR 1L

long Rotate::Vectorize( 
   Point *hit, 
   Point *origin, 
   float radius, 
   TQ3Vector3D *vec ) 
{
   float            x,y,z, modulus;
   
   x = (float)(hit->h - origin->h)/radius;
   y = (float)(hit->v - origin->v)/radius;

   y *= -1.0;         // compensate for "inverted" screen y-axis!
   
   modulus = x*x + y*y;
   
   if (modulus > 1.)                      // outside radius!
      return BOUNDS_ERR;
   
   z = sqrt(1. - (x*x + y*y) );    // compute fictitious 'z' value
   
   Q3Vector3D_Set( vec, x,y,z );   // compute pseudo-3D mouse position
   
   return 0L;
}
//----------------------------------------------------------------------
Rotate::ZeroHysteresisRotation()
// From two 3D vectors representing the positions of 
// points on a unit sphere, calculate an axis of rotation
// and an amount of rotation such that Point A can be
// moved along a geodesic to Point B.
// CAUTION: Error-checking omitted for clarity.

void ZeroHysteresisRotation( TQ3Vector3D v1, TQ3Vector3D v2 ) 
{
   TQ3Vector3D               cross;
   TQ3Matrix4x4               theMatrix;
   TQ3Point3D                  orig = { 0.,0.,0. };
   float                        dot,angle;
   
   dot = Q3Vector3D_Dot( &v1, &v2 );

   if (dot == 1.0) 
      return;                                        // nothing to do

   Q3Vector3D_Cross( &v1, &v2, &cross ); // axis of rotation
   Q3Vector3D_Normalize( &cross,&cross );   
      
   angle = 2.*acos( dot );                           // angle of rotation
   
   // set up a rotation around our chosen axis...
   Q3Matrix4x4_SetRotateAboutAxis(&theMatrix,
                     &orig, &cross, angle);
   
   Q3Matrix4x4_Multiply(   &gDocument.fRotation,
                     &theMatrix,
                     &gDocument.fRotation);     // multiply

   DocumentDraw( &gDocument ) ;                 // draw
}
//----------------------------------------------------------------------
//Rotate::FreeRotateWithMouse()
// Call this function from the main event loop to
// do free-rotations of 3D objects

// the mouse action radius in pixels:
#define RADIUS_VALUE 300.0

void Rotate::FreeRotateWithMouse(void) 
{
   Point             now, oldPt, center;
   WindowPtr       win = FrontWindow();
   float            radius = RADIUS_VALUE;
   TQ3Vector3D   v1,v2;
   long               err;

   GetMouse( &oldPt );

   center.h = (win->portRect.right - win->portRect.left)/2;

   center.v = (win->portRect.bottom - win->portRect.top)/2;

   while (StillDown()) 
   {
      GetMouse( &now );

      err = Vectorize(&oldPt, &center, RADIUS_VALUE, &v1 );
      err += Vectorize(&now, &center, RADIUS_VALUE, &v2 );

      if (!err)
         ZeroHysteresisRotation( v1,v2 );

      oldPt = now;
   }
}
//----------------------------------------------------------------------
//----------------------------------------------------------------------
//----------------------------------------------------------------------

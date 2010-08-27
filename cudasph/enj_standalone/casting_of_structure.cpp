
#include <stdio.h>

typedef struct Vec4
{   
    float x;
    float y;
    float z;
    float w;

#if 0
    Vec4(){};
    Vec4(float xx, float yy, float zz, float ww):
        x(xx),
        y(yy),
        z(zz),
        w(ww)
    {}
#endif
} Vec4;

// size: 4*4 = 16 floats
// shared memory = 65,536 bytes = 16,384 floats
//               = 1024 triangles
typedef struct Triangle
{
    Vec4 verts[3];
    Vec4 normal;    //should pack this in verts array
    //float dummy; // for more efficient global -> shared
} Triangle;


int main()
{
	printf("sizeof(Triangle) = %d\n", sizeof(Triangle));
	return 0;
}

// SOMETHING WRONG with allcoation/accesses ()

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
//#include <Amira/HxMessage.h>
#include "Array3D.h"
//#include "Vec3i.h"

//Array3D& operator+(Array3D& a, Array3D& b)

#if 0
Array3D(int type, Vec3i& npts, Vec3i origin, GE_FLOAT* array=0)
{
    switch (type) {
        case CSTYLE:
            Vec3i mx_ = npts - origin + Vec3i(1,1,1);
            break;
        case FORTRANSTYLE:
            //Vec3i mx_ = npts - origin;
            break;
    }
}
#endif
//----------------------------------------------------------------------
Array3D:: Array3D(GE_FLOAT* array, Vec3i& n_)
{
    init(array, n_[0], n_[1], n_[2]);
}
//----------------------------------------------------------------------
Array3D::Array3D(GE_FLOAT* array, Vec3i& n_, Vec3i& nm_)
{
    init(array, n_[0], n_[1], n_[2], nm_[0], nm_[1], nm_[2]);
}
//----------------------------------------------------------------------
Array3D::Array3D(Vec3i& n_)
{
    init(0, n_[0], n_[1], n_[2]);
}
//----------------------------------------------------------------------
Array3D::Array3D(Vec3i& n_, Vec3i& nm_)
{
    init(0, n_[0], n_[1], n_[2], nm_[0], nm_[1], nm_[2]);
}
//----------------------------------------------------------------------
void Array3D::computeMinMax()
{
    min = max = data[0];
    for (int i=0; i < npts; i++) {
        // //- printf("i,data= %d, %g, min/max= %g, %g\n", i, data[i], min, max);
        min = data[i] < min ? data[i] : min;
        max = data[i] > max ? data[i] : max;
    }
}
//----------------------------------------------------------------------
/***
int Array3D::copyTo(const Array3D& a) const
// copy to "a" from "this"
// return 0 if 0k, -1 otherwise
{
    const int* dims = a.getDims();
	//- printf("inside const copyTo\n");
    if (dims[0] != np1 || dims[1] != np2 || dims[2] != np3) {
        return(-1);
    }
    GE_FLOAT* ptr = (GE_FLOAT*) a.getDataPtr();
	//- printf("ptr= %ld, npts= %d\n", (long) ptr, npts);
    memcpy(ptr, data, sizeof(GE_FLOAT)*npts);
    return(0);
}
***/
//----------------------------------------------------------------------
int Array3D::copyTo(Array3D& a)
// copy to "a" from "this"
// return 0 if 0k, -1 otherwise
{
	//- printf("inside copyTo\n");
    int* dims = a.getDims();
    if (dims[0] != np1 || dims[1] != np2 || dims[2] != np3) {
        return(-1);
    }
    GE_FLOAT* ptr = a.getDataPtr();
	//- printf("copyTo:: npts= %d\n", npts);
    memcpy(ptr, data, sizeof(GE_FLOAT)*npts);
    return(0);
}
//----------------------------------------------------------------------
int Array3D::copyFrom(Array3D& a)
// copy from "a" to "this"
{
    // //- printf("enter copyFrom\n");
    int* dims = a.getDims();
    // //- printf("dims= %d, %d, %d\n", dims[0], dims[1], dims[2]);
    if (dims[0] != np1 || dims[1] != np2 || dims[2] != np3) {
		// //- printf("dims from: %d, %d, %d\n", dims[0], dims[1], dims[2]);
		// //- printf("dims to: %d, %d, %d\n", np1, np2, np3);
        //- printf("error in copyFrom (array3D.cpp)\n");
        return(-1);
    }
    GE_FLOAT* ptr = a.getDataPtr();

    memcpy(data, ptr, sizeof(GE_FLOAT)*npts);
    return(0);
}
//----------------------------------------------------------------------
void Array3D::setTo(GE_FLOAT value)
{
    for (int i=0; i < npts; i++) {
        data[i] = value;
    }
    //memset(data, (int) value, sizeof(GE_FLOAT)*npts);
}
//----------------------------------------------------------------------
int Array3D::copyTo(Array3D& a, Vec3i& toOrigin, Vec3i& fromRange, Vec3i& fromOrigin)
// copy from "this" to "a"
{
    Vec3i fromMaxDims = fromOrigin + fromRange;
    fromMaxDims.clampMaxTo(maxDims);
    fromOrigin.clampMinTo(origin);
    fromRange = fromMaxDims - fromOrigin;
    Vec3i toMaxDims = toOrigin + fromRange;

    if (toOrigin < a.getOrigin()) return -1;
    if ((toOrigin+fromRange) > a.getMaxDims()) return -1;

    for (int k=0; k < fromRange.z(); k++) {
    for (int j=0; j < fromRange.y(); j++) {
    for (int i=0; i < fromRange.x(); i++) {
        a.set(i+toOrigin.x(), j+toOrigin.y(), k+toOrigin.z(), 
                get(i+fromOrigin.x(), j+fromOrigin.y(), k+fromOrigin.z()));
    }}}
    return(0);
}
//----------------------------------------------------------------------
int Array3D::copyFrom(Array3D& a, Vec3i& fromOrig, Vec3i& fromRange, Vec3i& toOrigin)
{
    Vec3i fromMaxDims = fromOrig + fromRange;
    fromMaxDims.clampMaxTo(a.getMaxDims());
    fromOrig.clampMinTo(a.getOrigin());
    fromRange = fromMaxDims - fromOrig;
    Vec3i toMaxDims = toOrigin + fromRange;

    if (toOrigin < this->origin) return -1;
    if ((toOrigin+fromRange) > maxDims) return -1;

    for (int k=0; k < fromRange[2]; k++) {
    for (int j=0; j < fromRange[1]; j++) {
    for (int i=0; i < fromRange[0]; i++) {
        set(i+toOrigin.x(), j+toOrigin.y(), k+toOrigin.z(),  a.get(i+fromOrig.x(), j+fromOrig.y(), k+fromOrig.z()));
    }}}
    return(0);
}
//----------------------------------------------------------------------
void Array3D::resize(Vec3i& arraySize, Vec3i& origin) // max nb points in 3 directions, origin
{
    if (externalPtr) {
        //- printf("Array3d: Error: cannot resize external array\n");
        return;
    }
    if (data) {
        delete [] data;
        data = 0;
    }
    init(0, arraySize[0], arraySize[1], arraySize[2], origin[0], origin[1], origin[2]);
}
//----------------------------------------------------------------------
void Array3D::resize(Vec3i& arraySize) // max nb points in 3 directions, origin
{
    Vec3i zero = Vec3i(0,0,0);
    resize(arraySize, zero); // zero only gets deleted after return from resize, so ok
}
//----------------------------------------------------------------------
Array3D* Array3D::createSubArray(Vec3i& subOrigin, Vec3i& subWidth, Vec3i& newOrigin)
{
// Given the current array, cut out an array of size subWidth at origin subOrigin.
// Create new array set to the cutout with an origin at newOrigin
// The method returns new origin and width of subarray in subOriginand subWidth

    // keep origin within bounds
    Vec3i orig = subOrigin;
    Vec3i maxDimens = subOrigin + subWidth;
    if (subOrigin > maxDims) return 0;
    orig.clampMinTo(origin);
    maxDimens.clampMaxTo(totPts);

    //Vec3i size = maxDimen - subOrigin;
    subOrigin = orig;
    subWidth = maxDimens - orig;

    return( new Array3D(subWidth, newOrigin) );
}
//----------------------------------------------------------------------
void Array3D::print(const char* msg, Vec3i orig, Vec3i size)
{
    //- printf("\n-------------- %s ---------------\n", msg);
    for (int k=0; k < size[2]; k++) {
    for (int j=0; j < size[1]; j++) {
    for (int i=0; i < size[0]; i++) {
        //- printf("  i,j,k=  %d, %d, %d, arr= %g\n", i,j,k,get(i,j,k));
    }}}
    //- printf("----------------------------------\n");
}
//----------------------------------------------------------------------
#ifdef STANDALONE
void main()
{
    float* x = new float [1000];
    Array3D arr(x, 10, 10, 10);
    //- printf("arr->getDataPtr() = %ld\n", (long) arr.getDataPtr());
    exit(0);
}
#endif

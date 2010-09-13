// May 2, 2002
// Change definition of Array3D so that constructor
//    is Array3D(npts, origin) where npts is tot nb of points in each dimension
//    For compatibility with current version, create new constructor
//    Array3D(Vec3i& npts, Vec3i origin, GE_FLOAT* array=0);

#ifndef _ARRAY3D_H_
#define _ARRAY3D_H_

#include <stdio.h>
#include "GEtypes.h"
#include "Vec3i.h"

class Array3D {
private:
    GE_FLOAT* data; // data allocated in constructor, unless float* pointer passed to constructor
    int npts; // total number of points in the array
    int dims[3];
    int n1, n2, n3;  // first index varies fastest (like Fortran)
    int np1, np2, np3;  // tot nb points in each dimension
    int n1m, n2m, n3m;  
    int externalPtr;
    Vec3i maxDims;
    Vec3i origin;
    Vec3i totPts; // maxDims - origin
    GE_FLOAT min, max;
private:
	Array3D(const Array3D& a) {
		// delete current array if exists 
		//
		printf("inside copy constructor\n");
	}
public:
    enum {CSTYLE=0,FORTRANSTYLE};
    // allocation, Fortran style
    // args: max1, max2, max3, min1=min2=min3=0
    Array3D(GE_FLOAT* array, Vec3i& n_);
    Array3D(GE_FLOAT* array, Vec3i& n_, Vec3i& nm_);
    //Array3D(int type, Vec3i& npts, Vec3i origin, GE_FLOAT* array=0);
    Array3D() {}
    Array3D(Vec3i& n_);
    Array3D(Vec3i& n_, Vec3i& nm_);
    Array3D(GE_FLOAT* array, const int* n_) {
		data = 0;
        init(array, n_[0], n_[1], n_[2]);
    }
    Array3D(GE_FLOAT* array, int n1_, int n2_=1, int n3_=1) {
    // Attach Array3D object to external pointer
		data = 0;
        init(array, n1_, n2_, n3_);
    }
    Array3D(int n1_, int n2_=1, int n3_=1, int n1m_=0, int n2m_=0, int n3m_=0) {
		data = 0;
        init(0, n1_, n2_, n3_, n1m_, n2m_, n3m_); 
    }
    ~Array3D() {
		//- printf("inside Array3D::destructor\n");
        remove();
    }
    inline void remove() {
        if (externalPtr == 0) {
            delete [] data;
        }
    }
    // ATTENTION: init() changed to zero default
    void init(GE_FLOAT* array, int n1_, int n2_, int n3_, int n1m_=0, int n2m_=0, int n3m_=0) {
		//- printf("inside init\n");
		//- printf("n1,n2,n3= %d, %d, %d\n", n1_, n2_, n3_);

        n1m = n1m_; // min dimension value, origin
        n2m = n2m_;
        n3m = n3m_;

		//- printf("n1,n2,n3= %d, %d, %d\n", n1_, n2_, n3_);
		//- printf("**** array3D init: %d, %d, %d\n", n1_, n2_, n3_);
        origin.setValue(n1m, n2m, n3m);

        np1 = n1_; // size in dimension 1
        np2 = n2_;
        np3 = n3_;

        totPts.setValue(np1, np2, np3);

        // maximum dimension: array[n1m:n1, n2m:n2, n3m:n3]
        n1 = n1m + np1; // C-style: origin[0], origin[0]+1, ...,np1-1
        n2 = n2m + np2; // C-style: origin[0], origin[0]+1, ...,np1-1
        n3 = n3m + np3; // C-style: origin[0], origin[0]+1, ...,np1-1

        maxDims = origin + totPts;
        npts = np1*np2*np3;

        if (array) {
            externalPtr = 1;
            data = array;
        } else {
            externalPtr = 0;
			if (data) remove();
            data = new GE_FLOAT [npts];
        }

		dims[0] = np1;
		dims[1] = np2;
		dims[2] = np3;
    }
    const int* getDims() const {return dims;}
    int* getDims() {return dims;}
    Vec3i& getMaxDims() {return maxDims;}
    inline GE_FLOAT* getDataPtr() {return data;}
    inline const GE_FLOAT* getDataPtr() const {return data;}
    inline int getSize() {return npts;}
    inline GE_FLOAT& get(int i, int j, int k) {
        return data[(i-n1m)+np1*((j-n2m)+np2*(k-n3m))];
    }
	// "return const" required to handle operator()(....) const
    inline GE_FLOAT& get(int i, int j=0, int k=0) const { 
		// //- printf("get const\n"); // not used 
        return data[(i-n1m)+np1*((j-n2m)+np2*(k-n3m))];
    }
    inline void set(int i, int j, int k, GE_FLOAT value) {
        data[(i-n1m)+np1*((j-n2m)+np2*(k-n3m))] = value;
    }
    inline GE_FLOAT operator[](int i) { return data[i]; }
// Copy from this to array
    int copyTo(Array3D& array);
    //int copyTo(const Array3D& array) const;
    const Vec3i& getOrigin() const {return origin;} // expensive (2 objects)
    Vec3i& getOrigin() {return origin;} // expensive (2 objects)
    const Vec3i& getTotPoints() const {return totPts;}
    Vec3i& getTotPoints() {return totPts;}
// Copy to this from array
    int copyFrom(Array3D& array);
// Copy from this to array
    //int copyTo(Array3D& array, Vec3i& toOrig, Vec3i& fromOrig, Vec3i& fromRange);
    int copyTo(Array3D& array, Vec3i& toOrig, Vec3i& fromRange, Vec3i& fromOrigin);
// Copy from array array
    int copyFrom(Array3D& array, Vec3i& fromOrig, Vec3i& fromRange, Vec3i& toOrigin);
// set to value
    void setTo(GE_FLOAT value);
    void computeMinMax();
    void printMinMax(char* msg) { 
        computeMinMax();
        //- printf("%s: min/max= %g, %g\n", msg, min, max);
    }
    void setOrigin(Vec3i& origin) {
        n1m = origin[0];
        n2m = origin[1];
        n3m = origin[2];
        this->origin.setValue(n1m, n2m, n3m);
    }
    void print(const char*, Vec3i orig, Vec3i size);
    // can only resize arrays allocated internally
    void resize(Vec3i& arraySize, Vec3i& origin); // max nb points in 3 directions, origin
    void resize(Vec3i& arraySize); // max nb points in 3 directions, origin
    int getCapacity() {return npts;} // return total size of array (product of 3 dimensions) 

    // User is responsible for deleting allocated memory of subArray
    Array3D* createSubArray(Vec3i& subOrigin, Vec3i& subWidth, Vec3i& newOrigin);

	//T operator()(int i, int j=0, int k=0) { return get(i,j,k);}
	//T operator()(int i, int j=0, int k=0) const { return get(i,j,k);}
	// This is the one used for lvalue and for rvalue
	// float& operator()(int i, int j=n2m, int k=n3m) { // DOES NOT WORK: should work!!

	// shoudl use n2m and n3m as defaults
	float& operator()(int i, int j=0, int k=0) { 
		return get(i, j, k); 
	} // OK
	// float& operator ... DOES NOT WORK (get() returns GE_FLOAT
	// required if operator() used as:   sig(i,j,k) where sig is a "const"
	// float& operator()(int i, int j=n2m, int k=n3m) { // DOES NOT WORK: should work!!
	float& operator()(const int i, const int j, const int k=0) const { 
		return get(i, j, k); 
	}
	//ArrayT<T>& operator=(ArrayT<T>& arr);
	Array3D& operator=(Array3D& arr) {
		//- printf("uses = operator\n");
		Vec3i t = arr.getTotPoints();
		Vec3i o = arr.getOrigin();
		if (&arr != this) { // Is there a memory leak here? 
			init(0, t[0], t[1], t[2], o[0], o[1], o[2]);
			//t.print("t= ");
			//o.print("o= ");
			copyFrom(arr);
			//set(11,7,2,    arr(11,7,2));  // works
			//- printf("*** inside loop: this(11,7,0)= %f\n", this->get(10,7,0));
			//- printf("*** inside loop:  arr(11,7,0)= %f\n", arr.get(10,7,0));
			//int err = arr.copyTo(*this);
			//if (err != 0) {
				// //- printf("error in copyTo\n");
			//}
		}
		//- printf("operator= this(11,7,2)= %f\n", this->get(10,7,0));
		//- printf("operator= arr(11,7,2)= %f\n", arr.get(10,7,0));
		return *this;
	}

/*****
	friend Array3D& operator+(const Array3D& a, const Array3D& b) {
		int nbPts = a.getSize();
		Array3D* c = new Array3D(a.getTotPoints());
		const GE_FLOAT* af = a.getDataPtr();
		const GE_FLOAT* bf = b.getDataPtr();
		GE_FLOAT* cf = c->getDataPtr();
		for (int i=0; i < nbPts; i++) {
			cf[i] = af[i] + bf[i];
		}
		return *c;
	}
*****/

	/***
	Array3D&  operator+()
	Array3D&  operator-()
	Array3D&  operator-()
	***/
};

#endif

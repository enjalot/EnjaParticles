//================================================================================
 
// $Author: erlebach $
// $Date: 2002/01/05 15:18:28 $
// $Header: /home/erlebach/D2/src/CVS/computeCP/Vec3i.h,v 2.5 2002/01/05 15:18:28 erlebach Exp $
// Sticky tag, $Name:  $
// $RCSfile: Vec3i.h,v $
// $Revision: 2.5 $
// $State: Exp $
 
//================================================================================

#ifndef _VEC3i_H_
#define _VEC3i_H_

// The routines in Vec3i do NOT affect its size.
// sizeof(Vec3i) = 24

class Vec3i 
{
public:
    Vec3i();
    Vec3i(const Vec3i& v) {
        this->vec[0] = v.vec[0];
        this->vec[1] = v.vec[1];
        this->vec[2] = v.vec[2];
    }
    Vec3i(int x, int y=1, int z=1);
    Vec3i(const int* pt); // allocated in calling routine
    Vec3i(int* pt); // allocated in calling routine
    //Vec3i(int pt[3]);
    ~Vec3i() {};
    Vec3i(Vec3i&);
    int* getVec(); // memory allocated in class
    void getVec(int* x, int* y, int* z);
    void setValue(int x, int y, int z);
    inline void setValue(int i, int val) {vec[i] = val;}
    void print(const char *msg=0) const ;
    //void print() const {print("");}
    inline int x() const {return vec[0];}
    inline int y() const {return vec[1];}
    inline int z() const {return vec[2];}
	inline void x(int j) {vec[0] = j;}
	inline void y(int j) {vec[1] = j;}
	inline void z(int j) {vec[2] = j;}
// overloaded operators (as needed)
    friend Vec3i operator+(const Vec3i& a, const Vec3i& b) {
        return(Vec3i(a.vec[0]+b.vec[0], a.vec[1]+b.vec[1], a.vec[2]+b.vec[2]));
    }
    friend Vec3i operator-(const Vec3i& a, const Vec3i& b) {
        return(Vec3i(a.vec[0]-b.vec[0], a.vec[1]-b.vec[1], a.vec[2]-b.vec[2]));
    }
    friend Vec3i operator*(const Vec3i& a, int f) {
        return(Vec3i(a.vec[0]*f, a.vec[1]*f, a.vec[2]*f));
    }
    friend Vec3i operator*(int f, const Vec3i& a) {
        return(Vec3i(a.vec[0]*f, a.vec[1]*f, a.vec[2]*f));
    }
    friend Vec3i operator*(const Vec3i& a, float f) {
        return(Vec3i((int) (a.vec[0]*f), (int) (a.vec[1]*f), (int) (a.vec[2]*f)));
    }
    friend Vec3i operator*(float f, const Vec3i& a) {
        return(Vec3i((int) (a.vec[0]*f), (int) (a.vec[1]*f), (int) (a.vec[2]*f)));
    }
    Vec3i& operator*=(int f) {
        vec[0]*=f; vec[1]*=f; vec[2]*=f; return *this;
    }
    Vec3i& operator=(Vec3i& a) {
        vec[0] = a.vec[0]; vec[1] = a.vec[1]; vec[2] = a.vec[2]; return *this;
    }
    Vec3i& operator=(const Vec3i& a) {
        vec[0] = a.vec[0]; vec[1] = a.vec[1]; vec[2] = a.vec[2]; return *this;
    }
	int operator[](int i) {
		return(vec[i]);
	}

	//const int operator[](const int i) const {
	int operator[](const int i) const {
		return(vec[i]);
	}
    friend int operator>(const Vec3i& a, const Vec3i& b) {
        return((a.vec[0] > b.vec[0]) && (a.vec[1] > b.vec[1]) && (a.vec[2] > b.vec[2]));
    }
    friend int operator<(const Vec3i& a, const Vec3i& b) {
        return((a.vec[0] < b.vec[0]) && (a.vec[1] < b.vec[1]) && (a.vec[2] < b.vec[2]));
    }
    friend int operator==(const Vec3i& a, const Vec3i& b) {
        return((a.vec[0] == b.vec[0]) && (a.vec[1] == b.vec[1]) && (a.vec[2] == b.vec[2]));
    }
    friend int operator!=(const Vec3i& a, const Vec3i& b) {
        return(!(a == b));
    }
	void clampMinTo(const Vec3i& v);
	void clampMaxTo(const Vec3i& v);
	Vec3i& max(Vec3i& v) {
		for (int i=0; i < 3; i++) {
			if (v[i] > vec[i]) vec[i] = v[i];
		}
		return *this;
	}
	Vec3i& min(Vec3i& v) {
		for (int i=0; i < 3; i++) {
			if (v[i] < vec[i]) vec[i] = v[i];
		}
		return *this;
	}

private:
    int vec[3];

};

#endif

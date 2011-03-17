
/*
 * Written by Dr. G. Erlebacher.
 */

#ifndef _CVEC3_H_
#define _CVEC3_H_

#include <math.h>
#include <complex>
#include "Vec3.h"

//class ostream;
#include <iosfwd>

// sizeof(CVec3) = 24

// Complex Vec3 class

typedef std::complex<double> CMPLX;

class CVec3
{

    public:
        inline CVec3()
			{ vec[0] = vec[1] = vec[2] = 0.0; };
        inline CVec3(const CVec3& v) {
            this->vec[0] = v.vec[0];
            this->vec[1] = v.vec[1];
            this->vec[2] = v.vec[2];
        }
        inline CVec3(int x, int y=0, int z=0)
    		{ CVec3((float)x, (float)y, (float)z); }
        inline CVec3(float x, float y=0.f, float z=0.f)
			{ vec[0] = CMPLX(x,0.); vec[1] = CMPLX(y,0.); vec[2] = CMPLX(z,0.); }
        inline CVec3(double x, double y=0.f, double z=0.f)
			{ vec[0] = CMPLX(x,0.); vec[1] = CMPLX(y,0.); vec[2] = CMPLX(z,0.); }
        inline CVec3(CMPLX x, CMPLX y=CMPLX(0.,0.), CMPLX z=CMPLX(0.,0.))
			{ vec[0] = x; vec[1] = y; vec[2] = z; }
        /*
         * allocated in calling routine.
         */
		/// Memory is controlled by CVec3
        inline CVec3(float* pt)
			{ 
				vec[0] = CMPLX(pt[0],0.); 
				vec[1] = CMPLX(pt[1],0.); 
				vec[2] = CMPLX(pt[2],0.); 
			}
		/// Memory is controlled by CVec3 (since CVec3 is float, and argument is CMPLX)
        inline CVec3(CMPLX* pt)
			{ vec[0] = pt[0]; vec[1] = pt[1]; vec[2] = pt[2]; }
        inline CVec3(const CMPLX* pt)
			{ vec[0] = pt[0]; vec[1] = pt[1]; vec[2] = pt[2]; }

        ~CVec3() {};
        CVec3(CVec3&);
	 	CVec3(Vec3&);
        /*
         * memory allocated in class
         */
        CMPLX* getVec();
        void getVec(CMPLX* x, CMPLX* y, CMPLX* z);
        void setValue(float x, float y, float z=0.);
        void setValue(float x);
        void setValue(CVec3& v);
        void setValue(float* val);
        void normalize(float scale=1.0);
        double magnitude();
        double magnitude() const;
		CMPLX square() {
			return (vec[0]*vec[0]+vec[1]*vec[1]+vec[2]*vec[2]);
		}
		const CMPLX square() const {
			return (vec[0]*vec[0]+vec[1]*vec[1]+vec[2]*vec[2]);
		}
        void print(const char *msg=0) const;
        //void printcx(const char *msg=0) const;
        //void print() {print("");}

        inline CMPLX x() {return vec[0];}
        inline CMPLX y() {return vec[1];}
        inline CMPLX z() {return vec[2];}

        inline CMPLX x() const {return vec[0];}
        inline CMPLX y() const {return vec[1];}
        inline CMPLX z() const {return vec[2];}

		CVec3 conj(const CVec3& a) {
			return CVec3(std::conj(vec[0]), std::conj(vec[1]), std::conj(vec[2]));
		}

        CVec3 cross(const CVec3& a, const CVec3& b) {
            return (CVec3(a.vec[1]*b.vec[2]-a.vec[2]*b.vec[1],
                         a.vec[2]*b.vec[0]-a.vec[0]*b.vec[2],
                         a.vec[0]*b.vec[1]-a.vec[1]*b.vec[0]));
        }

        CVec3 cross(const CVec3& b) {
            return (CVec3(this->vec[1]*b.vec[2]-this->vec[2]*b.vec[1],
                         this->vec[2]*b.vec[0]-this->vec[0]*b.vec[2],
                         this->vec[0]*b.vec[1]-this->vec[1]*b.vec[0]));
		}
        /*
         * Overload operators, as needed.
         */
       const CVec3 operator+(const CVec3& a) const {
            return (CVec3(a.vec[0]+vec[0], a.vec[1]+vec[1], a.vec[2]+vec[2]));
	   }
       CVec3 operator+(CVec3& a) {
            return (CVec3(a.vec[0]+vec[0], a.vec[1]+vec[1], a.vec[2]+vec[2]));
	   }
       const CVec3 operator-(const CVec3& a) const {
           return (CVec3(vec[0]-a.vec[0], vec[1]-a.vec[1], vec[2]-a.vec[2]));
       }
       CVec3 operator-(CVec3& a) {
           return (CVec3(vec[0]-a.vec[0], vec[1]-a.vec[1], vec[2]-a.vec[2]));
       }
		#if 0
    	bool operator<(const CVec3& b) const {
        	return((vec[0] < b.vec[0]) && (vec[1] < b.vec[1]) && (vec[2] < b.vec[2]));
    	}
    	bool operator<=(const CVec3& b) const {
        	return((vec[0] <= b.vec[0]) && (vec[1] <= b.vec[1]) && (vec[2] <= b.vec[2]));
    	}
    	bool operator>(const CVec3& b) const {
        	return((vec[0] > b.vec[0]) && (vec[1] > b.vec[1]) && (vec[2] > b.vec[2]));
    	}
    	bool operator>=(const CVec3& b) const {
        	return((vec[0] >= b.vec[0]) && (vec[1] >= b.vec[1]) && (vec[2] >= b.vec[2]));
    	}
		#endif
        friend CVec3 operator*(const CVec3& a, CMPLX f) {
            return (CVec3(a.vec[0]*f, a.vec[1]*f, a.vec[2]*f));
        }
        friend CVec3 operator*(CMPLX f, const CVec3& a) {
            return (CVec3(a.vec[0]*f, a.vec[1]*f, a.vec[2]*f));
        }
	   // GENERATES WARNING since a temporary is returned, and I might lose it. 
       //const CVec3& operator/(const CVec3& a) const {
	   // SOME COMPILERS MIGHT OPTIMIZE THIS TO AVOID COPIES!
       const CVec3 operator/(const CVec3& a) const {
           return (CVec3(vec[0]/a.vec[0], vec[1]/a.vec[1], vec[2]/a.vec[2]));
       }
	   // NOT ALLOWED TO RETURN NON-CONSTANT BECAUSE A TEMPORARY SHOULD NEVER BE CHANGED!
       CVec3 operator/(const CVec3& a) {
           return (CVec3(vec[0]/a.vec[0], vec[1]/a.vec[1], vec[2]/a.vec[2]));
       }
        /*
         * Addition: Brian M. Bouta
         */
        CMPLX operator*(const CVec3& a) const {
            return (a.vec[0]*vec[0] + a.vec[1]*vec[1] + a.vec[2]*vec[2]);
        }
        /*
         * Addition: Brian M. Bouta
         * The bivector formed from the outer product of two vectors
         * is treated as a vector, i.e., vec[0] = e1 ^ e2,
         * vec[1] = e2 ^ e3, vec[2] = e3 ^ e1.
         */
        CVec3 operator^(const CVec3& b) {
            return (CVec3(vec[0]*b.vec[1]-vec[1]*b.vec[0],
                         vec[1]*b.vec[2]-vec[2]*b.vec[1],
                         vec[2]*b.vec[0]-vec[0]*b.vec[2]));
        }
        /*
         * Addition: Brian M. Bouta
         */
        CVec3& operator+=(const CVec3& a)
        {
            vec[0] += a.vec[0];
            vec[1] += a.vec[1];
            vec[2] += a.vec[2];
            return *this;
        }


        CVec3& operator*=(CMPLX f)
        {
            vec[0] *= f;
            vec[1] *= f;
            vec[2] *= f;
            return *this;
        }
        CVec3& operator=(const CVec3& a) 
        {
            vec[0] = a.vec[0];
            vec[1] = a.vec[1];
            vec[2] = a.vec[2];
            return *this;
        }
		const CMPLX& operator[](const int i) const {
			return(vec[i]);
		}
		CMPLX operator()(int i) { // 1/5/08
			return(vec[i]);
		}
		CMPLX& operator[](int i) {
			return(vec[i]);
		}

		//friend CVec3 operator-(const CVec3&, const CVec3&);
		//friend CVec3 operator+(const CVec3&, const CVec3&);

	friend double cosine(CVec3& a, CVec3& b) {
    	double am = a.magnitude();
    	double bm = b.magnitude();
    	if ((fabs(am) < 1.e-7) || (fabs(bm) < 1.e-7)) {
        	return 1.0;
    	}

    	return scalprod(a,b)/(am*bm);
	}

	friend double scalprod(CVec3& a, CVec3& b)
	{
		double s0 = real(a[0])*real(b[0]) + imag(a[0])*imag(b[0]);
		double s1 = real(a[1])*real(b[1]) + imag(a[1])*imag(b[1]);
		double s2 = real(a[2])*real(b[2]) + imag(a[2])*imag(b[2]);
		return s0+s1+s2;
	}

	friend double scalprod(const CVec3& a, const CVec3& b) 
	{
		double s0 = real(a[0])*real(b[0]) + imag(a[0])*imag(b[0]);
		double s1 = real(a[1])*real(b[1]) + imag(a[1])*imag(b[1]);
		double s2 = real(a[2])*real(b[2]) + imag(a[2])*imag(b[2]);
		return s0+s1+s2;
	}

	int isColinear(CVec3& a, CVec3& b)
	{
    	CMPLX am = a.magnitude();
    	CMPLX bm = b.magnitude();
    	if ((abs(am) < 1.e-7) || (abs(bm) < 1.e-7)) {
        	return 0;
    	}
	
    	if (abs((a*b)/(am*bm)-1.0) < 1.e-5) {
        	return 1;
    	} else {
        	return 0; // not collinear
   		}
	}
	int isZero(float tolerance)
	{
		//(vec[0]*vec[0]+vec[1]*vec[1]+vec[2]*vec[2]) 
		if (scalprod(*this, *this)  < (tolerance*tolerance)) {
			return 1;
		} else {
			return 0;
		}
	}
//----------------------------------------------------------------------

    public:
        CMPLX vec[3];
};


#endif


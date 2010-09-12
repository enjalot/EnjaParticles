
// extreal.h ----------------------------------------------------------


#ifndef EXTREAL_LIB
#define EXTREAL_LIB 0

#include <ostream>

#ifdef use_namespace
namespace NEWRAN { using namespace RBD_COMMON; }
namespace RBD_LIBRARIES { using namespace NEWRAN; }
namespace NEWRAN {
#endif

/************************ extended real class ***************************/

enum EXT_REAL_CODE
   { Finite, PlusInfinity, MinusInfinity, Indefinite, Missing };

class ExtReal
{
   Real value;
   EXT_REAL_CODE c;
public:
   ExtReal operator+(const ExtReal&) const;
   ExtReal operator-(const ExtReal&) const;
   ExtReal operator*(const ExtReal&) const;
   ExtReal operator-() const;
   friend ostream& operator<<( ostream&, const ExtReal& );
   ExtReal(Real v) { c=Finite; value=v; }
   ExtReal(const EXT_REAL_CODE& cx) { c = cx; }
   ExtReal() { c = Missing; }
   Real Value() const { return value; }
   bool IsReal() const { return c==Finite; }
   EXT_REAL_CODE Code() const { return c; }
};

#ifdef use_namespace
}
#endif


#endif

// body file: extreal.cpp




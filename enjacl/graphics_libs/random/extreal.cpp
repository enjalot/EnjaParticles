
// extreal.cpp ----------------------------------------------------------

#define WANT_STREAM
#include "include.h"
#include "boolean.h"
#include "extreal.h"

#ifdef use_namespace
namespace NEWRAN {
#endif

ExtReal ExtReal::operator+(const ExtReal& er) const
{
   if (c==Finite && er.c==Finite) return ExtReal(value+er.value);
   if (c==Missing || er.c==Missing) return ExtReal(Missing);
   if (c==Indefinite || er.c==Indefinite) return ExtReal(Indefinite);
   if (c==PlusInfinity)
   {
      if (er.c==MinusInfinity) return ExtReal(Indefinite);
      return *this;
   }
   if (c==MinusInfinity)
   {
      if (er.c==PlusInfinity) return ExtReal(Indefinite);
      return *this;
   }
   return er;
}

ExtReal ExtReal::operator-(const ExtReal& er) const
{
   if (c==Finite && er.c==Finite) return ExtReal(value-er.value);
   if (c==Missing || er.c==Missing) return ExtReal(Missing);
   if (c==Indefinite || er.c==Indefinite) return ExtReal(Indefinite);
   if (c==PlusInfinity)
   {
      if (er.c==PlusInfinity) return ExtReal(Indefinite);
      return *this;
   }
   if (c==MinusInfinity)
   {
      if (er.c==MinusInfinity) return ExtReal(Indefinite);
      return *this;
   }
   return er;
}

ExtReal ExtReal::operator*(const ExtReal& er) const
{
   if (c==Finite && er.c==Finite) return ExtReal(value*er.value);
   if (c==Missing || er.c==Missing) return ExtReal(Missing);
   if (c==Indefinite || er.c==Indefinite) return ExtReal(Indefinite);
   if (c==Finite)
   {
      if (value==0.0) return ExtReal(Indefinite);
      if (value>0.0) return er;
      return (-er);
   }
   if (er.c==Finite)
   {
      if (er.value==0.0) return ExtReal(Indefinite);
      if (er.value>0.0) return *this;
      return -(*this);
   }
   if (c==PlusInfinity) return er;
   return (-er);
}

ExtReal ExtReal::operator-() const
{
   switch (c)
   {
      case Finite:        return ExtReal(-value);
      case PlusInfinity:  return ExtReal(MinusInfinity);
      case MinusInfinity: return ExtReal(PlusInfinity);
      case Indefinite:    return ExtReal(Indefinite);
      case Missing:       return ExtReal(Missing);
   }
   return 0.0;
}

ostream& operator<<(ostream& os, const ExtReal& er)
{
   switch (er.c)
   {
      case Finite:        os << er.value;         break;
      case PlusInfinity:  os << "plus-infinity";  break;
      case MinusInfinity: os << "minus-infinity"; break;
      case Indefinite:    os << "indefinite";     break;
      case Missing:       os << "missing";        break;
   }
   return os;
}

#ifdef use_namespace
}
#endif


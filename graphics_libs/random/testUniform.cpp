#define WANT_STREAM

#include <stdio.h>
#include "newran.h"


int main(char** argv, int argc)
{
      Uniform u;
	  Random::Set(0.3); // initial seed in [0,1]
	  printf("u.next = %31.24f\n", u.Next());
	  printf("u.next = %f\n", u.Next());

   return 0;
}

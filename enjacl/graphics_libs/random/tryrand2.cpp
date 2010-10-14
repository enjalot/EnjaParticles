#define WANT_STREAM
#define WANT_TIME

#include "include.h"
#include "newran.h"
#include "tryrand.h"

//#ifdef use_namespace
//using namespace NEWRAN;
//#endif


void test2(int n)
{
   {
      Uniform u;
      SumRandom sr1 = -u;
      SumRandom sr2 = 5.0-u;
      SumRandom sr3 = 5.0-2*u;
      MixedRandom sr4 = u(0.5) + (-u)(0.5);
      Histogram(&sr1,n);
      cout << "Mean and variance should be -0.5 and 0.083333" << endl;
      Histogram(&sr2,n);
      cout << "Mean and variance should be 4.5 and 0.083333" << endl;
      Histogram(&sr3,n);
      cout << "Mean and variance should be 4.0 and 0.33333" << endl;
      Histogram(&sr4,n);
      cout << "Mean and variance should be 0.0 and 0.33333" << endl;
   }


   {
      Uniform u;
      SumRandom sr1 = u*u;
      SumRandom sr2 = (u-0.5)*(u-0.5);
      Histogram(&sr1,n);
      cout << "Mean and variance should be 0.25 and 0.048611" << endl;
      Histogram(&sr2,n);
      cout << "Mean and variance should be 0.0 and 0.006944" << endl;
   }
   {
      static Real probs[]={.4,.2,.4};
      DiscreteGen discrete(3,probs); Normal nn;
      SumRandom sr=discrete+(nn*0.25)(2)+10.0;
      Histogram(&discrete,n);
      Histogram(&sr,n);
   }
   {
      static Real probs[]={.2,.8};
      Random* rv[2];
      Normal nn; SumRandom snn=nn*10.0;
      rv[0]=&snn; rv[1]=&nn;
      MixedRandom mr(2,probs,rv);
      MixedRandom mr2=snn(.2)+nn(.8);
      Histogram(&mr2,n);
      Histogram(&mr,n);
   }

   {
      Normal nn; Constant c1(0.0); Constant c2(10.0);
      MixedRandom mr=c1(0.25)+(nn+5.0)(0.5)+c2(0.25);
      Histogram(&mr,n);
   }
   {
      Cauchy cy; Normal nn; SumRandom sr = cy*.01+nn+2.0;
      MixedRandom mr=sr(0.1)+nn(0.9);
      Histogram(&sr,n);
      Histogram(&mr,n);
   }
   {
      Constant c0(0.0); Constant c1(1.0); Constant c2(2.0);
      Constant c3(3.0); Constant c4(4.0); Constant c5(5.0);
      MixedRandom mr=c0(.1)+c1(.2)+c2(.2)+c3(.2)+c4(.2)+c5(.1);
      Histogram(&mr,n);
   }
   {
      Uniform u; Normal nl;
      MixedRandom m=( u(3)-1.5 )(0.5)+( nl*0.5+10.0 )(0.5);
      Histogram(&m,n);
   }
   {
      Real prob[] = { .25, .25, .25, .25 };
      Real val[] = { 3, 1.5, 1, 0.75 };
      DiscreteGen X(4, prob, val);
      SumRandom Y = 1/X;
      Histogram(&Y,n);  // mean should be 0.83333, variance should be 0.13889
      cout << "Mean and variance should be 0.83333 and 0.13889" << endl;
      Uniform U;
      SumRandom Z = U/X;
      Histogram(&Z,n);  // mean should be 0.41667, variance should be 0.10417
      cout << "Mean and variance should be 0.41667 and 0.10417" << endl;
   }
   {
      int M = 5, N = 9;
      ChiSq Num(M); ChiSq Den(N);
      SumRandom F = (double)N/(double)M * Num / Den;
      Histogram(&F,n);
      cout << "Mean and variance should be " << N / (double)(N-2)
         << " and "
         << 2 * N * N * (M+N-2) / (double)(M * (N-2) * (N-2) * (N-4))
         << endl;
   }


}

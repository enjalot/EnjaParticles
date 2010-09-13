#define WANT_STREAM
#define WANT_MATH
#define WANT_TIME

#include "include.h"
#include "newran.h"
#include "tryrand.h"

#ifdef use_namespace
using namespace NEWRAN;
#endif



Real phi(Real x)                          // normal density
{ return (fabs(x)>8.0) ? 0 : 0.398942280 * exp(-x*x / 2); }

Real NORMAL10(Real x) { return 0.5*phi(0.5*(x-10.0)); }

Real UNIF(Real x) { return (x>=0.0 && x<=1.0) ? 1.0 : 0.0; }

Real TRIANG(Real x) { return (x>1.0) ? 0.0 : 1.0-x; }



void test1(int n)
{
   Normal nn;
   Uniform uniform;
   cout << "Print 20 N(0,1) random numbers - should be the same as in sample output" << endl;
   { for (int i=0; i<20; i++) cout << nn.Next() << "\n" << flush; }
   cout << endl;
   cout << "Print histograms of data from a variety distributions" << endl;
   cout << "Histograms should be close to those in sample output" << endl;
   cout << "s. mean and s. var should be close to p. mean and s. mean" << endl << endl;

   { Constant c(5.0);                         Histogram(&c, n); }
   { Uniform u;                               Histogram(&u, n); }
   { SumRandom sr=uniform(3)-1.5;             Histogram(&sr, n); }
   { SumRandom sr=uniform-uniform;            Histogram(&sr, n); }
   { Normal normal;                           Histogram(&normal, n); }
   { Cauchy cauchy;                           Histogram(&cauchy, n); }
   { AsymGenX normal10(NORMAL10, 10.0);       Histogram(&normal10, n); }
   cout << "Mean and variance should be 10.0 and 4.0" << endl;
   { AsymGenX uniform2(UNIF,0.5);             Histogram(&uniform2, n); }
   cout << "Mean and variance should be 0.5 and 0.083333" << endl;
   { SymGenX triang(TRIANG);                  Histogram(&triang, n); }
   cout << "Mean and variance should be 0 and 0.16667" << endl;
   { Poisson p(0.25);                         Histogram(&p, n); }
   { Poisson p(10.0);                         Histogram(&p, n); }
   { Poisson p(16.0);                         Histogram(&p, n); }
   { Binomial b(18,0.3);                      Histogram(&b, n); }
   { Binomial b(19,0.3);                      Histogram(&b, n); }
   { Binomial b(20,0.3);                      Histogram(&b, n); }
   { Binomial b(58,0.3);                      Histogram(&b, n); }
   { Binomial b(59,0.3);                      Histogram(&b, n); }
   { Binomial b(60,0.3);                      Histogram(&b, n); }
   { Binomial b(18,0.05);                     Histogram(&b, n); }
   { Binomial b(19,0.05);                     Histogram(&b, n); }
   { Binomial b(20,0.05);                     Histogram(&b, n); }
   { Binomial b(98,0.01);                     Histogram(&b, n); }
   { Binomial b(99,0.01);                     Histogram(&b, n); }
   { Binomial b(100,0.01);                    Histogram(&b, n); }
   { Binomial b(18,0.95);                     Histogram(&b, n); }
   { Binomial b(19,0.95);                     Histogram(&b, n); }
   { Binomial b(20,0.95);                     Histogram(&b, n); }
   { Binomial b(98,0.99);                     Histogram(&b, n); }
   { Binomial b(99,0.99);                     Histogram(&b, n); }
   { Binomial b(100,0.99);                    Histogram(&b, n); }
   { NegativeBinomial nb(100,6.0);            Histogram(&nb, n); }
   { NegativeBinomial nb(11,9.0);             Histogram(&nb, n); }
   { NegativeBinomial nb(11,1.9);             Histogram(&nb, n); }
   { NegativeBinomial nb(11,0.10);            Histogram(&nb, n); }
   { NegativeBinomial nb(1.5,1.9);            Histogram(&nb, n); }
   { NegativeBinomial nb(1.0,1.9);            Histogram(&nb, n); }
   { NegativeBinomial nb(0.3,19);             Histogram(&nb, n); }
   { NegativeBinomial nb(0.3,1.9);            Histogram(&nb, n); }
   { NegativeBinomial nb(0.3,0.05);           Histogram(&nb, n); }
   { NegativeBinomial nb(100.8,0.18);         Histogram(&nb, n); }
   { ChiSq c(1,2.0);                          Histogram(&c, n); }
   { ChiSq c(2,2.0);                          Histogram(&c, n); }
   { ChiSq c(3,2.0);                          Histogram(&c, n); }
   { ChiSq c(4,2.0);                          Histogram(&c, n); }
   { ChiSq c(1    );                          Histogram(&c, n); }
   { ChiSq c(2    );                          Histogram(&c, n); }
   { ChiSq c(3    );                          Histogram(&c, n); }
   { ChiSq c(4    );                          Histogram(&c, n); }
   { Gamma g1(1.0);                           Histogram(&g1, n); }
   { Gamma g2(0.5);                           Histogram(&g2, n); }
   { Gamma g3(1.01);                          Histogram(&g3, n); }
   { Gamma g4(2.0);                           Histogram(&g4, n); }
   { Pareto p1(0.5);                          Histogram(&p1, n); }
   { Pareto p2(1.5);                          Histogram(&p2, n); }
   { Pareto p3(2.5);                          Histogram(&p3, n); }
   { Pareto p4(4.5);                          Histogram(&p4, n); }
   Real probs[]={.1,.3,.05,.11,.05,.04,.05,.05,.1,.15};
   Real val[]={2,3,4,6,8,12,16,24,32,48};
   { DiscreteGen discrete(10,probs);          Histogram(&discrete, n); }
   { DiscreteGen discrete(10,probs,val);      Histogram(&discrete, n); }
}

#define WANT_STREAM
#define WANT_TIME

#include "include.h"
#include "newran.h"
#include "tryrand.h"

#ifdef use_namespace
using namespace NEWRAN;
#endif


// ****** simple tests of Poisson, binomial and log normal random number generators ******

void TestVariPoisson(Real mu, int N)
{
   VariPoisson VP;                  // Poisson RNG
   Real sum = 0.0, sumsq = 0.0;
   for (int i = 1; i <= N; ++i)
      { int x = VP.iNext(mu); Real d = x - mu; sum += d; sumsq += d * d; }
   cout
      << " " << setw(10) << setprecision(5) << mu
      << " " << setw(15) << setprecision(10) << mu + sum / N
      << " " << setw(15) << setprecision(10) << (sumsq - sum * sum / N) / (N - 1) << endl;
}

void TestVariBinomial(int n, Real p, int N)
{
   VariBinomial VB;                // Binomial RNG
   Real sum = 0.0, sumsq = 0.0;
   Real mu = n * p;
   for (int i = 1; i <= N; ++i)
      { int x = VB.iNext(n, p); Real d = x - mu; sum += d; sumsq += d * d; }
   cout
      << " " << setw(10) << setprecision(5) << n
      << " " << setw(10) << setprecision(5) << p
      << " " << setw(10) << setprecision(5) << mu
      << " " << setw(10) << setprecision(5) << mu * (1.0 - p)
      << " " << setw(15) << setprecision(10) << mu + sum / N
      << " " << setw(15) << setprecision(10) << (sumsq - sum * sum / N) / (N - 1) << endl;
}

void TestVariLogNormal(double mean, double sd, int N)
{
   VariLogNormal VLN;
   Real sum = 0.0, sumsq = 0.0;
   for (int i = 1; i <= N; ++i)
   {
      Real x = VLN.Next(mean, sd);
      Real d = x - mean; sum += d; sumsq += d * d;
   }
   cout
      << " " << setw(10) << setprecision(5) << mean
      << " " << setw(10) << setprecision(5) << sd
      << " " << setw(10) << setprecision(5) << sd * sd
      << " " << setw(15) << setprecision(10) << mean + sum / N
      << " " << setw(15) << setprecision(10) << (sumsq - sum * sum / N) / (N - 1) << endl;
}

void test5(int N)
{
   cout << endl << endl;

   cout << "Testing VariPoisson" << endl;
   cout << " - columns 2 (mean) and 3 (variance) should be close to column 1" << endl;

   TestVariPoisson(0.25, N);
   TestVariPoisson(1, N);
   TestVariPoisson(4, N);
   TestVariPoisson(10, N);
   TestVariPoisson(20, N);
   TestVariPoisson(30, N);
   TestVariPoisson(39.5, N);
   TestVariPoisson(40, N);
   TestVariPoisson(50, N);
   TestVariPoisson(59.5, N);
   TestVariPoisson(60, N);
   TestVariPoisson(60.5, N);
   TestVariPoisson(99.5, N);
   TestVariPoisson(100, N);
   TestVariPoisson(100.5, N);
   TestVariPoisson(199.5, N);
   TestVariPoisson(200, N);
   TestVariPoisson(200.5, N);
   TestVariPoisson(299.5, N);
   TestVariPoisson(300, N);
   TestVariPoisson(300.5, N);
   TestVariPoisson(399.5, N);
   TestVariPoisson(400, N);
   TestVariPoisson(400.5, N);
   TestVariPoisson(500, N);
   TestVariPoisson(10000, N);
   TestVariPoisson(10000.5, N);
   TestVariPoisson(100000, N);
   TestVariPoisson(100000.5, N);
   TestVariPoisson(10000000, N);
   TestVariPoisson(10000000.5, N);

   cout << endl;
   cout << "Testing VariBinomial" << endl;
   cout << " - columns 5 (mean) and 6 (variance) should be close to columns 1 and 4" << endl;

   TestVariBinomial(1, 0.2, N);
   TestVariBinomial(2, 0.1, N);
   TestVariBinomial(5, 0.35, N);
   TestVariBinomial(20, 0.2, N);
   TestVariBinomial(50, 0.5, N);
   TestVariBinomial(100, 0.4, N);
   TestVariBinomial(200, 0.2, N);
   TestVariBinomial(500, 0.3, N);
   TestVariBinomial(1000, 0.19, N);
   TestVariBinomial(1000, 0.21, N);
   TestVariBinomial(10000, 0.4, N);
   TestVariBinomial(1000000, 0.1, N);
   TestVariBinomial(1000000, 0.5, N);
   TestVariBinomial(1, 0.7, N);
   TestVariBinomial(2, 0.6, N);
   TestVariBinomial(5, 0.9, N);
   TestVariBinomial(20, 0.55, N);
   TestVariBinomial(50, 0.9, N);
   TestVariBinomial(100, 0.99, N);
   TestVariBinomial(200, 0.51, N);
   TestVariBinomial(500, 0.7, N);
   TestVariBinomial(1000, 0.81, N);
   TestVariBinomial(1000, 0.79, N);
   TestVariBinomial(10000, 0.9, N);
   TestVariBinomial(1000000, 0.55, N);
   TestVariBinomial(1000000, 0.7, N);

   cout << endl;
   cout << "Testing VariLogNormal" << endl;
   cout << " - columns 4 (mean) and 5 (variance) should be close to columns 1 and 3" << endl;

   TestVariLogNormal(0.25, 0.5, N);
   TestVariLogNormal( 0.5, 1.5, N);
   TestVariLogNormal( 1.5, 2.5, N);
   TestVariLogNormal( 2.0, 1.0, N);

   cout << endl;
}



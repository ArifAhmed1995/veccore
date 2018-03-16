#include "VecCore/Timer.h"

#include "VecRng/MRG32k3a.h"

#include "RngTest.h"

namespace vecrng {

// Scalar

double ScalarMRG32k3a(int nsample, double& result)
{
  // Scalar MRG32k3a
  static vecRng::cxx::MRG32k3a<ScalarBackend> rng;
  rng.Initialize();

  static Timer<nanoseconds> timer;
  double elapsedTime = 0.;

  double sum = 0;
  double norm = 0;

  timer.Start();

  for (int i = 0; i < nsample ; ++i) {
    sum += rng.Gauss<ScalarBackend>(0.0,1.0);
  }

  elapsedTime = timer.Elapsed();
  result = sum;

  return elapsedTime;
}

// Vector

double VectorMRG32k3a(int nsample, double& result)
{
  // Vector MRG32k3a
  using Double_v = typename VectorBackend::Double_v;
  int vsize = VectorSize<Double_v>();

  vecRng::cxx::MRG32k3a<VectorBackend> rng;
  rng.Initialize();

  static Timer<nanoseconds> timer;
  double elapsedTime = 0.;

  Double_v sum = 0.;

  timer.Start();

  for (int i = 0; i < nsample/vsize ; ++i) {
    sum += rng.Gauss<VectorBackend>(0.0,1.0);
  }

  elapsedTime = timer.Elapsed();
  for (int i = 0; i < vsize ; ++i) result += sum[i];

  return elapsedTime;
}

} // end namespace vecrng

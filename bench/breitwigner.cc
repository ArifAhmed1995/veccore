#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>

#include "VecCore/Timer.h"
#include <VecCore/VecCore>

#include <umesimd/UMESimd.h>
using namespace vecCore;

static constexpr size_t kNruns = 10;
static constexpr size_t kN = (32 * 1024 * 1024);

template <typename T>
T BreitWigner(T x, T mean, T gamma)
{
  return T(0.15915494309189535f) * T(gamma / ( (x-mean) * (x-mean) +
                                 gamma * gamma / 4));
}

template <class Backend>
void BreitWigner(typename Backend::Float_v const &x,
                 typename Backend::Float_v const &mean,
                 typename Backend::Float_v const &gamma,
                 typename Backend::Float_v &out)
{
  using Float_v = typename Backend::Float_v;
  Float_v dev = (x - mean) / gamma;
  Mask_v<Float_v> mask_dev = !(dev < -1.e19|| dev > 1.e19);

  Float_v num = Float_v(0.15915494309189535f) / gamma;
  Float_v denom = dev * dev + 1 / Float_v(4.0f);
  MaskedAssign<Float_v>(out, mask_dev, num / denom);
}

//Timings for vectorized implementation.

template <class Backend>
VECCORE_FORCE_NOINLINE
void TestBreitWignerVectorized(const float *__restrict__ x,
                     const float *__restrict__ mean,
                     const float *__restrict__ gamma,
                      float *__restrict__ out, size_t kN,  const char *name)
{
  using Float_v = typename Backend::Float_v;
  Timer<milliseconds> timer;
  double t[kNruns], time_mean = 0.0, sigma = 0.0;
  for (size_t n = 0; n < kNruns; n++) {
    timer.Start();
    for (size_t i = 0; i < kN; i += VectorSize<Float_v>())
        BreitWigner<Backend>((Float_v &)(x[i]), (Float_v &)(mean[i]),
                             (Float_v &)(gamma[i]), (Float_v &)(out[i]));
    t[n] = timer.Elapsed();
  }

  for (size_t n = 0; n < kNruns; n++)
    time_mean += t[n];

  time_mean = time_mean / kNruns;

  for (size_t n = 0; n < kNruns; n++)
    sigma += std::pow(t[n] - time_mean, 2.0);

  sigma = std::sqrt(sigma);
  
  #ifdef VERBOSE
  size_t index = (size_t)((kN - 100) * drand48());
  for (size_t i = index; i < index + 10; i++)
    printf("%d: x = % 8.3f, mean = % 8.3f, gamma = % 8.3f,\
              breit-wigner result = % 8.3f\n", x[i], mean[i],\
              gamma[i], out[i]);
#endif
  printf("%20s %8.1lf %7.1lf\n", name, time_mean, sigma);
}

//Timings for scalar implementation
VECCORE_FORCE_NOINLINE
void TestBreitWignerNaiveScalar(const float *__restrict__ x,
                     const float *__restrict__ mean,
                     const float *__restrict__ gamma,
                      float *__restrict__ out, size_t kN, const char *name)
{
  Timer<milliseconds> timer;
  double t[kNruns], time_mean = 0.0, sigma = 0.0;
  for (size_t n = 0; n < kNruns; n++) {
    timer.Start();
    for (size_t i = 0; i < kN; i ++)
        out[i] = BreitWigner(x[i], mean[i], gamma[i]);
    t[n] = timer.Elapsed();
  }

  for (size_t n = 0; n < kNruns; n++)
    time_mean += t[n];

  time_mean = time_mean / kNruns;
  
  for (size_t n = 0; n < kNruns; n++)
    sigma += std::pow(t[n] - time_mean, 2.0);

  sigma = std::sqrt(sigma);

#ifdef VERBOSE
  size_t index = (size_t)((kN - 100) * drand48());
  for (size_t i = index; i < index + 10; i++)
    printf("%d: x = % 8.3f, mean = % 8.3f,\
    gamma = % 8.3f, result = %d\n", i, x[i], mean[i], gamma[i], out[i]);
#endif
  printf("%20s %8.1lf %7.1lf\n", name, time_mean, sigma);
}

int main()
{
    float *x, *mean, *gamma, *out;

    x     = (float *)AlignedAlloc(VECCORE_SIMD_ALIGN, kN * sizeof(float));
    mean     = (float *)AlignedAlloc(VECCORE_SIMD_ALIGN, kN * sizeof(float));
    gamma  = (float *)AlignedAlloc(VECCORE_SIMD_ALIGN, kN * sizeof(float));
    out    = (float *)AlignedAlloc(VECCORE_SIMD_ALIGN, kN * sizeof(float));

    srand48(time(NULL));

    for (size_t i = 0; i < kN; i++) {
      x[i]     = 10.0 * (drand48() - 0.5);
      mean[i]     = 10.0 * (drand48() - 0.5);
      gamma[i]     = 50.0 * (drand48() - 0.5);
      out[i]    = 0.0;
    }

    printf("             Backend     Mean / Sigma (ms)\n");
    printf("------------------------------------------\n");    
    
    TestBreitWignerNaiveScalar(x, mean, gamma, out, kN, "Naive Scalar");
    
    #ifdef VECCORE_ENABLE_VC
      TestBreitWignerVectorized<backend::VcScalar>(x, mean, gamma, out, kN, "VcScalar");
      TestBreitWignerVectorized<backend::VcVector>(x, mean, gamma, out, kN, "VcVector");
      TestBreitWignerVectorized<backend::VcSimdArray<8>>(x, mean, gamma, out, kN, "VcSimdArray<8>");
      TestBreitWignerVectorized<backend::VcSimdArray<16>>(x, mean, gamma, out, kN, "VcSimdArray<16>");
      TestBreitWignerVectorized<backend::VcSimdArray<32>>(x, mean, gamma, out, kN, "VcSimdArray<32>");
    #endif
    
    #ifdef VECCORE_ENABLE_UMESIMD
      TestQuadSolve<backend::UMESimd>(x, mean, gamma, out, kN, "UME::SIMD");
      TestQuadSolve<backend::UMESimdArray<8>>(x, mean, gamma, out, kN, "UME::SIMD<8>");
      TestQuadSolve<backend::UMESimdArray<16>>(x, mean, gamma, out, kN, "UME::SIMD<16>");
      TestQuadSolve<backend::UMESimdArray<32>>(x, mean, gamma, out, kN, "UME::SIMD<32>");
    #endif
    
    return 0;
}

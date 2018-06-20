#include <vector>
#include <complex>
#include<mkl.h>

#ifndef CREATEPOTENTIAL_H_INCLUDED
#define CREATEPOTENTIAL_H_INCLUDED

std::vector<double> harmonicPotential (int const n, double const omega, std::vector<double> xVec);
std::vector<double> randomPotential (int const n, double const corLength, double const length, double const potStr, std::complex<double> I, std::vector<double> kVec);
std::vector<double> specklePotential(int const n, double const corLength, double const length, double const potStr, std::complex<double> I, std::vector<double> kVec);
void createHamiltonian(int const n, double dx, std::vector<double> potential, double *hamiltonian);
void CreateAandB(MKL_Complex16 *A, MKL_Complex16 *B, int const n, double const dt, double *H, std::complex<double> I);
void CreateB(MKL_Complex16 *B, int const n, double const dt, double *H, std::complex<double> I);
void psiThomasFermi(MKL_Complex16 *psi,int const n, double const omega,std::vector<double> hTrap);
void psiGaussian(MKL_Complex16 *psi, int const n, double const sigma, std::vector<double> xVec,double dx);
void normalizePsi(MKL_Complex16 *psi,double dx,int const n);
void ExponentiatePotential(MKL_Complex16 *ExpV, std::vector<double> potential, double dt, const int n, std::complex<double> I);
void temporalStepRamp(MKL_Complex16 *psi,MKL_Complex16 *potential, const int n, double factor, double dt, std::complex<double> I);
void apply_kick(MKL_Complex16 *psi, std::vector<double> xAr, const int n, double beta);
void temporalStep(MKL_Complex16 *psi,MKL_Complex16 *potential, const int n);
std::vector<double> slopeGradient (int const n, double maxVal);
#endif // CREATEPOTENTIAL_H_INCLUDED

#include <vector>
#include <complex>
#include<mkl.h>

#ifndef SPARSEMATRIX_H_INCLUDED
#define SPARSEMATRIX_H_INCLUDED

void createHamiltonianSparse(int const n, double dx, std::vector<double> potential, double *hVal, double *hCol, double *hRow);
void CreateASymSparse(MKL_Complex16 *A,int const n, double const dt, double dx, MKL_INT *aCol, MKL_INT *aRow);
void CreateBCSRSparse(MKL_Complex16 *B,int const n, double const dt, double dx, MKL_INT *bCol, MKL_INT *bRow);
void CreateACSRSparse(MKL_Complex16 *A,int const n, double const dt, double dx, MKL_INT *aCol, MKL_INT *aRow);

void CreateASymSparse2(MKL_Complex16 *A,int const n, double const dt, double dx, MKL_INT *aCol, MKL_INT *aRow, std::vector<double> pot);
void CreateBCSRSparse2(MKL_Complex16 *B,int const n, double const dt, double dx, MKL_INT *bCol, MKL_INT *bRow, std::vector<double> pot);

void CreateASymSparse3(MKL_Complex16 *A,int const n, double const dt, double dx, MKL_INT *aCol, MKL_INT *aRow, std::vector<double> pot);
void CreateBCSRSparse3(MKL_Complex16 *B,int const n, double const dt, double dx, MKL_INT *bCol, MKL_INT *bRow, std::vector<double> pot);
#endif // SPARSEMATRIX_H_INCLUDED

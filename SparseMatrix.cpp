#include<iostream>
#include <vector>
#include <complex>
#include<math.h>
#include <stdio.h>
#include <stdlib.h>
#include "mkl.h"

void createHamiltonianSparse(int const n, double dx, std::vector<double> potential, double *hVal, double *hCol, double *hRow){
    double coef = (pow(1.054e-34,2))/(2*87*1.667e-27*pow(dx,2.0));
    hVal[0] = -2 * coef + potential[0];
    hVal[1] = coef;
    hVal[2] = coef;

    int k = 1;
    for(int i = 3;i<2*n-1;i++){
        //Super Diagonal Vals.
        if(i%2==0){
            k++;
            hVal[i] = coef;
            hCol[i] = k;
        }
        //On Diagonal Vals.
        if(i%2==1){
            hVal[i] = -2 * coef + potential[k];
            hCol[i] = k;
        }
    }
    //Row Positioning
    hRow[0] = 0;
    k = 3;
    for(int j=1;j<n;j++){
        hRow[j] = k;
        k = k+2;
    }
    hRow[n]=2*n;
}

void CreateASymSparse(MKL_Complex16 *A,int const n, double const dt, double dx, MKL_INT *aCol, MKL_INT *aRow){
    double coef = (pow(1.054e-34,2))/(2*87*1.667e-27*pow(dx,2.0));

    A[0].real = 1.0;
    //A[0].imag = 0.5*dt*(-2*coef+pot[0])/1.054e-34;
    A[0].imag = dt*(-2*coef)/1.054e-34;
    A[1].real = 0.0;
    A[1].imag = dt*(coef)/1.054e-34;
    A[2].real = 0.0;
    A[2].imag = dt*(coef)/1.054e-34;

    aCol[0] = 0;
    aCol[1] = 1;
    aCol[2] = n-1;
    int k = 1;
    for(int i = 3; i<2*n; i++){
    //Super Diagonal Vals.
        if(i%2==0 ){
        k++;
        A[i].real = 0.0;
        A[i].imag = dt*(coef)/1.054e-34;
        aCol[i] = k;
        }
    //On Diagonal Vals.
        if(i%2==1){
        A[i].real = 1.0;
        //A[i].imag = 0.5*dt*(-2*coef+pot[k])/1.054e-34;
        A[i].imag = dt*(-2*coef)/1.054e-34;
        aCol[i] = k;
        }
    }
    //A[2*n-1].imag = 0.5*dt*(-2*coef+pot[n-1])/1.054e-34;
    A[2*n-1].imag = dt*(-2*coef)/1.054e-34;
    aRow[0] = 0;
    k = 3;
    for(int j=1;j<n;j++){
        aRow[j] = k;
        k = k+2;
    }
    aRow[n]=2*n;
}

void CreateACSRSparse(MKL_Complex16 *A,int const n, double const dt, double dx, MKL_INT *aCol, MKL_INT *aRow){
    double coef = (pow(1.054e-34,2))/(2*87*1.667e-27*pow(dx,2.0));
    A[0].real = 1.0;
    //B[0].imag = -0.5*dt*(-2*coef+pot[0])/1.054e-34;
    A[0].imag = 0.5*dt*(-2*coef)/1.054e-34;
    A[1].real = 0.0;
    A[1].imag = 0.5*dt*(coef)/1.054e-34;
    A[2].real = 0.0;
    A[2].imag = 0.5*dt*(coef)/1.054e-34;

    aCol[0] = 0;
    aCol[1] = 1;
    aCol[2] = n-1;
    int k = 1;
    for(int i=3;i<3*n-3;i=i+3){
        A[i].real = 0.0;
        A[i].imag = 0.5*dt*(coef)/1.054e-34;
        A[i+1].real = 1.0;
        //B[i+1].imag = -0.5*dt*(-2*coef+pot[k])/1.054e-34;
        A[i+1].imag = 0.5*dt*(-2*coef)/1.054e-34;
        A[i+2].real = 0.0;
        A[i+2].imag = 0.5*dt*(coef)/1.054e-34;

        aCol[i] = k-1;
        aCol[i+1] = k;
        aCol[i+2] = k+1;

        k++;
    }

    A[3*n-3].real = 0.0;
    A[3*n-3].imag = 0.5*dt*(coef)/1.054e-34;
    A[3*n-2].real = 0.0;
    A[3*n-2].imag = 0.5*dt*(coef)/1.054e-34;
    A[3*n-1].real = 1.0;
    //B[3*n-1].imag = -0.5*dt*(-2*coef+pot[n-1])/1.054e-34;
    A[3*n-1].imag = 0.5*dt*(-2*coef)/1.054e-34;
    aCol[3*n-3] = 0;
    aCol[3*n-2] = n-2;
    aCol[3*n-1] = n-1;

    for(int j=0;j<n+2;j++){
        aRow[j] = 3*j;
    }
}


void CreateBCSRSparse(MKL_Complex16 *B,int const n, double const dt, double dx, MKL_INT *bCol, MKL_INT *bRow){
    double coef = (pow(1.054e-34,2))/(2*87*1.667e-27*pow(dx,2.0));
    B[0].real = 1.0;
    //B[0].imag = -0.5*dt*(-2*coef+pot[0])/1.054e-34;
    B[0].imag = -dt*(-2*coef)/1.054e-34;
    B[1].real = 0.0;
    B[1].imag = -dt*(coef)/1.054e-34;
    B[2].real = 0.0;
    B[2].imag = -dt*(coef)/1.054e-34;

    bCol[0] = 0;
    bCol[1] = 1;
    bCol[2] = n-1;
    int k = 1;
    for(int i=3;i<3*n-3;i=i+3){
        B[i].real = 0.0;
        B[i].imag = -dt*(coef)/1.054e-34;
        B[i+1].real = 1.0;
        //B[i+1].imag = -0.5*dt*(-2*coef+pot[k])/1.054e-34;
        B[i+1].imag = -dt*(-2*coef)/1.054e-34;
        B[i+2].real = 0.0;
        B[i+2].imag = -dt*(coef)/1.054e-34;

        bCol[i] = k-1;
        bCol[i+1] = k;
        bCol[i+2] = k+1;

        k++;
    }

    B[3*n-3].real = 0.0;
    B[3*n-3].imag = -dt*(coef)/1.054e-34;
    B[3*n-2].real = 0.0;
    B[3*n-2].imag = -dt*(coef)/1.054e-34;
    B[3*n-1].real = 1.0;
    //B[3*n-1].imag = -0.5*dt*(-2*coef+pot[n-1])/1.054e-34;
    B[3*n-1].imag = -dt*(-2*coef)/1.054e-34;
    bCol[3*n-3] = 0;
    bCol[3*n-2] = n-2;
    bCol[3*n-1] = n-1;

    for(int j=0;j<n+2;j++){
        bRow[j] = 3*j;
    }
}

void CreateASymSparse2(MKL_Complex16 *A,int const n, double const dt, double dx, MKL_INT *aCol, MKL_INT *aRow, std::vector<double> pot){
    double coef = 1.054e-34*dt/(2*87*1.667e-27*pow(dx,2.0));
    double dth = dt/1.054e-34;
    A[0].real = 1.0;
    A[0].imag = -(2*coef+dth*pot[0]);
    //A[0].imag = -coef;
    A[1].real = 0.0;
    A[1].imag = coef;
    /*Boundary Factor*/
    A[2].real = 0.0;
    A[2].imag = coef;

    aCol[0] = 0;
    aCol[1] = 1;
    aCol[2] = n-1;
    int k = 1;
    for(int i = 3; i<2*n; i++){
    //Super Diagonal Vals.
        if(i%2==0 ){
        k++;
        A[i].real = 0.0;
        A[i].imag = coef;
        aCol[i] = k;
        }
    //On Diagonal Vals.
        if(i%2==1){
        A[i].real = 1.0;
        A[i].imag = -(2*coef+dth*pot[k]);
        //A[i].imag = -coef;
        aCol[i] = k;
        }
    }
    A[2*n-1].real = 1.0;
    A[2*n-1].imag = -(2*coef+dth*pot[n-1]);
    aRow[0] = 0;
    k = 3;
    for(int j=1;j<n;j++){
        aRow[j] = k;
        k = k+2;
    }
    aRow[n]=2*n;
}

void CreateBCSRSparse2(MKL_Complex16 *B,int const n, double const dt, double dx, MKL_INT *bCol, MKL_INT *bRow, std::vector<double> pot){
    double coef = 1.054e-34*dt/(87*1.667e-27*pow(dx,2.0));
    double dth = dt/1.054e-34;

    /*1st diag*/
    B[0].real = 1.0;
    B[0].imag = (2*coef-dth*pot[0]);
    //B[0].imag = coef;
    /*1st Off Diag*/
    B[1].real = 0.0;
    B[1].imag = -1.0*coef;
    /*Boundary point*/
    B[2].real = 0.0;
    B[2].imag = -1.0*coef;

    bCol[0] = 0;
    bCol[1] = 1;
    bCol[2] = n-1;
    int k = 1;
    for(int i=3;i<3*n-3;i=i+3){

        /*sub-Diag*/
        B[i].real = 0.0;
        B[i].imag = -1.0*coef;
        /*Diagonal term*/
        B[i+1].real = 1.0;
        B[i+1].imag = (2*coef-0.5*dth*pot[k]);
        //B[i+1].imag = coef;
        /*super Diag*/
        B[i+2].real = 0.0;
        B[i+2].imag = -1.0*coef;

        bCol[i] = k-1;
        bCol[i+1] = k;
        bCol[i+2] = k+1;

        k++;
    }
    /*Boundary Point*/
    B[3*n-3].real = 0.0;
    B[3*n-3].imag = -coef;
    /*Last sub-diagonal*/
    B[3*n-2].real = 0.0;
    B[3*n-2].imag = -coef;
    /*Last Diagonal*/
    B[3*n-1].real = 1.0;
    B[3*n-1].imag = (2*coef-0.5*dth*pot[n-1]);
    //B[3*n-1].imag = coef;

    bCol[3*n-3] = 0;
    bCol[3*n-2] = n-2;
    bCol[3*n-1] = n-1;

    for(int j=0;j<n+2;j++){
        bRow[j] = 3*j;
    }
}

void CreateASymSparse3(MKL_Complex16 *A,int const n, double const dt, double dx, MKL_INT *aCol, MKL_INT *aRow, std::vector<double> pot){
    double coef = 1.054e-34*dt/(2.0*87*1.667e-27*pow(dx,2.0));
    double dth = dt/1.054e-34;

    A[0].real = 1.0;
    A[0].imag = (-2.0*coef+dth*pot[0]);
    //A[0].imag = -coef;
    A[1].real = 0.0;
    A[1].imag = coef;
    /*Boundary Factor*/

    aCol[0] = 0;
    aCol[1] = 1;

    int k = 1;
    for(int i = 2; i<(2*n)-2; i++){
    //Super Diagonal Vals.
        if(i%2==1){
        k++;
        A[i].real = 0.0;
        A[i].imag = coef;
        aCol[i] = k;
        }
    //On Diagonal Vals.
        if(i%2==0){
        A[i].real = 1.0;
        A[i].imag = (-2.0*coef+dth*pot[k]);
        //A[i].imag = -coef;
        aCol[i] = k;
        }
    }
    A[2*n-2].real = 1.0;
    A[2*n-2].imag = (-2.0*coef+dth*pot[n-1]);
    aCol[(2*n)-2] = n-1;
    k = 0;
    for(int j=0;j<n;j++){
        aRow[j] = k;
        k = k+2;
    }
    aRow[n]=2*n-1;
}


void CreateBCSRSparse3(MKL_Complex16 *B,int const n, double const dt, double dx, MKL_INT *bCol, MKL_INT *bRow, std::vector<double> pot){
    double coef = 1.054e-34*dt/(2.0*87*1.667e-27*pow(dx,2.0));
    double dth = dt/1.054e-34;

    /*1st diag*/
    B[0].real = 1.0;
    B[0].imag = (2.0*coef-dth*pot[0]);
    //B[0].imag = coef;
    /*1st Super Diag*/
    B[1].real = 0.0;
    B[1].imag = -1.0*coef;
    bCol[0] = 0;
    bCol[1] = 1;

    bRow[0] = 0;

    int k = 1;
    int j = 1;
    for(int i=2;i<(3*n-3);i=i+3){

        /*sub-Diag*/
        B[i].real = 0.0;
        B[i].imag = -1.0*coef;
        bCol[i] = k-1;
        /*Diagonal term*/
        B[i+1].real = 1.0;
        B[i+1].imag = (2.0*coef-dth*pot[k]);
        bCol[i+1] = k;
        /*super Diag*/
        B[i+2].real = 0.0;
        B[i+2].imag = -1.0*coef;
        bCol[i+2] = k+1;

        k++;

        bRow[j] = 2+(j-1)*3;
        j++;
    }
    /*Last sub-diagonal*/
    B[3*n-4].real = 0.0;
    B[3*n-4].imag = -1.0*coef;
    /*Last Diagonal*/
    B[3*n-3].real = 1.0;
    B[3*n-3].imag = (2.0*coef-dth*pot[n-1]);

    bCol[3*n-4] = n-2;
    bCol[3*n-3] = n-1;


    bRow[n-2] = bRow[n-3]+3;
    bRow[n-1] = bRow[n-2]+2;
    bRow[n] = (3*n)-2;
}

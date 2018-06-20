//#inlcude <omp.h>
#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <complex>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "mkl.h"
#include <fstream>
#include "createPotential.h"
#include "sparseMatrix.h"

int main()
{
    int threads = 4;
    mkl_set_num_threads(threads);
/* Defines all constants */
    long long int n = pow(2,11);
    MKL_INT n_mkl = n;
    double const length = 0.0012;
    double const dt = 5e-9;
    int const nSteps = 40000000;
    //100000000;
    int const freeSteps = 4000000;
    int const rampSteps = -1;
    double const omega = 2.0 * M_PI * 50.0;
    double const sigma = 5e-6;
    double const corLength = 0.5e-6;
    double const potStr = 0.01;
    int saveEvery = nSteps/50;
    double beta = 0.1;
    double maxVal = 10*1.3806488e-23*1e-9;
    std::complex<double> I;										//complex i
    I.imag(1);

    std::vector<double> xVec(n,0), kVec(n,0), VRand(n,0), hPot(n,0);
    double dx,dk;

/*Creating Position and Momentum arrays*/
    for(int i=0;i<n;i++){
        xVec[i] = -1.0*(double)length/2.0 + length*i/n;
        kVec[i] = -1.0*(double)(((double)n/2.0)-i)*2.0*M_PI/length;
   }
    dx = xVec[1]-xVec[0];
    dk = kVec[1]-kVec[0];
    std::cout<<xVec[0]<<" "<<xVec[1]<<" "<<dx<<std::endl;
    std::cout<<kVec[0]<<" "<<kVec[1]<<" "<<dk<<std::endl;

    /*Creates Random Potential and Harmonic Trap*/
    VRand = randomPotential(n, corLength, length, potStr, I, kVec);
    //VRand = harmonicPotential(n,omega,xVec);
    //VRand = slopeGradient (n, maxVal);
    //VRand = specklePotential(n, corLength, length, potStr, I, kVec);
    std::ofstream randomOut("Results/RandomPot.txt");
    for(int i=0;i<n;i++){
        randomOut<< VRand[i]<<" ";
    }
    randomOut.close();
    kVec.clear();
    MKL_Complex16 *VRandExp;
    VRandExp = (MKL_Complex16 *)mkl_malloc(n*sizeof(MKL_Complex16),64);
    ExponentiatePotential(VRandExp,VRand,dt,n,I);
    VRand.clear();

    hPot = harmonicPotential(n, omega, xVec);
    //MKL_Complex16 *HPotExp;
    //HPotExp = (MKL_Complex16 *)mkl_malloc(n*sizeof(MKL_Complex16),64);
    //ExponentiatePotential(HPotExp,hPot,dt,n,I);
    std::cout << "Created Potentials." << std::endl;

/*Initializes Psi*/
    MKL_Complex16 *psi;
    psi = (MKL_Complex16 *)mkl_malloc(n*sizeof(MKL_Complex16),64);
    psiThomasFermi(psi, n, omega, hPot);
    //psiGaussian(psi,n,sigma,xVec,dx);
    hPot.clear();
    //apply_kick(psi,xVec,n,beta);
    std::ofstream output("Results/Psi_0.txt");
    for (int i = 0; i < n; ++i)
    {
        output << psi[i].real<< " "<<psi[i].imag<<"\n";
    }
    output.close();
    std::cout<< "Initialized Psi." << std::endl;

    /*Creates A and B finite difference Matrices*/

    MKL_Complex16 *AVal, *BVal;
    AVal = (MKL_Complex16 *)mkl_malloc((2*n)*sizeof(MKL_Complex16),64);
    //AVal = (MKL_Complex16 *)mkl_malloc((3*n)*sizeof(MKL_Complex16),64);
    BVal = (MKL_Complex16 *)mkl_malloc((3*n)*sizeof(MKL_Complex16),64);
    MKL_INT *aCol, *aRow;
    aCol = (MKL_INT *)mkl_malloc((2*n)*sizeof(MKL_INT),64);
    //aCol = (MKL_INT *)mkl_malloc((3*n)*sizeof(MKL_INT),64);
    aRow = (MKL_INT *)mkl_malloc((n+1)*sizeof(MKL_INT),64);
    MKL_INT *bCol, *bRow;
    bCol = (MKL_INT *)mkl_malloc((3*n)*sizeof(MKL_INT),64);
    bRow = (MKL_INT *)mkl_malloc((n+1)*sizeof(MKL_INT),64);

    CreateASymSparse(AVal,n,dt,dx,aCol,aRow);
    //CreateACSRSparse(AVal,n,dt,dx,aCol,aRow);
    CreateBCSRSparse(BVal,n,dt,dx,bCol,bRow);

//    MKL_Complex16 *AVal, *BVal;
//    AVal = (MKL_Complex16 *)mkl_malloc(((2*n)-1)*sizeof(MKL_Complex16),64);
//    BVal = (MKL_Complex16 *)mkl_malloc(((3*n)-2)*sizeof(MKL_Complex16),64);
//
//    MKL_INT *aCol, *aRow;
//    aCol = (MKL_INT *)mkl_malloc(((2*n)-1)*sizeof(MKL_INT),64);
//    aRow = (MKL_INT *)mkl_malloc((n+1)*sizeof(MKL_INT),64);
//
//    MKL_INT *bCol, *bRow;
//    bCol = (MKL_INT *)mkl_malloc(((3*n)-2)*sizeof(MKL_INT),64);
//    bRow = (MKL_INT *)mkl_malloc((n+1)*sizeof(MKL_INT),64);
//
//    CreateASymSparse3(AVal,n,dt,dx,aCol,aRow,VRand);
//    CreateBCSRSparse3(BVal,n,dt,dx,bCol,bRow,VRand);

    std::cout << "Created Finite Difference Matrices" <<std::endl;

/*Does time evolution*/

//    MKL_Complex16 alpha,beta;
//    alpha.real = 1.0;
//    alpha.imag = 0.0;
//    beta.real = 0.0;
//    beta.imag = 0.0;

    MKL_INT *para,*pt,*perm;
    para = (MKL_INT *)mkl_malloc(64*sizeof(MKL_INT),64);
    pt = (MKL_INT *)mkl_malloc(64*sizeof(MKL_INT),64);
    perm = (MKL_INT *)mkl_malloc(n*sizeof(MKL_INT),64);
    for(int i=0;i<64;i++){
        para[i]=0;
        pt[i]=0;
        perm[i]=0;
    }
    MKL_INT mtype = 6;
    //MKL_INT mtype = 13;
    MKL_INT maxfct = 1;
    MKL_INT nnum = 1;
    MKL_INT phase = 0;
    MKL_INT nrhs = 1;
    MKL_INT msglvl = 0;
    MKL_INT err = 0;
    pardisoinit(pt,&mtype,para);

    const char trans = 'N';
    para[1]=1; //Sets Paradiso to parallel mode.
    para[34]=1; //Sets zero-based indexing.
    para[26]=0; //Matrix Check parameter.
    para[23]=1; //Enables phase 22 to use more than 8 cores.

    MKL_Complex16 *psitmp;
    psitmp = (MKL_Complex16 *)mkl_malloc(n*sizeof(MKL_Complex16),64);

    std::cout<< "Starting Time Evolution." << std::endl;
    double factor = 0;
    int r = 0;
    phase = 11;
    pardiso(pt,&maxfct,&nnum,&mtype,&phase,&n_mkl,AVal,aRow,aCol,perm,&nrhs,para,&msglvl,psitmp,psi,&err);

    phase = 22;
    pardiso(pt,&maxfct,&nnum,&mtype,&phase,&n_mkl,AVal,aRow,aCol,perm,&nrhs,para,&msglvl,psitmp,psi,&err);

    phase = 33;
    for(int i = 0; i<nSteps;i++){

        if (i < freeSteps){
            //temporalStep(psi,HPotExp,n);
            mkl_cspblas_zcsrgemv(&trans, &n_mkl, BVal, bRow, bCol, psi, psitmp);
            pardiso(pt,&maxfct,&nnum,&mtype,&phase,&n_mkl,AVal,aRow,aCol,perm,&nrhs,para,&msglvl,psitmp,psi,&err);
            //temporalStep(psi,HPotExp,n);
        }
//        else if( i >= freeSteps && i < (freeSteps+rampSteps)){
//            factor = (double)r*1/(rampSteps-1);
//            temporalStepRamp(psi,VRandExp,n,factor,dt,I);
//            mkl_cspblas_zcsrgemv(&trans, &n_mkl, BVal, bRow, bCol, psi, psitmp);
//            pardiso(pt,&maxfct,&nnum,&mtype,&phase,&n_mkl,AVal,aRow,aCol,perm,&nrhs,para,&msglvl,psitmp,psi,&err);
//            temporalStepRamp(psi,VRandExp,n,factor,dt,I);
//        }
        else{
            temporalStep(psi,VRandExp,n);
            mkl_cspblas_zcsrgemv(&trans, &n_mkl, BVal, bRow, bCol, psi, psitmp);
            pardiso(pt,&maxfct,&nnum,&mtype,&phase,&n_mkl,AVal,aRow,aCol,perm,&nrhs,para,&msglvl,psitmp,psi,&err);
            temporalStep(psi,VRandExp,n);
        }

        if (err != 0 ){
            std::cout<<err<<std::endl;
        }
        if((i+1)%(1000)==0){
            std::cout<<i+1<<std::endl;
        }
        if(((i+1)%saveEvery)==0){
            std::string saveFile = "Results/Psi_";
            saveFile.append(std::to_string(i+1));
            saveFile.append(".txt");
            std::ofstream output(saveFile);
            for (int i = 0; i < n; ++i){
            output << psi[i].real<<" "<<psi[i].imag<<"\n";
            }
            output.close();
            std::cout<<"Saving file"<<std::endl;
        }
    }

//    std::ofstream output2("PsiFinal.txt");
//    for (int i = 0; i < n; ++i)
//    {
//        output2 << psi[i].real<< " "<<psi[i].imag<<"\n";
//    }
//    output2.close();

    std::cout<<"Cleaning up pardiso"<<std::endl;
    phase = -1;
    pardiso(pt,&maxfct,&nnum,&mtype,&phase,&n_mkl,AVal,aRow,aCol,perm,&nrhs,para,&msglvl,psitmp,psi,&err);
    std::cout<<"Freeing Willy" <<std::endl;
    mkl_free(AVal);
    mkl_free(BVal);
    mkl_free(aCol);
    mkl_free(aRow);
    mkl_free(bCol);
    mkl_free(bRow);
    //mkl_free(VRandExp);
    //mkl_free(HPotExp);
    mkl_free(psi);
    mkl_free(psitmp);
    return 0;

}

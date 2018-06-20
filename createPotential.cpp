#include<iostream>
#include<vector>
#include<complex>
#include<math.h>
#include<stdio.h>
#include<stdlib.h>
#include "mkl.h"
#include "mkl_dfti.h"
#include<random>
#include<chrono>
#include "createPotential.h"

std::vector<double> harmonicPotential (int const n, double const omega, std::vector<double> xVec){
    std::vector<double> harmonicPot(n);
    for(int i=0;i<n;i++){
        harmonicPot[i] = 0.5*(87.0*1.667e-27)*pow(omega,2.0)*pow(xVec[i],2.0);
    }
    return harmonicPot;
}

std::vector<double> slopeGradient (int const n, double maxVal){
    std::vector<double> slope(n);
    double grad = maxVal/(double)n;
    for(int i=0;i<n;i++){
        slope[i] = 0 + grad*(double)i;
    }
    return slope;
}

std::vector<double> randomPotential (int const n, double const corLength, double const length, double const potStr, std::complex<double> I, std::vector<double> kVec){
    MKL_Complex16 *rand = 0;
    rand = (MKL_Complex16 *)mkl_malloc(n*sizeof(MKL_Complex16),64);
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	double randStr = potStr*1.3806488e-23*1e-9;
	std::default_random_engine generator(seed);
	std::exponential_distribution<double> distribution(1);
    double kcut = 1.0/(2*corLength);
    /* Create exponentially distributed numbers*/
    for (int i = 0; i < n; i++)
	{
		rand[i].real = distribution(generator);
		rand[i].imag = 0;
	}
    std::vector<double> vRand(n);
    std::vector<double> kFilter(n);
/* Create KSpace filter*/
    for(int i=0;i<n;i++){
        if(kVec[i] < -kcut || kVec[i] > kcut){
            kFilter[i] = 0;
        }
        else{
            kFilter[i] = 1;
        }
    }
    /*Forward FFT*/
    DFTI_DESCRIPTOR_HANDLE handle;
    MKL_LONG status;

    status = DftiCreateDescriptor(&handle, DFTI_DOUBLE, DFTI_COMPLEX,1,n);
    status = DftiCommitDescriptor(handle);
    status = DftiComputeForward(handle,rand);
    status = DftiFreeDescriptor(&handle);

	MKL_Complex16 tmp;
	int n2 = n / 2;
	for (int i = 0; i < n2; i++)
	{
		tmp.real = rand[i].real;
		tmp.imag = rand[i].imag;
		rand[i].real = rand[i+n2].real;
		rand[i].imag = rand[i+n2].imag;
		rand[i+n2].real = tmp.real;
		rand[i+n2].imag = tmp.imag;
	}

    for(int i=0;i<n;i++){
        rand[i].real = rand[i].real*kFilter[i];
        rand[i].imag = rand[i].imag*kFilter[i];
    }

    /*Backward FFT*/

    	for (int i = 0; i < n2; i++)
	{
		tmp.real = rand[i].real;
		tmp.imag = rand[i].imag;
		rand[i].real = rand[i+n2].real;
		rand[i].imag = rand[i+n2].imag;
		rand[i+n2].real = tmp.real;
		rand[i+n2].imag = tmp.imag;
	}

    status = DftiCreateDescriptor(&handle, DFTI_DOUBLE, DFTI_COMPLEX,1,n);
    status = DftiCommitDescriptor(handle);
    status = DftiComputeBackward(handle,rand);
    status = DftiFreeDescriptor(&handle);
    /*Assign vRand numbers from MKL_Complex16 type*/
    for(int i=0;i<n;i++){
        vRand[i] = pow((pow(rand[i].real,2)+pow(rand[i].imag,2)),0.5);
    }
    mkl_free(rand);
    double mean = 0;
	for (int i = 0; i < n; i++){
		mean += vRand[i]/n;
	}
	double temp;
	for (int i = 0; i < n; i++){
        temp = vRand[i];
		vRand[i] = randStr*(temp/abs(mean)-1);
	}
	mean = 0;
	double meanSq = 0;
	for (int i = 0; i < n; i++){
		mean += vRand[i]/(double)n;
		meanSq += pow(vRand[i], 2)/(double)n;
	}
	std::cout << "Mean of V " << mean << std::endl;
	std::cout << "Mean of V^2 " << meanSq << std::endl;

    return vRand;
}

void createHamiltonian(int const n, double dx, std::vector<double> potential, double *hamiltonian){
    double coef = (pow(1.054e-34,2))/(2*87*1.667e-27*pow(dx,2));

    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
            if(i==j){
            hamiltonian[i*n+j]= -2 * coef + potential[i];
            }
            else if(i==j-1){
            hamiltonian[i*n+j] = coef;
            }
            else if(i==j+1){
            hamiltonian[i*n+j] = coef;
            }
            else{
            hamiltonian[i*n+j] = 0;
            }
        }
    }
    /*Boundry Conditions*/
    hamiltonian[0] = -2*coef;
    hamiltonian[n-1] = coef;
    hamiltonian[(n-1)*n+(n-1)] = -2*coef;
    hamiltonian[(n-1)*n] = coef;
}

void CreateAandB(MKL_Complex16 *A,MKL_Complex16 *B,int const n, double const dt, double *H, std::complex<double> I){
    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
            if(i==j){
                A[i*n+j].real = 1.0;
                B[i*n+j].real = 1.0;
                A[i*n+j].imag = 0.5*dt*H[i*n+j]/1.054e-34;
                B[i*n+j].imag = - 0.5*dt*H[i*n+j]/1.054e-34;
                }
            else{
                A[i*n+j].real = 0.0;
                A[i*n+j].imag = 0.5*dt*H[i*n+j]/1.054e-34;
                B[i*n+j].real = 0.0;
                B[i*n+j].imag = -0.5*dt*H[i*n+j]/1.054e-34;
                }
            }
        }
}

void CreateB(MKL_Complex16 *B, int const n, double const dt, double *H, std::complex<double> I){
    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
            if(i==j){
                B[i*n+j].real = 1.0;
                B[i*n+j].imag = - 0.5*dt*H[i*n+j]/1.054e-34;
                }
            else{
                B[i*n+j].real = 0.0;
                B[i*n+j].imag = -0.5*dt*H[i*n+j]/1.054e-34;
            }
            }
        }
}

void psiThomasFermi(MKL_Complex16 *psi, int const n, double const omega, std::vector<double> hTrap){
    double mu,a,xl,g,Nat;
    Nat = 600000.0;
    a = 5.186e-9;
    xl = pow(1.0546e-34/(87.0*1.667e-27*omega),0.5);
    mu = (1.0546e-34*omega/2.0)*pow((15.0*Nat*a/xl),2/5);
    g = 4.0*M_PI*a*pow(1.0546e-34,2)/(87.0*1.667e-27);

    for(int i=0;i<n;i++){
        if(mu>hTrap[i]){
            psi[i].real = (mu-hTrap[i])/g;
            psi[i].imag = 0.0;
        }
        else{
            psi[i].real = 0.0;
            psi[i].imag = 0.0;
        }
    }
}

void psiGaussian(MKL_Complex16 *psi, int const n, double const sigma, std::vector<double> xVec, double dx){
    double psiSum = 0;
    for(int i = 0; i<n ;i++){
        if(abs(xVec[i])<sigma*3){
            psi[i].real = (1.0/(sigma*pow(2*M_PI,0.5)))*exp(-0.5*pow((xVec[i]/sigma),2));
            psi[i].imag = 0.0;
            psiSum = psiSum+pow(psi[i].real,2);
        }
        else{
            psi[i].real = 0;
            psi[i].imag = 0;
        }
    }
    for(int i = 0; i<n; i++){
        psi[i].real = psi[i].real*pow(1/(psiSum*dx),0.5);
    }
}

void normalizePsi(MKL_Complex16 *psi,double dx, int const n){
    double psiSum = 0;
    double val;
    #pragma omp parallel for
    for(int i = 0;i<n; i++)
    {
        val = pow(psi[i].real,2.)+pow(psi[i].imag,2.);
        psiSum = psiSum+val;
    }
    double normFactor = pow(1/(psiSum*dx),0.5);
    #pragma omp parallel for
    for(int i = 0; i<n; i++){
        psi[i].real = psi[i].real*normFactor;
        psi[i].imag = psi[i].imag*normFactor;
        }
}

void ExponentiatePotential(MKL_Complex16 *ExpV, std::vector<double> potential, double dt, const int n, std::complex<double> I){
    double h = dt/(2.0*1.0546e-34);
    for(int i=0;i<n;i++){
        ExpV[i].real = cos(h*potential[i]);
        ExpV[i].imag = sin(-h*potential[i]);
    }
}

void temporalStepRamp(MKL_Complex16 *psi,MKL_Complex16 *potential, const int n, double factor, double dt, std::complex<double> I){
    std::complex<double> expVal;
    double tmpR, tmpI;
    #pragma omp parallel for
    for(int i = 0;i<n;i++){
        expVal = pow(potential[i].real+I*potential[i].imag,factor);

        tmpR = expVal.real()*psi[i].real-expVal.imag()*psi[i].imag;
        tmpI = expVal.imag()*psi[i].real+expVal.real()*psi[i].imag;
        psi[i].real = tmpR;
        psi[i].imag = tmpI;
    }
}

void temporalStep(MKL_Complex16 *psi,MKL_Complex16 *potential, const int n){
    double tmpR, tmpI;
    #pragma omp parallel for
    for(int i = 0;i<n;i++){
        tmpR = (potential[i].real)*psi[i].real-(potential[i].imag)*psi[i].imag;
        tmpI = (potential[i].imag)*psi[i].real+(potential[i].real)*psi[i].imag;
        psi[i].real = tmpR;
        psi[i].imag = tmpI;
    }

}

void interactionStep(MKL_Complex16 *psi, const int n){
    double g, tmpR, tmpI, tmpR2, tmpI2;
    g = 0;
    for(int i = 0; i<n; i++){
        tmpR = cos(g*(psi[i].real*psi[i].real+psi[i].imag*psi[i].imag));
        tmpI = -sin(g*(psi[i].real*psi[i].real+psi[i].imag*psi[i].imag));
        tmpR2 = psi[i].real*tmpR - psi[i].imag*tmpI;
        tmpI2 = psi[i].real*tmpI + psi[i].imag*tmpR;
        psi[i].real = tmpR2;
        psi[i].imag = tmpI2;
    }
}

void apply_kick(MKL_Complex16 *psi, std::vector<double> xAr, const int n, double beta){
    double recoil = 2*M_PI/780e-9;
    double tmpR, tmpI;
    for(int i = 0; i<n; i++){
        tmpR = psi[i].real*cos(beta*recoil*xAr[i])-psi[i].imag*sin(-beta*recoil*xAr[i]);
        tmpI = psi[i].real*sin(-beta*recoil*xAr[i])+psi[i].imag*cos(beta*recoil*xAr[i]);
        psi[i].real = tmpR;
        psi[i].imag = tmpI;
    }

}

std::vector<double> specklePotential(int const n, double const corLength, double const length, double const potStr, std::complex<double> I, std::vector<double> kVec){

    MKL_Complex16 *phase = 0;
    phase = (MKL_Complex16 *)mkl_malloc(n*sizeof(MKL_Complex16),64);
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	double randStr = potStr*1.3806488e-23*1e-9;
	std::default_random_engine generator(seed);
	std::exponential_distribution<double> distribution(300);
    double kcut = 1.0/(2*corLength);
    double sigX = 450.0;
    double gaussian = 0;
    //Create exponentially distributed numbers
    double rand = 0;
    for (int i = 0; i < n; i++)
	{
		rand = fmod(distribution(generator),2*M_PI);
		phase[i].real = cos(rand);
		phase[i].imag = sin(rand);
	}
    std::vector<double> vRand(n);
    //Create initial Electric field

    MKL_Complex16 *EField = 0;
    EField = (MKL_Complex16 *)mkl_malloc(n*sizeof(MKL_Complex16),64);
    for(int i=0;i<n;i++){
        if(abs(kVec[i])<kcut){
            gaussian = pow(exp(-pow((i-((double)n/2.0)),2)/pow(2*sigX,2)),0.5);
            EField[i].real = gaussian*phase[i].real;
            EField[i].imag = gaussian*phase[i].imag;
        }
        else{
            EField[i].real = 0;
            EField[i].imag = 0;
        }
    }
    //Forward FFT
    DFTI_DESCRIPTOR_HANDLE handle;
    MKL_LONG status;

    status = DftiCreateDescriptor(&handle, DFTI_DOUBLE, DFTI_COMPLEX,1,n);
    status = DftiCommitDescriptor(handle);
    status = DftiComputeForward(handle,EField);
    status = DftiFreeDescriptor(&handle);

//	MKL_Complex16 tmp;
//	int n2 = n / 2;
//	for (int i = 0; i < n2; i++)
//	{
//		tmp.real = rand[i].real;
//		tmp.imag = rand[i].imag;
//		rand[i].real = rand[i+n2].real;
//		rand[i].imag = rand[i+n2].imag;
//		rand[i+n2].real = tmp.real;
//		rand[i+n2].imag = tmp.imag;
//	}

    for(int i=0;i<n;i++){
        vRand[i] = (pow(EField[i].real,2)+pow(EField[i].imag,2));
    }
    mkl_free(phase);
    mkl_free(EField);
    double mean = 0;
	for (int i = 0; i < n; i++){
		mean += vRand[i]/(double)n;
	}
	double temp;
	for (int i = 0; i < n; i++){
        temp = vRand[i];
		vRand[i] = randStr*(temp/abs(mean)-1);
	}
	mean = 0;
	double meanSq = 0;
	for (int i = 0; i < n; i++){
		mean += vRand[i]/(double)n;
		meanSq += pow(vRand[i], 2)/(double)n;
	}
	std::cout << "Mean of V " << mean << std::endl;
	std::cout << "Mean of V^2 " << meanSq << std::endl;

    return vRand;
}

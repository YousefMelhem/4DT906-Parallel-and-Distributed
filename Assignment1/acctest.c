/*
Test of C code speed with Apple Accelerate Framework and openmp
https://developer.apple.com/library/mac/documentation/Accelerate/Reference/BLAS_Ref/Reference/reference.html#//apple_ref/c/func/cblas_sgemm
compile with: 
gcc -Ofast -fopenmp -flax-vector-conversions -framework Accelerate acctest.c 
*/
 
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <sys/time.h>
#include <math.h>
#include <omp.h>
#include <Accelerate/Accelerate.h>
 
int main()
{
  int i,j,k,l;
  // nb of operations:
  const int dsize = 32;
  int nthreads = 8;
  int nbOfAverages = 100000;  //1e5
  int opsMAC = 2; // operations per MAC
  const float *A, *B;
  float *C;
  const float alpha = 1, beta = 1;
  //unsigned long int 
  double tops; //total ops

  // allocate matrices
  A = (const float *) malloc(dsize*dsize*sizeof(float));
  B = (const float *) malloc(dsize*dsize*sizeof(float));
  C = (float *) malloc(dsize*dsize*sizeof(float));
 
  struct timeval start,end;
  gettimeofday(&start, NULL);
 
 
#pragma omp parallel for private (i,k)
  for (k=0; k<nthreads; k++) {
    //printf("Hello from thread %d, nthreads %d\n", omp_get_thread_num(), omp_get_num_threads());
    for(i=0;i<nbOfAverages;i++) {
      // mul 2 matrices:
      //CblasRowMajor = 101, CblasNoTrans = 111
      cblas_sgemm(101,111,111, dsize,dsize,dsize,alpha, A, dsize, B, dsize, beta, C, dsize);
    }
  }
 
  gettimeofday(&end, NULL);
  double t = ((double) (end.tv_sec - start.tv_sec))
    + ((double) (end.tv_usec - start.tv_usec)) / 1e6; //reports time in [s] - verified!
 
  // report performance:
  tops = nthreads * opsMAC * dsize*dsize*dsize; // total ops
  printf("\nTotal M ops = %lf, # of treads = %d", tops/10, nthreads); // tops/10 because they are printed in M
  printf("\nTime in s: %lf:", t);
  printf("\nTest performance [G OP/s] %lf:", tops/t/1e4);
  printf("\n");
  return(0);
}
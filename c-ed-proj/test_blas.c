#include <stdlib.h>
#include <stdio.h>
#include <cblas.h>
#include "print_array.h"

void main ()
{
  int m = 3;
  int n = 3;
  int k = 3;
  double * A = malloc( m * n * sizeof(double) );
  double * B = malloc( n * k * sizeof(double) );
  double * C = malloc( m * k * sizeof(double) );
  double alpha, beta;

  alpha = 1.0;
  beta = 0.0;

  int ii;
  // initialize A
  for( ii = 0; ii < m * n; ii ++)
    {
      A[ii] = ii;
    }

  // initialize B
  for( ii = 0; ii < n * k; ii++)
    {
      B[ii] = n * k - ii;
    }
  
  // initialize C
  for( ii = 0; ii < m * k; ii++)
    {
      C[ii] = 0;
    }
  
  
  // C = alpha * A * B + beta * C
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, A, k, B, n, beta, C, n);

  // print results
  printf("C = alpha * A * B + beta * C\n");
  printf("alpha = %0.2f, beta = %0.2f\n", alpha, beta);
  printf("A\n");
  print_array_double(A, m, n);
  printf("B\n");
  print_array_double(B, n, k);
  printf("C\n");
  print_array_double(C, m, k);


}

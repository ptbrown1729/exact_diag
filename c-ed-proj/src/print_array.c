#include "print_array.h"
#include <stdio.h>
#include <stdlib.h>

int print_array_double(double * array, int num_rows, int num_cols)
{
  int ii, jj;
  for(ii=0;ii<num_rows;ii++)
    {
      printf("%d:\t",ii);
      for(jj=0;jj<num_cols-1;jj++)
	{
	  printf("%0.2f,",array[ii*num_cols+jj]);
	}
      printf("%0.2f\n",array[(ii+1)*num_cols - 1]);
    }
  printf("\n");
  return 0;
}

int print_array_int(int * array, int num_rows, int num_cols)
{
  int ii, jj;
  for(ii=0;ii<num_rows;ii++)
    {
      printf("%d:\t",ii);
      for(jj=0;jj<num_cols-1;jj++)
	{
	  printf("%d,",array[ii*num_cols+jj]);
	}
      printf("%d\n",array[(ii+1)*num_cols - 1]);
    }
  printf("\n");
  return 0;
}

int print_array_bool(bool * array, int num_rows, int num_cols)
{
  int ii, jj;
  for(ii=0;ii<num_rows;ii++)
    {
      printf("%d:\t",ii);
      for(jj=0;jj<num_cols-1;jj++)
	{
	  printf("%d,",array[ii*num_cols+jj]);
	}
      printf("%d\n",array[(ii+1)*num_cols - 1]);
    }
  printf("\n");
  return 0;
}




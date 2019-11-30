#include "utility.h"

int min(int n, int m)
{
  if(n <= m)
    {
      return n;
    }
  else
    {
      return m;
    }
}
  
int max(int n, int m)
{
  if(n <=m)
    {
      return m;
    }
  else
    {
      return n;
    }
}
  
int factorial(int n)
{
	int ii;
	for(ii=(n-1);ii>0;ii--){
		n = n*ii;
	}
	return n;
}

int nchoosek(int n, int k)
{
	int ii;
	int div = factorial(n-k);
	for(ii = n-1;ii>k;ii--)
	{
		n = n*ii;
	}
	return n/div;
}

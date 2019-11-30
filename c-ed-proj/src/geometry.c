#include "geometry.h"
#include <stdlib.h>
#include <math.h>

int create_geometry(int xsites, int ysites, int periodic_x, int periodic_y, int number_left2right_up2down, double * xlocs, double * ylocs, bool * connect_mat)
{
  // Given the number of sites in the x and y directions and boundary conditions,
  // return the locations of the sites and the site connection matrix, which is
  // an nsites x nsites matrix, such that the iith, jjth entry is 1 sites ii and
  // jj are nearest neighbors.
  //TODO: Currently this function does not handle 2x2 periodic b.c. correctly
  //create rectangular geometry
  int ii, jj;
  int nsites = xsites * ysites;

  //first convenient to zero connect_mat
  for(ii = 0; ii < (nsites * nsites); ii++)
    {
      connect_mat[ii] = 0;
    }

  for(ii = 0; ii < ysites; ii++)
    {
      for(jj = 0; jj < xsites; jj++)
	{
	  xlocs[ii * xsites + jj] = (double)jj;
	  ylocs[ii * xsites + jj] = -(double)ii; //minus sign so state 0 is upper left hand corner
	  //construct connection matrix
	  //to understand the indexing in this section, think of site_index = ii * ysites + jj
	  //for site at position [ii,jj]
	  //hopping along x (no periodic b.c.)
	  if(jj < xsites - 1)
	    {
	      connect_mat[(ii * xsites + jj) * nsites + (ii * xsites + jj + 1)] = 1;
	      connect_mat[(ii * xsites + jj + 1) * nsites + (ii * xsites + jj)] = 1;
	    }
	  //hopping along x (periodic b.c.)
	  if((jj == xsites - 1) && periodic_x)
	    {
	      connect_mat[(ii * xsites + jj) * nsites + (ii * xsites + jj - (xsites - 1))] = 1;
	      connect_mat[(ii * xsites + jj - (xsites - 1)) * nsites + (ii * xsites + jj)] = 1;
	    }
	  
	  // hopping along y (no periodic b.c.)
	  if(ii < ysites - 1)
	    {
	      connect_mat[(ii * xsites + jj) * nsites + ( (ii + 1) * xsites + jj)] = 1;
	      connect_mat[((ii + 1) * xsites + jj) * nsites + (ii * xsites + jj)] = 1;
	    }
	  //hopping along y (periodic b.c.)
	  if((ii == ysites - 1) && periodic_y)
	    {
	      connect_mat[(ii * xsites + jj) * nsites + ((ii - (ysites - 1 )) * xsites + jj)] = 1;
	      connect_mat[((ii - (ysites - 1 )) * xsites + jj) * nsites + (ii * xsites + jj)] = 1;
	      }
	}
    }
  return 0;
}

int center_locs(double * xlocs, double * ylocs, int nsites)
{
  //center xlocs and ylocs about the origin
  int ii;
  double xcom, ycom;
  xcom = 0.0;
  ycom = 0.0;
  for(ii = 0; ii < nsites; ii++)
    {
      xcom = xcom + xlocs[ii];
      ycom = ycom + ylocs[ii];
    }
  xcom = xcom / (double)nsites;
  ycom = ycom / (double)nsites;

  for(ii = 0; ii < nsites; ii++)
    {
      xlocs[ii] = xlocs[ii] - xcom;
      ylocs[ii] = ylocs[ii] - ycom;
    }
		 
  return 0;
}

#include "transform.h"
#include "states.h"
#include "utility.h"
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <cblas.h>

int reflfn(double * xlocs, double * ylocs, int nsites, double * refl_vect, double * xlocs_refl, double * ylocs_refl)
// Given site coordinates, find the new coordinates after applying a reflection about refl_vect
{
  //TODO: implement besides reflection along y
  int ii;
  for(ii = 0; ii < nsites; ii++)
    {
      xlocs_refl[ii] = -xlocs[ii];
      ylocs_refl[ii] = ylocs[ii];
    }
  return 0;
}

int get_transformed_sites(int (*trans_fn)(double *, double *, int, double *, double *, double*), double * xlocs, double * ylocs, int * trans_site_labels, int nsites)
// Given sites locations and a transformation function, find which site each site
// transforms into under the transformation. Returns an array where the ith
// element is T(site[ii])
// TODO: implement taking arbitrary function
{
  int ii, jj;
  double * xlocs_trans = malloc(nsites * sizeof(double));
  double * ylocs_trans = malloc(nsites * sizeof(double));
  double * refl_vect;
  double round_precision = 0.0001;
  //reflfn(xlocs, ylocs, nsites, refl_vect, xlocs_trans, ylocs_trans);
  (*trans_fn)(xlocs, ylocs, nsites, refl_vect, xlocs_trans, ylocs_trans);
  for(ii = 0; ii < nsites; ii++)
    {
      for(jj = 0; jj < nsites; jj ++)
	{
	  if(fabs(xlocs[ii] - xlocs_trans[jj])<round_precision && fabs(ylocs[ii] - ylocs_trans[jj])<round_precision)
	    {
	      trans_site_labels[ii] = jj;
	    }
	}
    }
  free(xlocs_trans);
  free(ylocs_trans);
  return 0;
}

int get_site_cycles(int (*trans_fn)(double *, double *, int, double *, double *, double*), double * xlocs, double * ylocs, int nsites, int max_transforms, int * num_cycles, int * cycle_lens, int * cycles)
//int get_site_cycles(double * xlocs, double * ylocs, int nsites, int max_transforms, int * num_cycles, int * cycle_lens, int * cycles)
//num_cycles is a single number, cycle_lens should be an array of size nsites, and cycles should be an array of size nsites x max_transforms
//due to the extra size, cycles has a number of entries taht must be ignored. You need only care about the first *num_cycles rows, and the first
//cycle_lens[ii] columns of row ii.
{
  int ii, jj;
  int * trans_site_labels = malloc(nsites * sizeof(int));
  int * site_trans_arr = malloc(nsites * max_transforms * sizeof(int));
  //  int (*fptr)(double *, double *, int, double *, double *, double *);
  //  fptr = reflfn;
  
  for (ii = 0; ii < nsites; ii++)
    {
      trans_site_labels[ii] = ii; //first state labels are {0, 1, ...}
      cycle_lens[ii] = 0; //initialize to zero
    }

  //first get set of all sites transforming into each other
  for(ii = 0; ii < max_transforms; ii++)
    {
      for(jj = 0; jj < nsites; jj++)
	{
	  site_trans_arr[jj * max_transforms + ii] = trans_site_labels[jj];
	}    
      get_transformed_sites(trans_fn, xlocs, ylocs, trans_site_labels, nsites);
      //get_transformed_sites(xlocs, ylocs, trans_site_labels, nsites);
      //TODO: uncomment this...just testing function pointer
    }

  //now reduce it to a set of cycles

  *num_cycles = 0; //this line is causing segmetntation fault! Why??
  
  for(ii = 0; ii < nsites; ii++)
    {
      //during this loop, *num_cycles gives the number of cycles we've completed so far
      //start work on this cycle if we have not already excluded it
      if(site_trans_arr[ii * max_transforms] != -1)
	{
	  cycle_lens[*num_cycles] = 1;
	  cycles[*num_cycles * max_transforms] = site_trans_arr[ii * max_transforms];
	  
	  jj = 1;
	  while(site_trans_arr[ii * max_transforms + jj] != site_trans_arr[ii * max_transforms] & jj < max_transforms)
	    {
	      cycle_lens[*num_cycles] = cycle_lens[*num_cycles] + 1;
	      // set corresponding entry to cycle_array to minus one so we know to ignore it later
	      site_trans_arr[ site_trans_arr[ii * max_transforms + jj] * max_transforms] = -1;
	      // set entry in output array
	      cycles[*num_cycles * max_transforms + jj] = site_trans_arr[ii * max_transforms + jj];
	      jj = jj +1;
	    }
	  (*num_cycles)++; // = *num_cycles + 1;
	}
    }
  return 0;
}

int get_swap_op(bool * states_bool, int nsites, int nstates, int site1, int site2, double * swap_op)
// return operators swapping sites ii and jj. nstates x nstates matrix.
// changed swap_op to type double because blas doesn't support int
{
  int min_site = min(site1,site2);
  int max_site = max(site1,site2);
  //printf("max site = %d, min site = %d\n",max_site, min_site);
  int ii, jj;
  int sign = 1;
  int index_trans_state;
  bool * trans_state = malloc(nsites * sizeof(bool));
  for(ii = 0; ii < nstates * nstates; ii++)
    {
      swap_op[ii] = 0;
    }
  //printf("initialized swap_op to zeros\n");
  for(ii = 0; ii < nstates; ii++)
    {
      //printf("ii = %d\n",ii);
      for(jj = 0; jj < nsites; jj++)
	{
	  //printf("jj = %d\n",jj);
	  trans_state[jj] = states_bool[ii * nsites + jj];
	}
      trans_state[min_site] = states_bool[ii * nsites + max_site];
      trans_state[max_site] = states_bool[ii * nsites + min_site];
      index_trans_state = find_state(states_bool, trans_state, nsites, nstates);
      //sign related to number of occupied sites between the sites we are switching
      for(jj = min_site + 1; jj < max_site; jj++)
	{
	  if(trans_state[jj])
	    {
	      sign = -1 * sign;
	    }
	}
      swap_op[ii * nstates + index_trans_state] = (double) sign;
      swap_op[index_trans_state * nstates + ii] = (double) sign;
      sign = 1;
    }
  return 0;
}

int get_trans_op_from_swap(bool * states_bool, int nsites, int nstates, int max_transforms, int ncycles, int * cycle_lens, int * cycles, double * trans_op)
// not sure what the best way to do this in this case is...
{
  int ii, jj;
  double * curr_swap_op = malloc(nsites * nsites * sizeof(double));
  //double * cmat = malloc(nsites * nsites * sizeof(double)); // get rid of this in the end
  double alpha = 1.0;
  double beta = 0.0;
  //initialize trans_op to identity
  for(ii = 0; ii < nstates; ii++)
    {
      for(jj = 0; jj < nstates; jj++)
	{
	  if( ii == jj)
	    {
	      trans_op[ii * nstates + jj] = 1;
	    }
	  else
	    {
	      trans_op[ii * nstates + jj] = 0;
	    }
	}
    }
  
  //get cycle transform from swap operators
  for(ii = 0; ii < ncycles; ii++)
    {
      for(jj = cycle_lens[ii] - 1; jj > 0; jj--)
	{
	  get_swap_op(states_bool, nsites, nstates, cycles[ii * max_transforms + jj - 1], cycles[ii * max_transforms + jj], curr_swap_op);
	  //cycle_op = curr_op * cycle_op
	  //output stored in third mat input
	  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, nstates, nstates, nstates, alpha, curr_swap_op, nstates, trans_op, nstates, beta, trans_op, nstates);
	}
    }
  //compose these to get full transformation operator
  return 0;
}

int get_trans_op(bool * states_bool, int nsites, int nstates, int * trans_sites, int * swap_op)
// return operators swapping sites ii and jj. nstates x nstates matrix.
// trying to do this directly without swaps...but I think it is too hard to get the sign correct
{
  int ii, jj;
  int sign = 1;
  int index_trans_state;
  bool * trans_state = malloc(nsites * sizeof(bool));

  for(ii = 0; ii < nstates * nstates - 1; ii++)
    // initialize to zeros
    {
      swap_op[ii] = 0;
    }

  for(ii = 0; ii < nstates; ii++)
    {     
      for(jj = 0; jj < nsites; jj++)
	{
	  trans_state[trans_sites[jj]] = states_bool[ii * nsites + jj];
	}

      index_trans_state = find_state(states_bool, trans_state, nsites, nstates);
      //sign related to number of occupied sites between the sites we are switching
            for(jj = 0; jj < nsites; jj++)
	{
	  // how do I get the right sign in this case???
	  if(trans_state[jj])
	    {
	      sign = -1 * sign;
	    }
	}
      swap_op[ii * nstates + index_trans_state] = sign;
      swap_op[index_trans_state * nstates + ii] = sign;
      sign = 1;
    }
  return 0;
}

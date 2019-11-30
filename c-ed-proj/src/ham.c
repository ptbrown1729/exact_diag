#include "ham.h"
#include "states.h"
#include <stdlib.h>
#include <math.h>

int get_onespin_hoppingH(bool * connect_mat, bool * states_bool, int nsites, int nstates, double * ham)
// Get hopping Hamiltonian given a connection matrix and a list of states
{
  int ii, jj, kk;
  bool states_xor[nsites];
  int diff; //number of hopes to transform between two states
  int first_site; //first site for hop
  int second_site; //second site for hop
  int occ_sites_btw; //number of occupied sites numerically between hopping states
  for(ii=0;ii<nstates;ii++)
    {
      // don't sum over full matrix, only the half above diagonal (including diagonal)
      for(jj=ii;jj<nstates;jj++)
	{
	  xor_states(&states_bool[ii * nsites + 0], &states_bool[jj * nsites + 0], states_xor, nsites);
	  diff = 0;
	  for(kk=0;kk<nsites;kk++)
	    {
	      diff = diff + states_xor[kk];
	    }
	  if(diff != 2)
	    //if two states differ by more than one hop, matrix element is zero
	    {
	      ham[ii * nstates + jj] = 0;
	      ham[jj * nstates + ii] = 0;
	    }
	  else
	    {
	      // find which sites are occupied
	      first_site = 0;
	      second_site = 0;
	      for(kk=0;kk<nsites;kk++)
		{
		  if (states_xor[kk] != 0)
		    {
		      if (first_site == 0)
			{
			  first_site = kk;
			}
		      else
			{
			  second_site = kk;
			}
		    }
		}
		    
	      if(connect_mat[first_site * nsites + second_site] == 1)
		//check if those two sites are coupled by H
		{
		  occ_sites_btw = 0;
						
		  for(kk=first_site;kk<second_site;kk++)
		    {
		      occ_sites_btw = occ_sites_btw + states_bool[jj * nsites + kk];
		    }
		  ham[ii * nstates + jj] = pow(-1,occ_sites_btw);
		  ham[jj * nstates + ii] = ham[ii * nstates + jj];
		}
	      else
		//if sites not coupled, matrix element again zero
		{
		  ham[ii * nstates + jj] = 0;
		  ham[jj * nstates + ii] = 0;
		}
	    }
	}
    }
  return 0;
}

int get_int_ham(bool * states_bool, int nstates, int nsites, int nspins, double * int_ham)
// Generate a vector representing the diagonal of the interaction Hamiltonian
// TODO: generalize to spin ups and spin downs having different number of spins
{
  //our final space will be kron(down-spins,up-spins)
  int ii, jj;
  int num_double_occs;
  for(ii = 0; ii < nstates; ii++)
    //ii's label down spins
    {
      for(jj = 0; jj < nstates; jj++)
	//jj's label up spins
	{
	  int_ham[ii * nstates + jj] = (double)get_num_double_occs(&states_bool[ii * nsites], &states_bool[jj * nsites], nsites);
	}
    }
  return 0;
}

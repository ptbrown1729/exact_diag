#include "states.h"
#include <stdlib.h>
#include <math.h>
#include <stdio.h>


int xor_states(bool * state1, bool * state2, bool * state_xor, int state_len)
{
   int ii;
   for(ii=0;ii<state_len;ii++)
     {
       if (state1[ii] != state2[ii])
	 {
	   state_xor[ii] = 1;
	 }
       else
	 {
	   state_xor[ii] = 0;
	 }
       
     }
   return 0;
}

int compare_states(bool * state1, bool * state2, int nsites)
// Return 1 or 2, depending on which state is larger when comparing the
// binary representation in dictionary order. If the states are equal,
// return  0. e.g. (1,1,0,0)>(0,1,1,0).
{
  int ii = 0;
  int cont = 1;
  int bigger_state;
  while(cont)
    {
      if(state1[ii] > state2[ii])
	{
	  bigger_state = 1;
	  cont = 0;
	}
      else if(state1[ii] < state2[ii])
	{
	  bigger_state = 2;
	  cont = 0;
	}
      else
	{
	  if(ii == nsites - 1)
	    {
	      bigger_state = 0;
	      cont = 0;
	    }
	}
      ii = ii + 1;
    }
  //return 0, 1, or 2 if states are equal, state 1 is bigger, or state2 is bigger, respectively
  return bigger_state;
}

int find_state(bool * states_bool, bool * state, int nsites, int nstates)
{
  //return the index of a specific state from our list of states
  //make use of the fact our list is ordered
  int last_bigger = 0;
  int last_smaller = nstates - 1;
  int compare_index = (int)floor((last_bigger + last_smaller) / 2);
  int compare_result = 1;
  while(compare_result != 0)
    {
      //printf("last_bigger = %d, last_smaller = %d, compare_index = %d, compare_result = %d\n",last_bigger,last_smaller, compare_index, compare_result);
      compare_result = compare_states(state, &states_bool[compare_index * nsites], nsites);
      if (compare_result == 2) //if our state is smaller
	{
	  //printf("compare_result = 2\n");
	  last_bigger = compare_index;
	  compare_index = (int)ceil((double)(compare_index + last_smaller) / 2);
	}
      else if (compare_result == 1) //if our state is bigger
	{
	  //printf("compare_resultt = 1\n");
	  last_smaller = compare_index;	
	  compare_index = (int)floor((double)(compare_index + last_bigger) / 2);
	}
    }
  return compare_index;
}

int get_num_double_occs(bool * state1, bool * state2, int state_len)
// Return the number of doubly occupied sites given two states for different
// spins
{
  int ii;
  int num_double_occs = 0;
  for(ii = 0; ii < state_len; ii ++)
    {
      if(state1[ii] == 1)
	{
	  if(state1[ii] == state2[ii])
	    {
	      //state_and[ii] = 1;
	      num_double_occs = num_double_occs + 1;
	    }
	}
    }
  return num_double_occs;
}


int get_state_reps(int * states_int, bool * states_bool, int nsites, int nspins, int nstates)
//return a list of all basis states.
//states_int rows are integers representing which sites are occupied by an atom. nstates x nspins matrix
//states_bool rows are zeros or ones indicating which sites are occupied or not. nstates x nsites matrix
{
  int ii, jj, kk;
  //initial state, boolean storage method
  for(ii=0;ii<nsites;ii++)
    {
      states_bool[ii] = 0;
	
    }
	
  // initial state, first in our ordering, spins on sites 0...nspins-1
  for(ii=0;ii<nspins;ii++)
    {
      states_int[ii] = ii;
      states_bool[ii] = 1;

    }
	
  // generate next states
  for(ii=1;ii<nstates;ii++)
    {
      // most of our moves are incrementing the last spin
      if(states_int[(ii - 1) * nspins + (nspins - 1)] < nsites - 1)
	{
	  //increment last spin
	  states_int[ii * nspins + (nspins - 1)] = states_int[(ii - 1) * nspins + (nspins - 1)] + 1;
	  //keep others the same
	  for(kk=0;kk<nspins-1;kk++)
	    {
	      states_int[ii * nspins + kk] = states_int[(ii-1) * nspins +  kk];
	    }
	}
      // once we can't increment it anymore, look to the next spin
      else
	{
	  jj = nspins-2;
	  //look for the first state we can increment
	  while(states_int[(ii - 1) * nspins + jj] == states_int[(ii - 1) * nspins + (jj + 1)] - 1)
	    {
	      jj = jj-1;
	    }
	  //increment it
	  states_int[ii * nspins + jj] = states_int[(ii-1) * nspins + jj] + 1;
	  //set all the further spins right after it
	  for(kk=jj+1;kk<nspins;kk++)
	    {			 
	      states_int[ii * nspins  + kk] = states_int[ii * nspins + jj] + (kk-jj);
	    }
	  //set all previous spins same as previous state
	  for(kk=0;kk<jj;kk++)
	    {		
	      states_int[ii * nspins + kk] = states_int[(ii-1) * nspins + kk];
	    }
	}
      //initialize boolean array to zeros
      for(jj = 0;jj<nsites;jj++)
	{		 
	  states_bool[ii * nsites + jj] = 0;
	}
      //assign ones in appropriate places
      for(jj = 0;jj<nspins;jj++)
	{
	  states_bool[ii * nsites + states_int[ii * nspins + jj]] = 1;
	}
    }

  return 0;
}

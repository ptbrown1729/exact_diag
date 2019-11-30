#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <math.h>
//#include "mkl.h"
//#include "csparse.h"
#include <time.h>
#include "states.h"
#include "ham.h"
#include "transform.h"
#include "geometry.h"
#include "utility.h"
#include "print_array.h"

void main()
{
	// c uses row major indexing
        // i.e. if you have a point to an array of size n x m and you want to think of it as a 2D array
        // array[ii * m + jj] = array[ii,jj]
        clock_t begin = clock();

	//initialize the variables we are going to need
	int print_states = 1;
	int xsites = 2;
	int ysites = 2;
	int x_periodic_bc = 0;
	int y_periodic_bc = 0;
	int twisted_numbering = 0;
        int nsites = xsites * ysites;
	int nspins = floor((double)nsites / 2.0);
	int nstates = nchoosek(nsites, nspins);
	double * xlocs = malloc(nsites * sizeof(double)); // x-locations of each site
	double * ylocs = malloc(nsites * sizeof(double));
	int * states = malloc(nstates * nspins * sizeof(int)); 	// store integer position of each spins, e.g. |0 1 0 0 1> = {1,4}
	bool * states_bool = malloc(nstates * nsites * sizeof(bool));
	bool * connect_mat = malloc(nsites * nsites * sizeof(bool));
	double * ham = malloc(nstates * nstates * sizeof(double)); // hopping hamiltonian for single spin state
	bool * rot_mat = malloc(nstates * nstates * sizeof(bool)); // rotation matrix for single spin state
	double * ham_u = malloc(nstates * nstates * sizeof(double)); //diagonal of invteraction matrix       	
	
	//construct geometry: x and y locations and hopping connection matrix
	create_geometry(xsites, ysites, x_periodic_bc, y_periodic_bc, twisted_numbering, xlocs, ylocs, connect_mat);
	center_locs(xlocs, ylocs, nsites);
	//transformation
	int * trans_site_labels = malloc(nsites * sizeof(int));
	int (*fptr)(double *, double *, int, double *, double *, double *);
	fptr = reflfn;
	get_transformed_sites(fptr, xlocs, ylocs, trans_site_labels, nsites);
	//get_transformed_sites(xlocs, ylocs, trans_site_labels, nsites);
	//test site cycle finder
	int num_cycles;
	int * cycle_lens = malloc( 2 * sizeof(int));
	int * cycles = malloc(nsites * 2 *sizeof(int));
	get_site_cycles(fptr, xlocs, ylocs, nsites, 2, &num_cycles, cycle_lens, cycles);
	
	//generate reperesentations of states
	get_state_reps(states, states_bool, nsites, nspins, nstates);
	//create Hamiltonian for a single spin state
	//TODO: rewrite this in terms of indptr,indices,data numpy sparse matrix???
	get_onespin_hoppingH(connect_mat, states_bool, nsites, nstates, ham);
	//generate Hamiltonian for both spins, including interaction term
	//this is essentially tensor product of the two spin spaces. but then need to loop along the diagonal to get U's.
	get_int_ham(states_bool, nstates, nsites, nspins, ham_u);       
	
	//test finding state
	bool state[4] = {0,0,1,1};
	int state_ind = find_state(states_bool, state, nsites, nstates);
	printf("Found state %d\n",state_ind);

	//test swap op
	double * swap_op = malloc(nstates * nstates * sizeof(double));
	get_swap_op(states_bool, nsites, nstates, 0, 3, swap_op);
	printf("swap op\n");
	print_array_double(swap_op, nstates, nstates);

	//test transform op
	double * trans_op = malloc(nstates * nstates * sizeof(double));
	get_trans_op_from_swap(states_bool, nsites, nstates, 2, num_cycles, cycle_lens, cycles, trans_op);
	printf("transform op\n");
	print_array_double(trans_op, nstates, nstates);
	

	//print representations of matrices
	if (print_states == 1)
	  {
	    //display locations
	    printf("xlocs\n");
	    print_array_double(xlocs, nsites, 1);
	    printf("ylocs\n");
	    print_array_double(ylocs, nsites, 1);
	    //display transformations
	    printf("transform\n");
	    print_array_int(trans_site_labels, nsites, 1);
	    //display site cycles
	    printf("site cycles\n");
	    print_array_int(cycles, nsites, 2);
	    printf("cycle lens\n");
	    print_array_int(cycle_lens, nsites, 1);
	    printf("number of cycles = %d\n", num_cycles);
    	    //display hopping matrix
	    printf("hopping connections matrix\n");
	    print_array_bool(connect_mat, nsites, nsites);
	    //display states
	    printf("States\n");
	    print_array_int(states, nstates, nspins);
	    print_array_bool(states_bool, nstates, nsites);	    
	    //display one-spin hopping H
	    printf("Hopping hamiltonian\n");
	    print_array_double(ham, nstates, nstates);
	    //display interaction H
	    printf("Interaction diagonal\n");
	    print_array_double(ham_u, nstates * nstates, 1);
	  }
	
	//free memory
	free(xlocs);
	free(ylocs);
	free(states);
	free(states_bool);
	free(connect_mat);	
	free(ham);
	free(rot_mat);
	free(ham_u);
       
	//print timing information
	clock_t end = clock();
	double total_time = (double)(end - begin) / CLOCKS_PER_SEC;
	printf("Total time for %d sites and %d spins was %0.2f s\n",nsites,nspins,total_time);
}

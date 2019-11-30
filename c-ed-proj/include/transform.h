#ifndef TRANSFORM_INCLUDED
#define TRANSFORM_INCLUDED

#include <stdbool.h>

int reflfn(double * xlocs, double * ylocs, int nsites, double * refl_vect, double * xlocs_refl, double * ylocs_refl);

//int get_transformed_sites(double * xlocs, double * ylocs, int * trans_site_labels, int nsites);

int get_transformed_sites(int (*trans_fn)(double *, double *, int, double *, double *, double*), double * xlocs, double * ylocs, int * trans_site_labels, int nsites);

int get_site_cycles(int (*trans_fn)(double *, double *, int, double *, double *, double*), double * xlocs, double * ylocs, int nsites, int max_transforms, int * num_cycles, int * cycle_lens, int * cycles);

//int get_site_cycles(double * xlocs, double * ylocs, int nsites, int max_transforms, int * num_cycles, int * cycle_lens, int * cycles);

int get_swap_op(bool * states_bool, int nsites, int nstates, int site1, int site2, double * swap_op);

int get_trans_op_from_swap(bool * states_bool, int nsites, int nstates, int max_transforms, int ncycles, int * cycle_lens, int * cycles, double * trans_op);

int get_trans_op(bool * states_bool, int nsites, int nstates, int * trans_sites, int * swap_op);
  
#endif

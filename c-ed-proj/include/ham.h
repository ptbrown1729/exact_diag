#ifndef HAM_INCLUDED
#define HAM_INCLUDED

#include <stdbool.h>

int get_onespin_hoppingH(bool * connect_mat, bool * states_bool, int nsites, int nstates, double * ham);

int get_int_ham(bool * states_bool, int nstates, int nsites, int nspins, double * int_ham);

#endif

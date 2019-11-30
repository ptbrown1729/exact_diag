#ifndef STATES_INCLUDED
#define STATES_INCLUDED

#include <stdbool.h>

int xor_states(bool * state1, bool * state2, bool * state_xor, int state_len);

int get_num_double_occs(bool * state1, bool * state2, int state_len);

int get_state_reps(int * states_int, bool * states_bool, int nsites, int nspins, int nstates);

int compare_states(bool * state1, bool * state2, int nsites);

int find_state(bool * states_bool, bool * state, int nsites, int nstates);

#endif

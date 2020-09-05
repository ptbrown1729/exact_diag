import numpy as np
import math
import time

def nchoosek(n,k):
	return math.factorial(n)/math.factorial(k)/math.factorial(n-k)

def get_connect_mat(nsites):
	connect_mat = np.zeros([nsites,nsites])
	for ii in range(0,nsites):
		for jj in range(0,nsites):
			if jj==ii+1 or jj==ii-1:
				connect_mat[ii,jj] = 1
			else:
				connect_mat[ii,jj] = 0
	return connect_mat

def get_state_reps(nsites, nspins, nstates):
	states_int = np.zeros([nstates,nspins])
	states_bool = np.zeros([nstates,nsites])

	states_int[0,:] = np.arange(0,nspins)
	states_bool[0,0:nspins] = 1

	for ii in range(1,nstates):
		if states_int[ii - 1, nspins - 1] < nsites -1:
			states_int[ii, nspins - 1] = states_int[ii - 1, nspins -1] + 1
			
			for kk in range(0,nspins-1):
				states_int[ii, kk] = states_int[ii-1, kk]
		
		else:
			jj = nspins - 2
			while states_int[ii - 1, jj] == states_int[ii - 1, jj + 1] - 1:
				jj = jj - 1
			states_int[ii, jj] = states_int[ii - 1, jj] + 1
			
			for kk in range(jj + 1, nspins):
				states_int[ii, kk] = states_int[ii, jj] + (kk - jj)
			for kk in range(0,jj):
				states_int[ii, kk] = states_int[ii - 1, kk]
		
		for ll in range(0,nspins):
			ind = int(states_int[ii,ll])
			states_bool[ii,ind] = 1
	return states_int, states_bool

def get_onespin_hopping_H(connect_mat, states_bool, nsites, nstates):
	ham = np.zeros([nstates, nstates])

	for ii in range(0, nstates):
		for jj in range(ii, nstates):
			states_xor = np.logical_xor(states_bool[ii,:],states_bool[jj,:]).astype(int)
			diff = np.sum(states_xor)
			if diff == 2:
				occ_sites = np.nonzero(states_xor)[0]
				first_site = occ_sites[0]
				second_site = occ_sites[1]
				if int(connect_mat[first_site,second_site]) == 1:
					ham[ii,jj] = 1;
					ham[jj,ii] = 1;
	return ham

if __name__=="__main__":
	
	start = time.time()
	
	print_results = True
	nsites = 14
	nspins = 7
	nstates = nchoosek(nsites, nspins)

	cmat = get_connect_mat(nsites)
	states, states_bool = get_state_reps(nsites, nspins, nstates)
	ham = get_onespin_hopping_H(cmat, states_bool, nsites, nstates)
	
	end = time.time()
	
	if print_results:
		print "connect mat:"
		print cmat
		print "states:"
		print states
		print states_bool
		print ham
	print "Time to run for %d sites and %d spins was %0.3f s" % (nsites, nspins, end-start)
			

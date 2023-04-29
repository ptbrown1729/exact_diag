## Exact diagonalization (ED) code for working with Fermion systems (spinless, Fermi-Hubbard) and spin models (Heisneberg, Ising, Rydberg Ising) on clusters
This project also includes tools for reducing the size of the matrices which must be
diagonalized by taking advantage of symmetries of the Hamiltonians. The approach adopted
here is described in detail in [ed-symmetries.tex](ed-symmetries.tex) and many useful references
can be found both here and in [bibliography.bib](bibliography.bib).

### [ed_base.py](ed_base.py)
This file defines a base class for use with different exact diagonalization models. This class defines
 basic functions such as taking expectation values, calculating partition functions, calculating Green's
 functions, etc.
 
### [ed_fermions.py](ed_fermions.py)
This file defines a class for working with Fermion systems, including spinless fermions of Fermi-Hubbard
type models. This class inherits from ed_base.py.

### [ed_spins.py](ed_spins.py)
This file defines a class for working with spin systems, including Heisenberg, Ising, and Rydberg Ising.
This class inherits from ed_base.py

### [ed_geometry.py](ed_geometry.py) 
This file implements a class for defining clusters of sites, either using a specific lattice setup or
an arbitrary geometry

### [ed_symmetry.py](ed_symmetry.py)
This file implements symmetry operations and functions for creating projectors for reducing the Hamiltonians
using these symmetry operations. This can allow diagonalization of much larger systems.

### [ed_nlce.py](ed_nlce.py)
This file implements cluster generation tools. Combining these clusters with the exact diagonalization code
allows an easy implementation of the numerical linked cluster expansion (NLCE)

### [fermi_gas.py](fermi_gas.py)
functions for evaluating properties of a non-interacting Fermi gas, including equation of state, compressibility,
spin correlations (for a two component gas), etc. Useful for comparing with Fermi-Hubbard model results.

### [greens_fns.py](greens_fns.py)
Some tools for working with Green's functions, and transforming between various representations

### [store_data.py](store_data.py)
unfinished script to save ED data to a SQL database

### [unittests](unittests)
This directory contains various unittests for comparing exact diagonalization results with published data,
and checking that the symmetry operators work appropriately.

### [scripts](examples)
This directory contains various examples which make use of the exact diagonalization code

### [cluster](cluster)
This directory contains code that was used run exact diagonalizations scripts on Princeton Physics Feynman cluster

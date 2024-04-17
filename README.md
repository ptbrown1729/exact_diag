# Exact diagonalization (ED) code for working with Fermion systems and spin models on lattices
The Fermion models supported include Fermi-Hubbard systems and spinless Fermions with nearest-neighbor
interactions. The spin models supported include the Heisenberg model, Ising model, and an extended
Rydberg Ising model relevant to rydberg atoms.

This project also includes tools for reducing the size of the matrices which must be
diagonalized by taking advantage of symmetries of the Hamiltonians. The approach adopted
here is described in detail in [ed-symmetries.tex](ed-symmetries.tex) and many useful references
can be found both here and in [bibliography.bib](bibliography.bib).

## Installation
```
git clone https://github.com/ptbrown1729/exact_diag.git
cd exact_diag
pip install .
```

If you would like to modify the code, then use the `-e` option so it is not necessary to reinstall
the package for your changes to take effect
```
pip install -e .
```

## Package contents

### [ed_base.py](exact_diag/ed_base.py)
This file defines a base class for use with different exact diagonalization models. This class defines
 basic functions such as taking expectation values, calculating partition functions, calculating Green's
 functions, etc.
 
### [ed_fermions.py](exact_diag/ed_fermions.py)
This file defines a class for working with Fermion systems, including spinless fermions of Fermi-Hubbard
type models. This class inherits from ed_base.py.

### [ed_spins.py](exact_diag/ed_spins.py)
This file defines a class for working with spin systems, including Heisenberg, Ising, and Rydberg Ising.
This class inherits from ed_base.py

### [ed_geometry.py](exact_diag/ed_geometry.py) 
This file implements a class for defining clusters of sites, either using a specific lattice setup or
an arbitrary geometry

### [ed_symmetry.py](exact_diag/ed_symmetry.py)
This file implements symmetry operations and functions for creating projectors for reducing the Hamiltonians
using these symmetry operations. This can allow diagonalization of much larger systems.

### [ed_nlce.py](exact_diag/ed_nlce.py)
This file implements cluster generation tools. Combining these clusters with the exact diagonalization code
allows an easy implementation of the numerical linked cluster expansion (NLCE)

### [fermi_gas.py](exact_diag/fermi_gas.py)
functions for evaluating properties of a non-interacting Fermi gas, including equation of state, compressibility,
spin correlations (for a two component gas), etc. Useful for comparing with Fermi-Hubbard model results.

### [greens_fns.py](exact_diag/greens_fns.py)
Some tools for working with Green's functions, and transforming between various representations

### [examples](examples)
This directory contains various examples which make use of the exact diagonalization code

### [unittests](unittests)
This directory contains various unittests for comparing exact diagonalization results with published data,
and checking that the symmetry operators work appropriately. To test
the code in `ed_geometry.py`, navigate to the `unittests` folder and run
```
python -m unittest geom_unittest.py 
```

### [cluster](cluster)
This directory contains code that was used run exact diagonalizations
scripts on Princeton Physics Feynman cluster using Slurm

## Documentation
Documentation is generated from function docstrings and the files in `doc` using 
[sphinx](https://www.sphinx-doc.org/en/master/). To build the html version, navigate to docs and run
```
make html
```
Then open `docs/_build/html/index.html` in your browser
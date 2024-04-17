from setuptools import setup, find_packages
from exact_diag import __version__

setup(
    name="exact_diag",
    version=__version__,
    description="A package for doing exact-diagonalization of Fermion and spin lattice models",
    long_description="",
    author='Peter T. Brown, qi2lab',
    author_email='ptbrown1729@gmail.com',
    packages=find_packages(include=['exact_diag', 'exact_diag.*']),
    python_requires='>=3.9',
    install_requires=["numpy",
                      "scipy"],
    extras_require={"display": ["matplotlib"]}
)
import numpy as np
import scipy.optimize
import warnings

def fermi(beta, mu, energy):
    """
    Fermi function, handling beta = 0 and infinity appropriately. Beta, mu, and energy should be provided in consistent
    energy units.

    This function is compatible with array broadcasting.
    :param beta: inverse temperature
    :param mu: chemical potential
    :param energy: energy
    :return:
    nf: fermi function at given beta, mu, energy
    """

    # ensure numpy arrays of at least 1 dimension
    beta = np.asarray(beta)
    if beta.ndim == 0:
        beta = beta[None]

    energy = np.asarray(energy)
    if energy.ndim == 0:
        energy = energy[None]

    mu = np.asarray(mu)
    if mu.ndim == 0:
        mu = mu[None]

    nf = np.divide(1, np.exp( beta * (energy - mu)) + 1)
    # handle zero temperature
    nf[np.logical_and(beta == np.inf, energy - mu <= 0)] = 1.
    nf[np.logical_and(beta == np.inf, energy - mu > 0)] = 0.
    # infinite temperature, nothing to do

    return nf

def fermi_derivatives(beta, mu, energy, eta=0.1):
    """
    Calculate the derivatives of the fermi function with respect to the various inputs. Beta, mu, and energy should be given in consistent units.

    :param beta: inverse temperature
    :param mu: chemical potential
    :param energy: energy
    :param eta: broadening parameter. TODO: Does this do anything?
    :return:
    dfdmu:
    dfde:
    dfdbeta:
    dfdT:
    """

    # ensure numpy arrays of at least 1 dimension
    beta = np.asarray(beta)
    if beta.ndim == 0:
        beta = beta[None]

    energy = np.asarray(energy)
    if energy.ndim == 0:
        energy = energy[None]

    mu = np.asarray(mu)
    if mu.ndim == 0:
        mu = mu[None]

    # compute derivative
    e_minus_mu = energy - mu
    beta_times_e_minus_mu = beta * e_minus_mu

    #dfdu = np.array( np.multiply(beta, np.divide(np.exp(beta_times_e_minus_mu), (np.exp(beta_times_e_minus_mu) + 1 ) ** 2 ) ))
    dfdx = -fermi(beta, mu, energy) * (1 - fermi(beta, mu, energy) )
    dfdu = -beta * dfdx
    dfde = beta * dfdx
    dfdbeta = (energy - mu) * dfdx

    # handle zero temperature. In this case, dfdu would be a delta function, because f is a step. To do numerical
    # calculations we must broaden the delta function by parameter eta
    # TODO: is this implemented correctly/what does it do?
    infinite_t_pts = np.logical_or(beta_times_e_minus_mu == np.inf, np.isnan(beta_times_e_minus_mu))
    if np.any(infinite_t_pts):
        dfdu[infinite_t_pts] = np.divide( np.pi / eta, e_minus_mu[infinite_t_pts] ** 2 + eta ** 2)
        dfde[infinite_t_pts] = np.divide( np.pi / eta, e_minus_mu[infinite_t_pts] ** 2 + eta ** 2)
        dfdbeta[infinite_t_pts] = np.divide(np.pi / eta, e_minus_mu[infinite_t_pts] ** 2 + eta ** 2)

    return dfdu, dfde, dfdbeta

def get_allowed_kvects(nx, ny, mode='balanced'):
    """
    Get k-vectors allowed on a periodic lattice of sizes Nx x Ny. 
    
    :param nx:
    :param ny:
    :param mode: either 'balanced' or
    :return:
    kxkx: grid of kx vectors of size Ny x Nx
    kyky: grid of ky vectors of size Ny x Nx
    dkx: kx  spacing
    dky: ky spacing
    """
    kxs = 2 * np.pi / nx * np.arange(0, nx)
    kys = 2 * np.pi / ny * np.arange(0, ny)
    dkx = 2 * np.pi / nx
    dky = 2 * np.pi / ny

    if mode == 'balanced':
        kxs[kxs >= np.pi] = kxs[kxs >= np.pi] - 2 * np.pi
        kys[kys >= np.pi] = kys[kys >= np.pi] - 2 * np.pi

        kxs.sort()
        kys.sort()
    elif mode == 'positive':
        pass
    else:
        raise Exception("mode should be either 'balanced' or 'positive' but was %s." % mode)

    kxkx, kyky = np.meshgrid(kxs, kys)

    return kxkx, kyky, dkx, dky

def get_energy_spacing(es):
    """
    Compute mean energy spacing from a 1D or 2D distribution of energies

    :param es: energies, array of arbitrary shape.
    :return:
    emin: minimum spacing between adjacent energies.
    emax: maximum spacing between adjacent energies.
    emean: mean spacing between adjacent energies.
    """

    es_list = np.sort(es.ravel())
    ediff = np.round(es_list[1:] - es_list[:-1], 12)
    ediff = ediff[ediff > 0]
    return np.min(ediff), np.max(ediff), np.mean(ediff)

def get_dos(es, e_end_pts):
    """
    Determine density of states in energy bins for given energy spectrum

    :param es:
    :param e_end_pts:
    :return:
    """
    e_means = 0.5 * (e_end_pts[1:] + e_end_pts[:-1])
    de = e_end_pts[1:] - e_end_pts[:-1]
    num_in_bin, _ = np.histogram(es.ravel(), e_end_pts)

    dos = np.divide(num_in_bin, de) / es.size
    return dos

def get_dos2(nsites=100, dim='2d'):
    # TODO: want function that will automatically choose good binning to get a smooth DOS
    if dim == '2d':
        kxkx, kyky, _, _ = get_allowed_kvects(nsites, nsites, 'balanced')
        dispersion = lambda kx, ky: -2 * (np.cos(kx) + np.cos(ky))
        es = dispersion(kxkx, kyky)
    elif dim == '1d':
        kxkx, _, _, _ = get_allowed_kvects(nsites, 1, 'balanced')
        dispersion = lambda kx: -2 * np.cos(kxkx)
        es = dispersion(kxkx)
    else:
        raise Exception("expected dim to be '1d' or '2d' but it was '%s'" % dim)

    # determine bin widths from energy spacing
    emin, emax, emean = get_energy_spacing(es)
    # empircally need something like this many points per bin to get smooth DOS
    pts_per_bin = 2000
    de = pts_per_bin * emean
    # determine bin number from bin widths
    nbins = np.floor( (np.max(es) - np.min(es)) / de)
    # ensure odd number of bins
    if nbins % 2 == 0:
        nbins = nbins + 1
    # bin edges and means
    e_bin_end_pts = np.linspace(np.min(es), np.max(es), nbins)
    e_means = 0.5 * (e_bin_end_pts[1:] + e_bin_end_pts[:-1])

    num_in_bin, _ = np.histogram(es.ravel(), e_bin_end_pts)

    dos = np.divide(num_in_bin, de) / es.size

    return dos, e_means

# green's functions
def fg_kspace_gfns(beta, mu, ks, time=0., type="greater", dim="2d"):
    """
    Calculate non-interacting gas green's function and their derivatives. All energies are in units of the tunneling energy.

    The various green's functions are defined as
    G_greater(k, t) = -i <c_k(t) c^\dag_k(0)>
    G_lesser(k, t) = i <c^\dag_k(0) c_k(t)>
    G_retarded(k, t) = -i * heaviside(t) * <[c_k(t), c^\dag_k(0)]_+>
    G_advanced(k, t) = i * heaviside(-t) * <[c_k(t), c^\dag_k(0)]_+>
    G_time-ordered(k, t) = -i * <T c_k(t) c^\dag_k(0)>

    :param beta: inverse temperature, in units of the tunneling energy.
    :param mu: chemical potential, in units of the tunneling energy.
    :param time: Time argument to Green's function. In units of hbar/t.
    :param nsites: Number of sites to use in the computation
    :param dim: dimensionality of gas: '1d' or '2d'
    :param type: type of Green's function to compute: "greater", "lesser", "retarded", "advanced" or "time-ordered"
    :return:
    gfn: Value of Green's function of given type at given parameters.
    dgdmu:
    """

    # TODO: handle either multiple beta/mus or multiple times
    # ensure numpy arrays of dim greater than zero
    beta = np.asarray(beta)
    if beta.ndim == 0:
        beta = beta[None]

    mu = np.asarray(mu)
    if mu.ndim == 0:
        mu = mu[None]

    ks = np.asarray(ks)
    if ks.ndim == 0:
        ks = ks[None]

    if dim == "1d":
        if ks.ndim != 1:
            raise Exception("Expected one dimensional k-vector, but ks had a different number of dimensions.")
    if dim == "2d":
        if ks.ndim == 1 and ks.size == 2:
            ks = ks[None, :]
        if ks.ndim != 2:
            raise Exception("Expected two dimensional k-vector, but ks had a different number of dimensions.")

    time = np.asarray(time)
    if time.ndim == 0:
        time = time[None]

    if dim == '2d':
        dispersion = lambda kx, ky: -2 * (np.cos(kx) + np.cos(ky))
        ek = dispersion(ks[:, 0], ks[:, 1])
    elif dim == '1d':
        dispersion = lambda kx: -2 * np.cos(kx)
        ek = dispersion(ks)
    else:
        raise Exception("dim should be either '1d' or '2d' but was %s." % dim)

    # calculations for various green's functions
    if type == "greater":
        gfn = -1j * np.exp( -1j * (ek - mu) * time) * (1. - fermi(beta, mu, ek))
        dfdmu = fermi_derivatives(beta, mu, ek)[0]
        dgdmu = time * np.exp( -1j * (ek - mu) * time) * (1. - fermi(beta, mu, ek)) + 1j * np.exp( -1j * (ek - mu) * time) * dfdmu

    elif type == "lesser":
        gfn = 1j * np.exp( -1j * (ek - mu) * time) * fermi(beta, mu, ek)
        dfdmu = fermi_derivatives(beta, mu, ek)[0]
        dgdmu = -time * np.exp( -1j * (ek - mu) * time) * fermi(beta, mu, ek) + 1j * np.exp( -1j * (ek - mu) * time) * dfdmu

    elif type == "retarded":
        # TODO: what is correct t=0 intepretation? I guess it should be taking limit from t>0?
        gfn = -1j * np.heaviside(time, 1.) * np.exp( -1j * (ek - mu) * time)
        dgdmu = time * np.heaviside(time, 1.) * np.exp( -1j * (ek - mu) * time)

    elif type == "advanced":
        gfn = 1j * np.heaviside(-time, 1.) * np.exp(-1j * (ek - mu) * time)
        dgdmu = -time * np.heaviside(-time, 1.) * np.exp(-1j * (ek - mu) * time)

    elif type == "time-ordered":
        gfn = -1j * np.exp( -1j * (ek - mu) * time) * (np.heaviside(time, 1.) * (1. - fermi(beta, mu, ek))
                                                     - np.heaviside(-time, 1.) * fermi(beta, mu, ek))
        # TODO: implement
        dgdmu = 0 * ek

    else:
        raise Exception("Expected type argument to be 'greater', 'lesser', 'retarded', 'advanced' or 'time-ordered', but it was '%s'." % type)

    return gfn, dgdmu

def fg_realspace_gfn(beta, mu, corr_index=[1,0], time=0., nsites=100, dim='2d', type="greater"):
    """
    Calculate real space green's functions G(t-t', r-r') by Fourier transforming momentum spacing functions.

    :param beta: inverse temperature
    :param mu: chemical potential
    :param corr_index: two component index which gives the separation r-r' in terms of lattice sites
    :param time: time, which gives t-t' in
    :param nsites: number of sites to use in k-space computation
    :param dim: Dimension of the fermi gas, either '1d' or '2d'
    :param type: 'greater', 'lesser', 'retarded', 'advanced', or 'time-ordered'
    :return:
    """

    # ensure numpy arrays of at least 1 dimension
    beta = np.asarray(beta)
    if beta.ndim == 0:
        beta = beta[None]

    mu = np.asarray(mu)
    if mu.ndim == 0:
        mu = mu[None]

    # ensure correct sizes
    if beta.size == 1 and mu.size > 1:
        beta = beta * np.ones(mu.shape)
    elif beta.size > 1 and mu.size == 1:
        mu = mu * np.ones(beta.shape)
    elif beta.size == mu.size:
        pass
    else:
        raise Exception('mu and beta must be the same size, or one of them should be a single number and the other an array.')

    # TODO: implement so when calculating multiple temperatures don't need to calculate k-vector each time

    if beta.size > 1:
        gfn = np.zeros(beta.shape, dtype=np.complex)
        dgdmu = np.zeros(beta.shape, dtype=np.complex)
        for ii in range(beta.size):
            coord = np.unravel_index(ii, beta.shape)
            gfn[coord], dgdmu[coord] = fg_realspace_gfn(beta[coord], mu[coord], corr_index, time, nsites=nsites, dim=dim, type=type)
    else:
        # TODO: use FFT instead?
        if dim == '2d':
            kxkx, kyky, dkx, dky = get_allowed_kvects(nsites, nsites, 'balanced')
            dispersion = lambda kx, ky: -2 * (np.cos(kx) + np.cos(ky))
            es = dispersion(kxkx, kyky)
            ft = np.exp(1j * kxkx * corr_index[0] + 1j * kyky * corr_index[1]).ravel()
            ks = np.concatenate((np.ravel(kxkx)[:, None], np.ravel(kyky)[:, None]), axis=1)
            gfn_k, dgdmu_k = fg_kspace_gfns(beta, mu, ks, time, type, dim)

        elif dim == '1d':
            kxkx, _, dkx, _ = get_allowed_kvects(nsites, 1, 'balanced')
            kxkx = kxkx.ravel()
            dispersion = lambda kx: -2 * np.cos(kxkx)
            es = dispersion(kxkx)
            ft = np.exp(1j * kxkx * corr_index[0])
            gfn_k, dgdmu_k = fg_kspace_gfns(beta, mu, kxkx, time, type, dim)

        else:
            raise Exception("expected dim to be '1d' or '2d' but it was '%s'" % dim)

        emin, emax, emean = get_energy_spacing(es)
        if emean > 1. / beta:
            warnings.warn("mean energy spacing is larger than temperature. Minimum energy spacing is %0.3e but temperature is %0.3e" % (emean, 1. / beta))

        # real space green's function is fourier transform of k-space green's function
        # 1 / N = dkx * dky / (2*np.pi)**2 for 2D or dkx / (2*np.pi) for 1D
        gfn = 1 / float(kxkx.size) * np.sum(ft * gfn_k)
        dgdmu = 1 / float(kxkx.size) * np.sum(ft * dgdmu_k)

    return gfn, dgdmu

# thermodynamic quantities
def fg_density(beta, mu, nsites=100, dim='2d'):
    """
    Compute the density of a Fermi gas in a lattice with tight-binding dispersion

    :param beta: temperature in units of the tunneling
    :param mu: chemical potential in units of the tunneling
    :param nsites: number of sites along each dimension to use in the calculation
    :param dim: '1d' or '2d'
    :return:
    density:
    """

    density, _ = fg_realspace_gfn(beta, mu, corr_index=[0, 0], time=0., nsites=nsites, dim=dim, type="lesser")
    density = np.abs(density)

    return density

def fg_mu(beta, density, nsites=100, dim='2d'):
    """
    Compute the chemical potential for a non-interacting Fermi gas with tightbinding dispersion at given temperature
    and density.
    :param beta:
    :param density:
    :param nsites:
    :param dim:
    :return:
    # TODO: this still fails at T = 0 and low T near half-filling
    """

    beta = np.asarray(beta)
    if beta.ndim == 0:
        beta = beta[None]

    density = np.asarray(density)
    if density.ndim == 0:
        density = density[None]

    if beta.size == 1 and density.size > 1:
        beta = beta * np.ones(density.shape)
    elif beta.size > 1 and density.size == 1:
        density = density * np.ones(beta.shape)
    elif beta.size == density.size:
        pass
    else:
        raise Exception(
            'n and beta must be the same size, or one of them should be a single number and the other an array.')

    if beta.size > 1:
        mu = np.zeros(beta.shape)
        for ii in range(beta.size):
            coord = np.unravel_index(ii, beta.shape)

            mu[coord] = fg_mu(beta[coord], density[coord], nsites, dim)
    else:
        min_fn = lambda mu: np.abs(fg_density(beta, mu, nsites, dim) - density)
        jacobian = lambda mu: fg_compressibility(beta, mu, nsites, dim)
        jacobian_fsqr = lambda mu: 2 * min_fn(mu) * jacobian(mu)

        # mu_guess = np.array([0.0])
        # result = scipy.optimize.minimize(min_fn, mu_guess)
        # mu = result.x

        if density == 0.5:
            mu = 0
        elif beta != np.inf:
            # seems to work more robustly than scipy.optimize.minimize
            # but fails at half-filling. jacobian_fsqr has a zero there...is that the reason?
            fit_handle = scipy.optimize.root_scalar(jacobian_fsqr, x0=-0.1, x1=0.1)
            mu = fit_handle.root
        else:
            # jacobian_fsqr = -inf here, which is why this fials otherwise...
            mu_guess = np.array([0.0])
            result = scipy.optimize.minimize(min_fn, mu_guess)
            mu = result.x

    return mu

def fg_compressibility(beta, mu, nsites=100, dim='2d'):
    """
    Compute the compressibility of a single component Fermi gas in a lattice with tight-binding dispersion
    :param beta: temperature in units of the tunneling
    :param mu: chemical potential in units of the tunneling
    :param nsites: number of sites along each dimension to use in the calculation
    :param dim: '1d' or '2d'
    :return:
    density
    TODO: doesn't work for zero temperature...
    """

    _, dgdmu = fg_realspace_gfn(beta, mu, corr_index=[0, 0], time=0., nsites=nsites, dim=dim, type="lesser")
    dndmu = np.abs(dgdmu)

    return dndmu

# correlators and response function
def fg_corr(beta, mu, corr_index=[1, 0], nsites=100, dim='2d'):
    """
    Compute the correlator <ni nj> - <n>**2 for a single component non-interacting fermi gas with a lattice dispersion.
    :param beta: Inverse temperature, in units of the hopping
    :param mu: Chemical potential, in units of the hopping.
    :param corr_index:
    :param nsites: Number of lattice sites in each dimension to be used in the calculation
    :return:
    """
    gfn_lesser = fg_realspace_gfn(beta, mu, corr_index, 0., nsites, dim, type="lesser")[0]
    delta = not np.any(corr_index)
    corr = delta * np.abs(gfn_lesser) - np.abs(gfn_lesser)**2

    return corr

def fg_density_response(beta, mu, k=np.array([0.,0.]), omega=0., eta=0., nsites=100, dim='2d'):
    # ensure numpy arrays of at least 1 dimension

    # TODO: is there a better way to do broadcasting etc?
    # TODO: allow finite omega
    # TODO: also get imaginary part

    beta = np.asarray(beta)
    if beta.ndim == 0:
        beta = beta[None]

    mu = np.asarray(mu)
    if mu.ndim == 0:
        mu = mu[None]

    k = np.asarray(k)
    if k.ndim == 0:
        k = k[None]

    # ensure correct sizes
    if beta.size == 1 and mu.size > 1:
        beta = beta * np.ones(mu.shape)
    elif beta.size > 1 and mu.size == 1:
        mu = mu * np.ones(beta.shape)
    elif beta.size == mu.size:
        pass
    else:
        raise Exception('mu and beta must be the same size, or one of them should be a single number and the other an array.')

    if dim == '2d':
        kxkx, kyky, _, _ = get_allowed_kvects(nsites, nsites, 'balanced')

        # get e_k+q
        kxkx_shift = kxkx + k[0]
        kyky_shift = kyky + k[1]

        # get energies
        dispersion = lambda kx, ky: -2 * (np.cos(kx) + np.cos(ky))
        es = dispersion(kxkx, kyky)
        es_q = dispersion(kxkx_shift, kyky_shift)
    elif dim == '1d':
        kxkx, _, _, _ = get_allowed_kvects(nsites, 1, 'balanced')
        kxkx_shift = kxkx + k[0]

        # get energies
        dispersion = lambda kx: -2 * np.cos(kxkx)
        es = dispersion(kxkx)
        es_q = dispersion(kxkx_shift)

    else:
        raise Exception()

    if beta.size > 1:
        chi_real = np.zeros(beta.shape)
        chi_imag = np.zeros(beta.shape)

        for ii in range(beta.size):
            coord = np.unravel_index(ii, beta.shape)
            chi_real[coord], chi_imag[coord] = fg_density_response(beta[coord], mu[coord], k, omega=omega, eta=eta, nsites=nsites, dim=dim)
    else:
        # this expression holds except where we have accidental degeneracies.
        chi_terms = np.divide(fermi(beta, mu, es) - fermi(beta, mu, es_q), omega + es - es_q + 1j * eta)
        # at those points, use L'Hopital's rule to get finite value
        lim = -fermi_derivatives(beta, mu, es, eta=0.0)[0]
        chi_terms[np.isnan(chi_terms)] = lim[np.isnan(chi_terms)]
        # sum them to get chi
        chi = np.sum(chi_terms) / kxkx.size

        chi_real = np.real(chi)
        chi_imag = np.imag(chi)

    return chi_real, chi_imag

def fg_magnetic_response(beta, mu, k, omega=0., eta=0., nsites=100, dim='2d'):
    """
    Fermi gas magnetic response function
    :param beta:
    :param mu:
    :param k:
    :param omega:
    :param nsites:
    :param dim:
    :return:
    """
    return fg_density_response(beta, mu, k, omega, eta, nsites, dim)

# non-interacting fg of two species
def fg_singles(beta, mu_up, mu_dn, nsites=100, dim='2d'):
    """
    Compute singles density for a two-component non-interacting Fermi gas

    :param beta:
    :param mu_up:
    :param mu_dn:
    :param nsites:
    :param dim:
    :return:
    """
    beta = np.asarray(beta)
    if beta.ndim == 0:
        beta = beta[None]

    mu_up = np.asarray(mu_up)
    if mu_up.ndim == 0:
        mu_up = mu_up[None]

    mu_dn = np.asarray(mu_dn)
    if mu_dn.ndim == 0:
        mu_dn = mu_dn[None]

    n_up = fg_density(beta, mu_up, nsites, dim)
    n_dn = fg_density(beta, mu_dn, nsites, dim)

    singles = n_up + n_dn - 2 * n_up * n_dn
    return singles

def fg_singles_corr(beta, mu_up, mu_dn, corr_index=[0, 1], nsites=100, dim='2d'):
    """
    Compute correlations between singles for a two-component non-interacting Fermi gas

    < [nup(i) + ndn(i) - 2 * nup(i) * ndn(i)] [nup(j) + ndn(j) - 2 * nup(j) * ndn(j)]>
    = < nup(i) nup(j)> + <ndn(i) ndn(j)> + 4 <d(i) d(j)> ...
      - 2 < nup(i) d(j)> - 2 < ndn(i) d(j)> - 2 <d(i) nup(j)> - 2 < d(i)ndn(j)>

    We also evaluate the <n_i_up d_j> term using Wick's theorem.
    Only one term contributes ...
    < nup(i) d(j)> = - <c^dag_up(i) c_up(j)><c^dag_j_upc_i_up><c^dag_j_down c_j_down> ...
    = <nup(i) nup(j)>_c <ndn(j)>

    :param beta:
    :param mu_up:
    :param mu_dn:
    :param corr_index:
    :param nsites:
    :param dim:
    :return:
    """
    beta = np.asarray(beta)
    if beta.ndim == 0:
        beta = beta[None]

    mu_up = np.asarray(mu_up)
    if mu_up.ndim == 0:
        mu_up = mu_up[None]

    mu_dn = np.asarray(mu_dn)
    if mu_dn.ndim == 0:
        mu_dn = mu_dn[None]

    n_up = fg_density(beta, mu_up, nsites, dim)
    n_dn = fg_density(beta, mu_dn, nsites, dim)
    n_up_corr = fg_corr(beta, mu_up, corr_index, nsites, dim)
    n_dn_corr = fg_corr(beta, mu_dn, corr_index, nsites, dim)

    singles_corr = n_up_corr * (1 + 4 * n_dn ** 2 - 4 * n_dn) \
                 + n_dn_corr * (1 + 4 * n_up ** 2 - 4 * n_up)\
                 + 4 * n_up_corr * n_dn_corr

    return singles_corr

def fg_doubles(beta, mu_up, mu_dn, nsites=100, dim='2d'):
    """
    Compute doubles density for a two-component non-interacting Fermi gas

    :param beta:
    :param mu_up:
    :param mu_dn:
    :param nsites:
    :param dim:
    :return:
    """
    beta = np.asarray(beta)
    if beta.ndim == 0:
        beta = beta[None]

    mu_up = np.asarray(mu_up)
    if mu_up.ndim == 0:
        mu_up = mu_up[None]

    mu_dn = np.asarray(mu_dn)
    if mu_dn.ndim == 0:
        mu_dn = mu_dn[None]

    n_up = fg_density(beta, mu_up, nsites, dim)
    n_dn = fg_density(beta, mu_dn, nsites, dim)

    d = n_up * n_dn

    return d

def fg_doubles_corr(beta, mu_up, mu_dn, corr_index=[0, 1], nsites=100, dim='2d'):
    """
    Compute correlations between doubles for a two-component non-interacting Fermi gas

   Wick's theorem according to
   <d(i) d(j)>_c = n_up^2 * <n_up(i) n_up(j)>_c
                   n_dn^2 * <n_dn(i) n_dn(j)>_c
                   + <n_up(i) n_up(j)>_c * <n_dn(i) n_dn(j)>_c

    :param beta:
    :param mu_up:
    :param mu_dn:
    :param corr_index:
    :param nsites:
    :param dim:
    :return:
    """
    beta = np.asarray(beta)
    if beta.ndim == 0:
        beta = beta[None]

    mu_up = np.asarray(mu_up)
    if mu_up.ndim == 0:
        mu_up = mu_up[None]

    mu_dn = np.asarray(mu_dn)
    if mu_dn.ndim == 0:
        mu_dn = mu_dn[None]

    n_up = fg_density(beta, mu_up, nsites, dim)
    n_dn = fg_density(beta, mu_dn, nsites, dim)
    n_up_corr = fg_corr(beta, mu_up, corr_index, nsites, dim)
    n_dn_corr = fg_corr(beta, mu_dn, corr_index, nsites, dim)

    doubles_corr = n_up ** 2 * n_dn_corr + n_dn ** 2 * n_up_corr + n_up_corr * n_dn_corr

    return doubles_corr

def fg_sz_corr(beta, mu_up, mu_dn, corr_index=[0, 1], nsites=100, dim='2d'):
    """
    4*<S^z S^z>_c = <(n_up - n_dn)*(n_up - n_dn)> spin correlations for Fermi gas
    :param beta:
    :param mu_up:
    :param mu_dn:
    :param corr_index:
    :param nsites:
    :param dim:
    :return:
    """
    n_up_corr = fg_corr(beta, mu_up, corr_index, nsites, dim)
    n_dn_corr = fg_corr(beta, mu_dn, corr_index, nsites, dim)
    return n_up_corr + n_dn_corr

def fg_sx_corr(beta, mu_up, mu_dn, corr_index=[0, 1], nsites=100, dim='2d'):
    """
    Sx or Sy spin correlations for Fermi gas
    :param beta:
    :param mu_up:
    :param mu_dn:
    :param corr_index:
    :param nsites:
    :param dim:
    :return:
    """
    gfn_lesser_up = fg_realspace_gfn(beta, mu_up, corr_index, 0., nsites, dim, type="lesser")[0]
    gfn_lesser_dn = fg_realspace_gfn(beta, mu_dn, corr_index, 0., nsites, dim, type="lesser")[0]
    delta = not np.any(corr_index)
    corr_sx = np.real(delta * (np.abs(gfn_lesser_up) + np.abs(gfn_lesser_dn)) -gfn_lesser_up * gfn_lesser_dn.conj() - gfn_lesser_dn * gfn_lesser_up.conj())

    return corr_sx

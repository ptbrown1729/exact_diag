"""
Tools for working with Green's functions
"""
import numpy as np
import warnings

def get_matsubara_frqs(beta, max_frq=30, format="fermion"):
    """
    Calculate matsubara frequencies up to a cutoff
    :param beta: inverse temperature
    :param max_frq: maximum frequency that could be returend
    :param format: "fermion" for fermionic matsubara frequencies or "boson" for bosonic matsubara frequencies
    :return:
    """
    if format == "fermion":
        matsubara_frq_smallest = np.pi / beta
    elif format == "boson":
        matsubara_frq_smallest = 2 * np.pi / beta
    else:
        raise Exception("format must be 'fermion' or 'boson', but %s was supplied", format)

    delta_matsubara_frq = 2 * np.pi / beta
    matsubara_frqs = np.arange(matsubara_frq_smallest, max_frq, delta_matsubara_frq)
    matsubara_frqs = np.concatenate((-matsubara_frqs, matsubara_frqs))
    if format == "boson":
        matsubara_frqs = np.concatenate((matsubara_frqs, np.array([0])))

    return np.sort(matsubara_frqs)

# compute imaginary time or frequency green's functions from the spectral fn
def get_grnfn_imagfrq_spectralfn(matsubara_frqs, omegas, spectral_fn):
    """
    Compute the imaginary frequency green's function associated with a given spectral function.
    G(i*omega_n) = 1 / (2*pi) \int A(k, w) / (i * \omega_n - w).

    Here we assume that the spectral function is normalized such that \int dw A(k, w) = 2*pi
    :param matsubara_frqs:
    :param omegas: frequency data
    :param spectral_fn: spectral function evaluate at the omegas
    """

    #matsubara_frqs = get_matsubara_frqs(beta, format="fermion")

    matsubara_frqs_expanded = np.repeat(matsubara_frqs[:, None], len(omegas), 1)
    spectral_fn_expanded = np.repeat(spectral_fn[None, :], len(matsubara_frqs), 0)
    omega_expanded = np.repeat(omegas[None, :], len(matsubara_frqs), 0)

    # calculate imaginary frequency green's function
    integrand = np.divide(spectral_fn_expanded, 1j * matsubara_frqs_expanded - omega_expanded)
    gfn_imag_frq = np.trapz(integrand, omega_expanded, axis=1) / (2 * np.pi)

    return gfn_imag_frq

def get_grnfn_imagtime_spectralfn(taus, beta, omegas, spectral_fn, format="fermion"):
    """
    Compute the imaginary time green's function associated with a certain spectral function.
    G(\tau) = - \int A(k,w) * exp(-\tau * w) / ( 1 + exp(-\beta * w))

    Here we assume that the spectral function is normalized such that \int dw A(k, w) = 2*pi
    :param taus:
    :param beta: the inverse temperature
    :param omegas:
    :param spectral_fn:
    :param format: Either "fermion" or "boson" for the appropriate type of green's function
    :return:
    """
    # taus = np.arange(0.0, num_taus)
    # # TODO: I think it makes more sense to take the taus as an argument ...
    # if include_beta:
    #     taus = taus * beta / float(num_taus - 1)
    # else:
    #     taus = taus * beta / float(num_taus)

    omegas_expanded, tau_expanded = np.meshgrid(omegas, taus)
    # tau_expanded = np.repeat(taus[:, None], len(omegas), 1)
    # omegas_expanded = np.repeat(omegas[None, :], len(taus), 0)
    spectral_fn_expanded = np.repeat(spectral_fn[None, :], len(taus), 0)

    if format == "fermion":
        factor = 1.0
    elif format == "boson":
        factor = -1.0
    else:
        raise Exception("format must be boson or fermion, but was %s" % format)

    integrand = np.divide(np.multiply(spectral_fn_expanded, np.exp(- np.multiply(omegas_expanded, tau_expanded))),
                                      1 + factor * np.exp(- beta * omegas_expanded))
    #a = np.divide(np.exp(-omegas_expanded * tau_expanded), 1 + factor * np.exp(- beta * omegas_expanded))
    gfn_imag_time = -np.trapz(integrand, omegas_expanded, axis = 1) / (2 * np.pi)

    return gfn_imag_time

# alternatively, get kernels
def get_grnfn_imagfrq_spectralfn_kernel(matsubara_frqs, omegas, format="fermion"):
    """
    Produce kernel such that G_n = G(iw_n) = K * A(w_m). This multiplication is equivalent to supplying spectral_fn to
    get_frnfn_imagfrq_spectralfn.
    :param matsubara_frqs: imaginary frequencies
    :param omegas: real frequencies
    :param format:
    :return: kernel, a mabsubara_frqs.size x omegas.size array
    """

    omegas_expanded, matsubara_frqs_expanded = np.meshgrid(omegas, matsubara_frqs)

    # frequency differences for integration part
    frq_diffs = np.zeros(omegas_expanded.shape)
    frq_diffs[:, 1:-1] = 0.5 * (omegas_expanded[:, 2:] - omegas_expanded[:, :-2])
    frq_diffs[:, 0] = 0.5 * (omegas_expanded[:, 1] - omegas_expanded[:, 0])
    frq_diffs[:, -1] = 0.5 * (omegas_expanded[:, -1] - omegas_expanded[:, -2])
    # normalization
    norm_factor = 1 / (2 * np.pi)
    kernel = norm_factor * np.divide(1, 1j * matsubara_frqs_expanded - omegas_expanded) * frq_diffs

    return kernel

def get_grnfn_imagtime_spectralfn_kernel(taus, beta, omegas, format="fermion"):
    """
    Compute the kernel which related G_k = G(tau_k) = K * A_n = K * A(\omega_n). This multiplication is equivalent to
    supplying the spectral function to get_grnfn_imagtime_spectralfn.
    :param taus: Imaginary time points to evaluate the kernel at
    :param beta: Inverse temperature
    :param omegas: Real frequency
    :param format: "fermion" for fermionic kernel or "boson" for bosonic kernel
    :return: kernel, a taus.size x omegas.size matrix
    """
    omegas_expanded, tau_expanded = np.meshgrid(omegas, taus)

    if format == "fermion":
        factor = 1.0
    elif format == "boson":
        factor = -1.0
    else:
        raise Exception("format must be boson or fermion, but was %s" % format)

    # frequency differences for integration part
    frq_diffs = np.zeros(omegas_expanded.shape)
    frq_diffs[:, 1:-1] = 0.5 * (omegas_expanded[:, 2:] - omegas_expanded[:, :-2])
    frq_diffs[:, 0] = 0.5 * (omegas_expanded[:, 1] - omegas_expanded[:, 0])
    frq_diffs[:, -1] = 0.5 * (omegas_expanded[:, -1] - omegas_expanded[:, -2])
    # normalization
    norm_factor = - 1 / (2*np.pi)
    # full kernel
    kernel = norm_factor * np.divide(np.exp(- omegas_expanded * tau_expanded),
                                     1 + factor * np.exp(-beta * omegas_expanded)) * frq_diffs

    return kernel

# fourier transform, convert between imaginary time and frequency
def grnfn_imagtime_ft(matsubara_frqs, taus, grn_fn_imagtime, beta):
    """
    Fourier transform an imaginary time green's function to produce the imaginary frequency version
    :param matsubara_frqs: matsubara frequencies to evaluate the fourier transform at
    :param taus: imaginary time points
    :param grn_fn_imagtime: imaginary time green's function evaluated at taus
    :param beta:
    :return: grnfn_imagefrq
    """

    # TODO: it seems that there are some subtle issues about the maximum matsubara frequencies that we can get
    # information about for a given dtau. Understand these and add a warning in this function.

    if not np.any(np.round(np.abs(taus - beta), 14) == 0):
        warnings.warn('tau = beta was not supplied to grnfn_imagtime_ft.')

    if np.round(taus, 14).max() > beta or np.round(taus, 14).min() < 0:
        warnings.warn('taus supplied to grnfn_imagtime_ft contained values smaller than 0 or'
                      ' larger than beta. Those values will be ignored in the calculation',
                      RuntimeWarning)
        indices_to_use = np.logical_and(np.round(taus, 14) >= 0, np.round(taus, 14) <= beta)
        grn_fn_imagtime = grn_fn_imagtime[indices_to_use]
        taus = taus[indices_to_use]

    matsubara_frqs_expanded, taus_expanded = np.meshgrid(matsubara_frqs, taus)
    grn_fn_expanded = np.repeat(grn_fn_imagtime[:, None], matsubara_frqs.size, axis=1)

    # integrand for ft
    integrand = np.multiply(grn_fn_expanded, np.exp(1j * matsubara_frqs_expanded * taus_expanded))

    # integrate
    grn_fn_imagfrq = np.trapz(integrand, taus, axis=0)

    return grn_fn_imagfrq

def grnfn_imagfrq_ft(taus, matsubara_frqs, grn_fn_imagfrq, beta):
    """
    Fourier transform an imaginary frequency green's function to produce the imaginary time version
    :param taus: imaginary time points to evaluate FT at
    :param matsubara_frqs: matsubara frequency for imaginary frequency greens function
    :param grn_fn_imagfrq: imaginary frequency greens function evaluated at the matsubara_frqs
    :param beta: inverse temperature
    :return: grnfn_imagtime
    """

    taus_expanded, matsubara_frqs_expanded = np.meshgrid(taus, matsubara_frqs)
    grn_fn_expanded = np.repeat(grn_fn_imagfrq[:, None], taus.size, axis=1)

    # integrand for fourier transform
    summand = np.multiply(grn_fn_expanded, np.exp(- 1j * matsubara_frqs_expanded * taus_expanded))

    # integrate
    grnfn_imagtime = np.sum(summand, 0) / beta

    return grnfn_imagtime

# routines to simulate a non-interacting gas
def get_noninteracting_gfn_tau(taus, epsilon_k, beta, format="fermion"):
    """
    Compute the Green's function
    :param taus: imaginary times to evaluate the function at. These must be in the interval [-beta, beta]
    :param epsilon_k: energy dispersion evaluated at momentum k
    :param beta: inverse temperature
    :param format: "fermion" or "boson"
    :return:
    """

    if format == "fermion":
        fermi_fn = np.divide(1, 1 + np.exp(beta * epsilon_k))
        # TODO: should I enforce one of the heaviside functions to be 1 at \tau = 0 ?
        greens_fn = - np.exp(-epsilon_k * taus) * ((1 - fermi_fn) * np.heaviside(taus, 0) -
                                                   fermi_fn * np.heaviside(-1 * taus, 0))
    elif format == "bosone":
        raise Exception("TODO: noninteracting boson green's function calculation not implemented yet")
    else:
        raise Exception('Format strings for get_non_interacting_fn_tau must be boson on fermion')

    return greens_fn

def get_noninteracting_gfn_imagfrq(max_frq_index, epsilon_k, beta, format="fermion"):
    """
    Compuate the imaginary frequency green's function for a noninteracting problem
    :param max_frq_index:
    :param epsilon_k:
    :param beta:
    :param format:
    :return:
    """

    if format == "fermion":
        omega_matsubara = get_matsubara_frqs(beta, max_frq=max_frq_index, format=format)
        greens_fn = np.divide(1, 1j * omega_matsubara - epsilon_k)
    elif format == "boson":
        raise Exception("TODO: noninteracting boson green's function calculation not implemented yet")
    else:
        raise Exception('Format strings for get_non_interacting_fn_tau must be boson on fermion')

    return greens_fn, omega_matsubara

def get_noninteracting_spectral_fn(omegas, epsilon_k, eta):
    """

    :param omegas:
    :param epsilon_k:
    :param eta: broadening parameter for lorentzian
    :return:
    """

    # the lorentzian expression is normalized so that its integral is one.
    spectral_fn = 2 * np.pi * np.divide(eta / np.pi, eta ** 2 + (omegas - epsilon_k)**2)
    return spectral_fn

# linear response functions
def chi_hydro(ws, k, chi0, d0, d2=0, gamma=np.inf):
    """
    Hydrodynamic susceptibility function
    :param ws:
    :param k:
    :param chi0:
    :param d0:
    :param d2:
    :param gamma:
    :return:
    """
    if gamma != np.inf:
        chi = np.divide(chi0, 1. - 1j * ws / (d0 * k ** 2 + d2 * k ** 4) - ws ** 2 / (gamma * (d0 * k ** 2 + d2 * k ** 4)))
    else:
        chi = np.divide(chi0, 1. - 1j * ws / (d0 * k ** 2 + d2 * k ** 4))

    return chi

def field_snap_off_w(ws, ho, eta=0.1):
    """
    Field frequency dependence for h(t) = theta(t) * Ho
    :param ws:
    :param ho:
    :param eta: broadening parameter
    :return:
    """
    h_w = np.divide(ho, 1j * ws + eta)
    return h_w

def phi_t(ts, ws, chi_ws):
    """
    Get real time response function from real frequency
    :param ts:
    :param ws:
    :param chi_ws:
    :return:
    """
    wsws, tsts = np.meshgrid(ws, ts)
    chichi, _ = np.meshgrid(chi_ws, ts)

    phi_t = 1 / (2 * np.pi) * np.trapz(chichi * np.exp(-1j * wsws * tsts), ws, axis=1)
    return phi_t

def obs_kt(ts, ws, chi_ws, field_ws):
    """
    Get observable response <O(t)> = \int dw e^{-iwt)* chi(w) * H(w)
    :param ts:
    :param ws:
    :param chi_ws:
    :param field_ws:
    :return:
    """
    wsws, tsts = np.meshgrid(ws, ts)
    chichi, _ = np.meshgrid(chi_ws, ts)
    ff, _ = np.meshgrid(field_ws, ts)

    obs_kt = 1 / (2*np.pi) * np.trapz(chichi * ff * np.exp(-1j * wsws * tsts), ws, axis=1)
    return obs_kt


if __name__ == "__main__":
    # from greens_fns import *
    import matplotlib.pyplot as plt

    # test using non-interacting green's function
    beta = 1.
    mu = 2.
    # imaginary times
    dtau = 0.02
    taus = np.arange(-beta, beta + dtau, dtau)
    taus_half = np.arange(0, beta + dtau, dtau)
    # imaginary frequencies
    max_m_frq = 1000
    omegas_m = get_matsubara_frqs(beta, max_m_frq, format="fermion")
    # real frequency
    omegas = np.linspace(-5, 5, 5000)
    eta = 0.01

    # num lattice points
    n_latt_pts = 30.
    ks = 2 * np.pi * np.arange(0, n_latt_pts - 1) / n_latt_pts

    # dispersion
    epsilon_k = lambda k: 2 * (1 - np.cos(k))

    # evaluate for all k's
    gfn_k_taus = np.zeros((taus.size, ks.size))
    gfn_k_omegas = np.zeros((omegas_m.size, ks.size), dtype=np.complex)
    spectral_fn = np.zeros((omegas.size, ks.size))
    for ii, k in enumerate(ks):
        gfn_k_taus[:, ii] = get_noninteracting_gfn_tau(taus, epsilon_k(k) - mu, beta, format="fermion")
        gfn_k_omegas[:, ii], _ = get_noninteracting_gfn_imagfrq(max_m_frq, epsilon_k(k) - mu, beta, format="fermion")
        spectral_fn[:, ii] = get_noninteracting_spectral_fn(omegas, epsilon_k(k) - mu, eta)

    # calculate greens functions from spectral function
    gfn_k_taus_from_spectralfn = np.zeros((taus_half.size, ks.size))
    gfn_k_omegas_from_spectralfn = np.zeros((omegas_m.size, ks.size), dtype=np.complex)
    kernel_imagtime = get_grnfn_imagtime_spectralfn_kernel(taus_half, beta, omegas, format="fermion")
    kernel_imagfrq = get_grnfn_imagfrq_spectralfn_kernel(omegas_m, omegas)
    for ii, k in enumerate(ks):
        gfn_k_taus_from_spectralfn[:, ii] = get_grnfn_imagtime_spectralfn(taus_half, beta, omegas,
                                                                          spectral_fn[:, ii], format="fermion")
        gfn_k_omegas_from_spectralfn[:, ii] = get_grnfn_imagfrq_spectralfn(omegas_m, omegas, spectral_fn[:, ii])

    # calculate imaginary time/frequency greens functions from fourier transform of the other
    gfn_k_taus_from_imagfrq = np.zeros((taus.size, ks.size))
    gfn_k_omegas_from_imagtime = np.zeros((omegas_m.size, ks.size), dtype=np.complex)
    for ii, k in enumerate(ks):
        gfn_k_taus_from_imagfrq[:, ii] = grnfn_imagfrq_ft(taus, omegas_m, gfn_k_omegas[:, ii], beta)
        gfn_k_omegas_from_imagtime[:, ii] = grnfn_imagtime_ft(omegas_m, taus, gfn_k_taus[:, ii], beta)

    # plot non-interacting results
    fig_handle = plt.figure()

    plt.subplot(2, 2, 1)
    plt.plot(taus, gfn_k_taus)
    plt.grid()
    plt.xlabel('tau')
    plt.ylabel('G(tau)')
    plt.title('Imaginary time Greens functions')

    plt.subplot(2, 2, 2)
    plt.plot(omegas, spectral_fn)
    plt.grid()
    plt.xlabel('omega_start')
    plt.ylabel('A(w)')
    plt.title('Spectral functions')

    plt.subplot(2, 2, 3)
    plt.plot(omegas_m, gfn_k_omegas.real, '.')
    plt.grid()
    plt.xlabel('matsubara frequency')
    plt.ylabel('G(iw), real part')
    plt.title('Imaginary frequency Greens functions, real part')

    plt.subplot(2, 2, 4)
    plt.plot(omegas_m, gfn_k_omegas.imag, '.')
    plt.grid()
    plt.xlabel('matsubara frequency')
    plt.ylabel('G(iw), imag part')
    plt.title('Imaginary frequency Greens functions, imag part')

    fig_handle2 = plt.figure()
    index = 1

    plt.subplot(2, 2, 1)
    plt.plot(taus, gfn_k_taus[:, index], '.')
    plt.plot(taus_half, gfn_k_taus_from_spectralfn[:, index], '.')
    plt.grid()
    plt.xlabel('tau')
    plt.ylabel('G(tau)')
    plt.legend(['analytic fn', 'inferred from spectral fn'])
    plt.title('Imaginary time')

    plt.subplot(2, 2, 3)
    plt.plot(omegas_m, gfn_k_omegas[:, index].real, '.')
    plt.plot(omegas_m, gfn_k_omegas_from_spectralfn[:, index].real, '.')
    plt.grid()
    plt.xlabel('Matsubara frequency')
    plt.ylabel('G(iw)')
    plt.legend(['analytic fn', 'inferred from spectral fn'])
    plt.title('Imaginary frequency, real part')

    plt.subplot(2, 2, 4)
    plt.plot(omegas_m, gfn_k_omegas[:, index].imag, '.')
    plt.plot(omegas_m, gfn_k_omegas_from_spectralfn[:, index].imag, '.')
    plt.grid()
    plt.xlabel('Matsubara frequency')
    plt.ylabel('G(iw)')
    plt.legend(['analytic fn', 'inferred from spectral fn'])
    plt.title('Imaginary frequency, imaginary part')

    plt.subplot(2, 2, 2)
    plt.plot(omegas, spectral_fn[:, index], '.')
    plt.grid()
    plt.xlabel('freuqency')
    plt.ylabel('A(w)')
    plt.title('spectral function')

    plt.suptitle('Greens functions from spectral function, non-interacting gas')

    fig_handle3 = plt.figure()
    index = 3

    plt.subplot(2, 2, 1)
    plt.plot(taus, gfn_k_taus[:, index], '.')
    plt.plot(taus, gfn_k_taus_from_imagfrq[:, index], '.')
    plt.grid()
    plt.xlabel('tau')
    plt.ylabel('G(tau)')
    plt.legend(['analytic fn', 'ft G(iw)'])
    plt.title('Imaginary time Greens functions')

    plt.subplot(2, 2, 3)
    plt.plot(omegas_m, gfn_k_omegas[:, index].real, '.')
    plt.plot(omegas_m, gfn_k_omegas_from_imagtime[:, index].real, '.')
    plt.grid()
    plt.xlabel('Matsubara frequency')
    plt.ylabel('G(iw)')
    plt.legend(['analytic fn', 'ft G(tau)'])
    plt.title('Imaginary frequency, real part')

    plt.subplot(2, 2, 4)
    plt.plot(omegas_m, gfn_k_omegas[:, index].imag, '.')
    plt.plot(omegas_m, gfn_k_omegas_from_imagtime[:, index].imag, '.')
    plt.grid()
    plt.xlabel('Matsubara frequency')
    plt.ylabel('G(iw)')
    plt.legend(['analytic fn', 'ft G(tau)'])
    plt.title('Imaginary frequency, imaginary part')

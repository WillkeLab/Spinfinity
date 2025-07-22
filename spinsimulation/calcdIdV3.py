import numpy as np
import scipy as sp
import scipy.constants as const
from spinsimulation.SpinSys import SpinSys
import spinsimulation.SpinHamiltonian as hamil 
import calcdIdV as dIdV
import matplotlib.pyplot as plt

def e_epsilon(SpinSys: SpinSys):
    """
    Compute third-order spin ε-tensors for all tunneling directions.
    Returns:
        e, er, en, enr : np.ndarray
            4×4×4 tensors representing tunneling pathways.
    """
    # Transport operators for tip → sample
    [x,y,z,u] = e_transport(SpinSys)
    tempSpinSys=SpinSys
    tempSpinSys.ptip=tempSpinSys.psample
    [xs,ys,zs,us] = e_transport(tempSpinSys)

    # so1 = tip, so2 = sample, so3 = conjugate of tip
    so1 = [x, y, z, u]              # list of 4 matrices: [Sx, Sy, Sz, S1]
    so2 = [xs,ys,zs,us]           # list of 4 matrices
    so3 = [x.conj().T, y.conj().T, z.conj().T, u.conj().T]  # Hermitian conjugates
    mone = np.ones((1, 2)) # Spin-½ basis vector |↑⟩

    e_tensor   = np.zeros((4, 4, 4), dtype=complex)
    er_tensor  = np.zeros_like(e_tensor)
    en_tensor  = np.zeros_like(e_tensor)
    enr_tensor = np.zeros_like(e_tensor)

    for j in range(4):
        for k in range(4):
            for l in range(4):
                for state in [0, 1]:
                    term_e = (
                        (mone.T * so3[l][state, :][np.newaxis, :]) *
                        so2[k].T *
                        (mone * so1[j][:, state][:, np.newaxis])
                    )
                    e_tensor[j, k, l] += np.sum(term_e)

                    term_er = -(
                        (mone.T * so3[l][state, :][np.newaxis, :]) *
                        (mone * so1[k][:, state][:, np.newaxis]) *
                        so2[j].T
                    )
                    er_tensor[j, k, l] += np.sum(term_er)

                    term_en = -(
                        (mone * so1[l][:, state][:, np.newaxis]) *
                        so2[k].T *
                        (mone.T * so3[j][state, :][np.newaxis, :])
                    )
                    en_tensor[j, k, l] += np.sum(term_en)

                    term_enr = (
                        (mone * so1[l][:, state][:, np.newaxis]) *
                        (mone.T * so3[k][state, :][np.newaxis, :]) *
                        so2[j].T
                    )
                    enr_tensor[j, k, l] += np.sum(term_enr)

    return e_tensor, er_tensor, en_tensor, enr_tensor


def e_transport(SpinSys: SpinSys):
    """
    Compute spin-½ transition matrices from ptip to psample.

    Returns:
        [x, y, z, u] : list of 2×2 complex matrices.
    """
    # Pauli matrices
    sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
    sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
    identity = np.eye(2, dtype=complex)

    # Density matrices
    rho_tip = dmatrix(SpinSys.ptip)
    rho_sample = dmatrix(SpinSys.psample)

    # Eigendecompose
    evals_tip, evecs_tip = np.linalg.eigh(rho_tip)
    evals_sample, evecs_sample = np.linalg.eigh(rho_sample)

    pt = np.real(evals_tip)     # ✅ Correct
    ps = np.real(evals_sample)

    pf_pi_sqrt = np.sqrt(np.outer(ps, pt))  # shape (2, 2)

    # Transform operators
    x = (evecs_sample.conj().T @ (2 * sigma_x) @ evecs_tip) * pf_pi_sqrt / 2
    y = (evecs_sample.conj().T @ (2 * sigma_y) @ evecs_tip) * pf_pi_sqrt / 2
    z = (evecs_sample.conj().T @ (2 * sigma_z) @ evecs_tip) * pf_pi_sqrt / 2
    u = 2 * (evecs_sample.conj().T @ identity @ evecs_tip) * pf_pi_sqrt

    return [x, y, z, u]


def dmatrix(p):
    """
    Construct a 2×2 spin-½ density matrix from a polarization vector.

    Parameters
    ----------
    p : array_like
        3-element vector representing spin polarization (components between -1 and 1)

    Returns
    -------
    rho : np.ndarray
        2×2 complex density matrix
    """
    p = np.array(p, dtype=float)
    sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
    sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
    identity = np.eye(2, dtype=complex)

    rho = 0.5 * (identity + p[0]*sigma_x + p[1]*sigma_y + p[2]*sigma_z)
    return rho


def compute_third_order_rates(SpinSys: SpinSys, initial_state_idx):
    """
    Compute third-order tunneling rates for IETS.

    Parameters
    ----------
    initial_state_idx : int
        Index of the initial state.
    spinsys : SpinSys
        Object containing spin operator matrices (Sx, Sy, Sz).

    Returns
    -------
    rate_e, rate_er, rate_en, rate_enr : np.ndarray
        Rate matrices for different tunneling processes.
    """
    E = SpinSys.eigVectors
    e_sig, er_sig, en_sig, enr_sig = e_epsilon(SpinSys)

    dim = E.shape[0]

    # Use @ for matrix multiplication instead of * (which is element-wise)
    matrxx = E.conj().T @ SpinSys.Sx[0] @ E
    matrxy = E.conj().T @ SpinSys.Sy[0] @ E
    matrxz = E.conj().T @ SpinSys.Sz[0] @ E
    I = np.eye(SpinSys.Sx[0].shape[0])
    matrxu=SpinSys.U * E.conj().T @ I @ E

    J = SpinSys.J*np.array([1, 1, 1])

    Smatrxx = J[0] * matrxx
    Smatrxy = J[1] * matrxy
    Smatrxz = J[2] * matrxz
    Smatrxu = np.zeros_like(Smatrxx)

    TJ=np.array([1, 1, 1]);  #standard value - no changes implemented yet
    matrxx = TJ[0]*matrxx
    matrxy = TJ[1]*matrxy
    matrxz = TJ[2]*matrxz
    matrxu = -np.sign(TJ[0]) * np.sign(TJ[1]) * np.sign(TJ[2]) * matrxu

    # so1 = tip, so2 = sample, so3 = conjugate of tip
    so1 = [matrxx, matrxy, matrxz, matrxu]              # list of 4 matrices: [Sx, Sy, Sz, S1]
    so2 = [Smatrxx,Smatrxy,Smatrxz,Smatrxu]           # list of 4 matrices
    so3 = so1

    mone = np.ones((1, 2)) # Spin-½ basis vector |↑⟩

    e_tensor   = 0
    er_tensor  = 0
    en_tensor  = 0
    enr_tensor = 0

    for j in range(4):
            for k in range(4):
                for l in range(4):
                    if e_sig[j][k][l] != 0:
                        term_e = (
                            (mone.T * so3[l][initial_state_idx, :][np.newaxis, :]) *
                            so2[k].T *
                            (mone * so1[j][:, initial_state_idx][:, np.newaxis])
                        )
                        e_tensor -= e_sig[j][k][l] * term_e

                    if er_sig[j][k][l] != 0:
                        term_er = (
                            (mone.T * so3[l][initial_state_idx, :][np.newaxis, :]) *
                            so1[k].T *
                            (mone * so2[j][:, initial_state_idx][:, np.newaxis])
                        )
                        er_tensor -= er_sig[j][k][l] * term_er

                    if en_sig[j][k][l] != 0:
                        term_en = (
                            (mone.T * so1[l][initial_state_idx, :][np.newaxis, :]) *
                            so2[k].T *
                            (mone * so3[j][:, initial_state_idx][:, np.newaxis])
                        )
                        en_tensor -= en_sig[j][k][l] * term_en

                    if enr_sig[j][k][l] != 0:
                        term_enr = (
                            (mone.T * so1[l][initial_state_idx, :][np.newaxis, :]) *
                            so3[k].T *
                            (mone * so2[j][:, initial_state_idx][:, np.newaxis])
                        )
                        enr_tensor -= enr_sig[j][k][l] * term_enr
    return e_tensor, er_tensor, en_tensor, enr_tensor



def spec3(SpinSys: SpinSys, initial_state_idx, V_array):
    """
    Third-order tunneling contribution (Kondo-like) to IETS spectrum
    """
    lt = SpinSys.lt
    occ = occupations(SpinSys)
    w = SpinSys.w  # parameter 'w' from target atom
    ve = np.real(SpinSys.E_All)  # Eigenvalues of the system

    # Get rate matrices: you must implement M35 to return these 4 arrays
    rate, rater, raten, ratenr = compute_third_order_rates(SpinSys,initial_state_idx)

    # Adjust precision threshold
    prec = SpinSys.precission**2
    # To be added from Ternes Code: If SpinSys has T_ts, use it; otherwise, use SpinSys.precission
    #if hasattr(SpinSys, 'T_ts'):
        #prec = SpinSys.precission**2 * (np.sum(SpinSys.T_ts**2)**2) / 9
    #else:
    #    prec = SpinSys.precission**2

    speclength = len(V_array)

    y = np.zeros(speclength)
    yr = np.zeros(speclength)

    dim = SpinSys.eigVectors.shape[0]
    for mid in range(dim):
        for fin in range(dim):
            if (
                np.abs(rate[mid, fin]) +
                np.abs(raten[mid, fin]) +
                np.abs(rater[mid, fin]) +
                np.abs(ratenr[mid, fin])
            ) > prec:
                E_mi = ve[mid] - ve[initial_state_idx]
                E_fi = ve[fin] - ve[initial_state_idx]

                # These are vectorized versions across V_array
                tlog1 = ln_t2(E_mi + V_array, w, SpinSys.kB * SpinSys.T, lt)
                tlog2 = ln_t2(E_mi - V_array, w, SpinSys.kB * SpinSys.T, lt)

                ep = fstep((E_fi - V_array) / (SpinSys.kB * SpinSys.T))
                en = fstep((E_fi + V_array) / (SpinSys.kB * SpinSys.T))

                ytemp   = -occ[initial_state_idx] * 2 * np.real(rate[mid, fin] * ep * tlog2)
                ytempn  = -occ[initial_state_idx] * 2 * np.real(raten[mid, fin] * en * tlog1)
                ytempr  = -occ[initial_state_idx] * 2 * np.real(rater[mid, fin] * ep * tlog1)
                ytempnr = -occ[initial_state_idx] * 2 * np.real(ratenr[mid, fin] * en * tlog2)

                y  += ytemp + ytempn
                yr += ytempr + ytempnr

    return V_array, y, yr

import numpy as np

def occupations(SpinSys: SpinSys):
    """
    Compute thermal occupation probabilities using the Boltzmann distribution.

    Parameters:
        eigenvalues : array_like
            Array of energy eigenvalues (in meV).
        temperature : float
            Temperature of the system (in meV).

    Returns:
        occupations : np.ndarray
            Normalized occupation probabilities for each energy level.
    """
    eigenvalues = np.real(SpinSys.E_All)  # Ensure eigenvalues are real
    shifted_energies = eigenvalues - np.min(eigenvalues)  # Shift ground state to 0

    if SpinSys.T != 0:
        unnormalized_probs = np.exp(-shifted_energies / (SpinSys.kB * SpinSys.T))
        occupations = unnormalized_probs / np.sum(unnormalized_probs)
    else:
        occupations = np.zeros_like(shifted_energies)
        ground_state_indices = np.where(shifted_energies == 0)[0]
        occupations[ground_state_indices] = 1.0
        occupations /= np.sum(occupations)

    return occupations

from scipy.signal import convolve

def ln_t2(x, e0, T, lt):
    """
    Temperature-folded log convolution function (used in third-order tunneling spectra).
    
    Parameters
    ----------
    x : ndarray
        Voltage bias array (1D).
    e0 : float
        Half-bandwidth of the substrate electrons (meV).
    T : float
        Temperature (meV).
    lt : float
        Lifetime broadening (meV).

    Returns
    -------
    e : ndarray
        Convolved result (same shape as x).
    """
    x = np.asarray(x, dtype=float)
    xspan = x[0] - x[-1]  # assumes x is descending; flip if needed
    i = 2 * len(x)

    # Extended energy window for convolution kernel
    y = np.linspace(xspan, -xspan, i + 1)

    # Define complex logarithmic kernel
    complex_lt = lt * 1j
    g = np.log((e0 + np.abs(y)) / (y + complex_lt)) + np.pi * 1j / 2

    # Fermi-broadened kernel
    z = x / T
    ez = np.exp(z)
    denom = (ez - 1)
    denom_sq = denom**2
    denom_cu = denom**3
    f = -((ez + ez * (z - 1)) / denom_sq - 2 * ez * (ez * (z - 1) + 1) / denom_cu)

    # Handle z ≈ 0 where numerical instabilities occur
    f[np.abs(z) < 1e-3] = 1 / 6
    f[np.isnan(f)] = 0

    # Perform convolution (same size output)
    conv_result = convolve(f, g, mode="same")
    e = 2 * conv_result / (T * i) * np.abs(xspan)

    return e


def fstep(x):
    """
    Temperature-broadened step function used in IETS simulations.
    
    Parameters
    ----------
    x : ndarray
        Normalized energy difference (e.g., (E_j - E_i - V) / (kT))

    Returns
    -------
    y : ndarray
        Broadened step function values (same shape as x)
    """
    x = np.asarray(x, dtype=float)

    # Clean small values near zero
    x[np.abs(x) < 1e-5] = 0

    z = np.exp(x)
    denom = (z - 1)**2
    y = (1 + (x - 1) * z) / denom

    # Set exact x == 0 case to 0.5
    y[x == 0] = 0.5

    # Clean up NaNs (due to overflows etc.)
    y[np.isnan(y)] = 0

    return y


def calcIETS_3rd( SpinSys: SpinSys, Vrange: list = [-10, 10], N: int = 100, norm: bool = False,
    plot: bool = False,
):
    """
    Calculate IETS spectrum with third-order contributions (e.g. Kondo processes)

    Parameters
    ----------
    SpinSys : SpinSys
        Spin system object containing all relevant parameters and operators.
    Vrange : list, optional
        Voltage range as [Vmin, Vmax] (default: [-10, 10]).
    N : int, optional
        Number of voltage points (default: 100).
    norm : bool, optional
        If True, normalize the resulting spectrum (default: False).
    plot : bool, optional
        If True, plot the resulting spectrum (default: True).

    Returns
    -------
    V_array : np.ndarray
        Array of bias voltages (mV).
    dIdV_3rd : np.ndarray
        Third-order differential conductance spectrum (a.u.).
    """

    # Bias voltage array
    V_array = np.linspace(Vrange[0], Vrange[1], N)
    dIdV_3rd = np.zeros_like(V_array, dtype=float)

    [V_array, dIdV2] = dIdV.calcIETS_2nd(SpinSys, Vrange=[np.min(V_array), np.max(V_array)],N=N, norm=False)  # Calculate the second order IETS:
    dIdV_3rd += dIdV2

    # Calculate thermal occupations
    p0 = np.exp(-SpinSys.E_All / (SpinSys.kB * SpinSys.T))
    pTot = np.sum(p0)
    SpinSys.p0 = p0 / pTot
    dim = SpinSys.eigVectors.shape[1]

    for i in range(dim):
        if SpinSys.p0[i] > 1e-6:
            x, y1, y1r = spec3(SpinSys, i, V_array)
            # Add contributions from the third-order processes
            dIdV_3rd += (y1 + y1r)

    # Normalize result
    if norm:
        dIdV_3rd = (dIdV_3rd - np.min(dIdV_3rd)) / (np.max(dIdV_3rd) - np.min(dIdV_3rd))

    if plot:
        plt.figure(189)
        plt.plot(V_array, dIdV_3rd, '-b', linewidth=2)
        plt.ylabel('dI/dV (3rd order, a.u.)')
        plt.xlabel('Bias Voltage (mV)')
        plt.xlim([Vrange[0], Vrange[1]])
        plt.title('IETS Spectrum - Third Order (Kondo)')
        plt.grid(True)
        plt.tick_params(axis='both', which='major', labelsize=16)
        plt.show()

    return V_array, dIdV_3rd

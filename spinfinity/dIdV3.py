"""
Calculating differential conductence (dI/dV) spectra in 3rd order.

This module provides routines to compute differential conductance spectra using
third-order perturbation theory.

Functions
---------
calc_IETS_3rd
    Calculate conductance in 3rd order.
calc_IETS_3rdR
    Calculate third-order spectra from corrected rates and populations.

Dependencies
------------
- Uses `spinfinity.SpinSys.SpinSys` class to represent the spin system
  and its properties.
- Uses `spinfinity.SpinHamiltonian` for eigenvalue and eigenvector calculations.
- Uses functions from `spinfinity.Bath` for population and rate calculations.

References
----------
- Ternes, M., *Spin excitations and correlations in scanning tunneling spectroscopy*.
New J. Phys. 17, 063016 (2015).
[https://dx.doi.org/10.1088/1367-2630/17/6/063016](https://dx.doi.org/10.1088/1367-2630/17/6/063016)
"""

import numpy as np
import scipy.constants as const
from scipy.integrate import cumulative_trapezoid, trapezoid
from scipy.linalg import null_space
from scipy.signal import fftconvolve
from scipy.special import expit

from spinfinity import Bath as bath
from spinfinity import SpinHamiltonian as hamil
from spinfinity import dIdV
from spinfinity.SpinSys import SpinSys


def _electron_operators(source, target):
    """Build weighted electron vertices between two reservoirs.

    Parameters
    ----------
    source, target : array_like
        Physical Cartesian reservoir polarizations with magnitude at most one.

    Returns
    -------
    ndarray
        Weighted Pauli x, y, z and 2*identity operators, shape (4, 2, 2).
        Rows belong to the target reservoir; columns to the source.
    """
    pauli = np.array(
        [
            [[0, 1], [1, 0]],
            [[0, -1j], [1j, 0]],
            [[1, 0], [0, -1]],
            [[2, 0], [0, 2]],
        ],
        dtype=complex,
    )
    rho_source = (np.eye(2) + np.einsum("a,aij->ij", source, pauli[:3])) / 2
    rho_target = (np.eye(2) + np.einsum("a,aij->ij", target, pauli[:3])) / 2
    p_source, v_source = np.linalg.eigh(rho_source)
    p_target, v_target = np.linalg.eigh(rho_target)
    weights = np.sqrt(np.outer(np.maximum(p_target, 0), np.maximum(p_source, 0)))
    return np.array([v_target.conj().T @ op @ v_source * weights for op in pauli])


def _pathways(spinsys, site, source, target, coupling, potential=True):
    """Contract the four electron/hole interference pathways.

    Parameters
    ----------
    spinsys : SpinSys
        Spin system with current eigenvectors and dimensionless spin operators.
    site : int
        Local scattering site.
    source, target : array_like
        Cartesian reservoir polarizations.
    coupling : float
        Signed dimensionless exchange coupling of the virtual target bath.
    potential : bool, optional
        Include U in transport vertices if True; default True.
        Use False for local ss and tt exchange corrections.

    Returns
    -------
    ndarray
        Cubic tensors with axes (pathway, initial, intermediate, final).
        Pathways are e, er, en, enr. No legacy M35 global minus is included.
    """
    dimension = spinsys.dimensionOfMatrix
    result = np.zeros((4, dimension, dimension, dimension), dtype=complex)
    if coupling == 0:
        return result

    transport = _electron_operators(source, target)
    intermediate = _electron_operators(target, target)
    vectors = spinsys.eigVectors
    operators = np.array(
        [
            vectors.conj().T @ spinsys.Sx[site] @ vectors,
            vectors.conj().T @ spinsys.Sy[site] @ vectors,
            vectors.conj().T @ spinsys.Sz[site] @ vectors,
            (spinsys.U[site] if potential else 0) * np.eye(dimension),
        ]
    )
    exchange = coupling * operators.copy()
    exchange[3] = 0

    for j in range(4):
        for k in range(4):
            for ell in range(4):
                e = np.trace(transport[ell].conj().T @ intermediate[k] @ transport[j])
                er = -np.trace(transport[ell].conj().T @ intermediate[j] @ transport[k])
                en = -np.trace(transport[j].conj().T @ intermediate[k] @ transport[ell])
                enr = np.trace(transport[k].conj().T @ intermediate[j] @ transport[ell])
                direct = np.einsum(
                    "if,fm,mi->imf", operators[ell], exchange[k], operators[j]
                )
                reverse = np.einsum(
                    "if,fm,mi->imf", operators[ell], operators[k], exchange[j]
                )
                result[0] += e * direct
                result[1] += er * reverse
                result[2] += en * direct
                result[3] += enr * reverse

    return result


def _log_grid(w, thermal_energy, lt, limit, step):
    """Fold the Scilab complex logarithm with the broadened-step derivative.

    Parameters
    ----------
    w : float
        Positive bandwidth in meV.
    thermal_energy : float
        Nonnegative kB*T in meV.
    lt : float
        Positive lifetime broadening in meV.
    limit : float
        Required positive energy extent in meV.
    step : float
        Positive grid spacing in meV.

    Returns
    -------
    grid : ndarray
        Symmetric uniform energy grid in meV, padded against convolution edges.
    logarithm : ndarray
        Thermally folded complex logarithm on this grid. Its real part is
        the regularized logarithmic peak; its imaginary part is dispersive.
    """
    padding = 30 * thermal_energy
    points = int(np.ceil((limit + padding) / step)) + 1
    grid = np.arange(-points, points + 1) * step
    logarithm = np.log(w + np.abs(grid)) - 0.5 * np.log(grid**2 + lt**2)
    logarithm = logarithm + 1j * np.arctan(grid / lt)

    if thermal_energy == 0:
        return grid, logarithm

    half_width = max(1, int(np.ceil(padding / step)))
    arguments = np.abs(np.arange(-half_width, half_width + 1) * step / thermal_energy)
    kernel = np.empty_like(arguments)

    small = arguments < 0.05
    z = arguments[small]
    kernel[small] = 1 / 6 - z**2 / 60 + z**4 / 1008 - z**6 / 21600

    z = arguments[~small]
    q = np.exp(-z)
    kernel[~small] = q * ((z - 2) + (z + 2) * q) / (-np.expm1(-z)) ** 3
    kernel /= np.sum(kernel)

    return grid, fftconvolve(logarithm, kernel, mode="same")


def _prepare(spinsys, V_array, energy_step):
    """Prepare eigenstates, reservoir pathways and the logarithm grid.

    Parameters
    ----------
    spinsys : SpinSys
        Spin system. J contains signed dimensionless substrate couplings
        per spin; J_tip is the signed dimensionless tip coupling of the
        readout spin. T is in K; w and lt are in meV.
        The Hamiltonian is rebuilt.
    V_array : array_like or None
        Bias voltages in mV; None uses the default sweep.
    energy_step : float or None
        Internal spacing in meV; None resolves lt and kB*T.

    Returns
    -------
    tuple
        Voltage array, kB*T, spacing, logarithm grid, logarithm values,
        substrate-intermediate pathways and tip-intermediate pathways.
        The latter are constructed with reversed source/target reservoirs.

    Raises
    ------
    ValueError
        If the voltage array is empty or not one-dimensional, polarization
        magnitudes exceed one, T is negative, or w, lt or spacing is nonpositive.
    """
    V_array = np.asarray(
        np.linspace(-30, 30, 200) if V_array is None else V_array, dtype=float
    )
    if V_array.ndim != 1 or V_array.size == 0:
        raise ValueError("V_array must be a nonempty one-dimensional array.")

    if any(
        np.linalg.norm(p) > 1 + 1e-12
        for p in (spinsys.TipPolarization, spinsys.SamplePolarization)
    ):
        raise ValueError("Reservoir polarization magnitudes must not exceed one.")

    if spinsys.T < 0 or spinsys.w <= 0 or spinsys.lt <= 0:
        raise ValueError("Require T >= 0, w > 0 and lt > 0.")

    hamil.calc_EigEnergies(spinsys, rebuildH=True)
    thermal_energy = spinsys.kB * spinsys.T
    step = energy_step
    if step is None:
        step = min(spinsys.lt, thermal_energy or spinsys.lt) / 4
    if step <= 0:
        raise ValueError("energy_step must be positive.")

    span = np.max(np.abs(V_array)) + np.max(spinsys.E_All) + 30 * thermal_energy
    grid, logarithm = _log_grid(
        spinsys.w,
        thermal_energy,
        spinsys.lt,
        span + np.max(spinsys.E_All),
        step,
    )

    site = spinsys.ReadoutSpin
    tip = spinsys.TipPolarization
    sample = spinsys.SamplePolarization
    paths = _pathways(spinsys, site, tip, sample, spinsys.J[site])
    tip_paths = _pathways(spinsys, site, sample, tip, spinsys.J_tip)

    return V_array, thermal_energy, step, grid, logarithm, paths, tip_paths


def _factorized_conductance(energies, voltages, paths, thermal_energy, grid, logarithm):
    """Evaluate the four step-times-log conductance pathways.

    Parameters
    ----------
    energies : ndarray
        Eigenenergies in meV, shape (dimension,).
    voltages : ndarray
        Physical bias in mV.
    paths : ndarray
        Cubic tensors with axes (e/er/en/enr, initial, intermediate, final).
    thermal_energy : float
        Nonnegative kB*T in meV.
    grid, logarithm : ndarray
        Energy grid in meV and thermally folded complex logarithm.

    Returns
    -------
    ndarray
        Dimensionless conductance coefficients, shape
        (2, len(voltages), dimension, dimension). Channels are ts and st;
        populations and the conductance prefactor have not been applied.

    Notes
    -----
    Follows the factorization in Ternes (2015), Eqs. (24)-(25).
    """
    dimension = len(energies)
    result = np.zeros((2, len(voltages), dimension, dimension))
    if not np.any(paths):
        return result

    for initial in range(dimension):
        shifts = (energies - energies[initial])[:, None]
        minus = np.interp(shifts - voltages, grid, logarithm)
        plus = np.interp(shifts + voltages, grid, logarithm)
        for final in range(dimension):
            delta = energies[final] - energies[initial]
            if thermal_energy == 0:
                ep = np.heaviside(voltages - delta, 0.5)
                en = np.heaviside(-voltages - delta, 0.5)
            else:
                ep = dIdV.tunnelingBroadenedStepFunction(
                    (delta - voltages) / thermal_energy
                )
                en = dIdV.tunnelingBroadenedStepFunction(
                    (delta + voltages) / thermal_energy
                )
            a, b, c, d = paths[:, initial, :, final]
            result[0, :, initial, final] = 2 * ep * (a @ minus + b @ plus).real
            result[1, :, initial, final] = 2 * en * (c @ plus + d @ minus).real
    return result


def _phase_integrals(energies, voltages, paths, thermal_energy, grid, logarithm, step):
    """Evaluate cubic scattering integrals.

    Parameters
    ----------
    energies : ndarray
        Eigenenergies in meV.
    voltages : ndarray
        Physical bias voltages in mV.
    paths : ndarray
        Cubic tensors with axes (e/er/en/enr, initial, intermediate, final).
    thermal_energy : float
        Nonnegative kB*T in meV.
    grid, logarithm : ndarray
        Energy grid in meV and thermally folded complex logarithm.
    step : float
        Positive electron-energy spacing in meV.

    Returns
    -------
    ndarray
        Rate-integral corrections in meV, shape
        (2, len(voltages), dimension, dimension), with ts/st channels.
        No populations, conductances or conversion to s^-1 are included.
    """
    dimension = len(energies)
    result = np.zeros((2, len(voltages), dimension, dimension))
    if not np.any(paths):
        return result

    span = np.max(np.abs(voltages)) + np.ptp(energies) + 30 * thermal_energy
    points = max(1, int(np.ceil(span / step)))
    electron_energy = np.arange(-points, points + 1) * step

    for initial in range(dimension):
        shifts = (energies - energies[initial])[:, None]
        minus = np.interp(shifts - electron_energy, grid, logarithm)
        plus = np.interp(shifts + electron_energy, grid, logarithm)
        for final in range(dimension):
            delta = energies[final] - energies[initial]
            a, b, c, d = paths[:, initial, :, final]
            forward = 2 * (a @ minus + b @ plus).real
            backward = 2 * (c @ minus + d @ plus).real
            for channel, function in enumerate((forward, backward)):
                driving = voltages if channel == 0 else -voltages
                if thermal_energy == 0:
                    primitive = cumulative_trapezoid(
                        function, electron_energy, initial=0
                    )
                    integral = np.interp(driving, electron_energy, primitive)
                    integral -= np.interp(delta, electron_energy, primitive)
                    result[channel, :, initial, final] = np.where(
                        driving > delta, integral, 0
                    )
                else:
                    holes = expit((electron_energy - delta) / thermal_energy)
                    for index, bias in enumerate(driving):
                        occupation = expit((bias - electron_energy) / thermal_energy)
                        result[channel, index, initial, final] = trapezoid(
                            occupation * holes * function, electron_energy
                        )
    return result


def calc_IETS_3rd(spinsys: SpinSys, V_array=None, norm=False, energy_step=None):
    """Calculate conductance in 3rd order.

    Parameters
    ----------
    spinsys : SpinSys
        Spin system. J contains signed dimensionless substrate couplings
        per spin; J_tip is the signed dimensionless tip coupling of the
        readout spin. T is in K; w and lt are in meV.
        The Hamiltonian is rebuilt.
    V_array : array_like, optional
        Bias voltages in mV. Default is 200 points from -30 to 30 mV.
    norm : bool, optional
        Divide by the first conductance value if True. Default is False.
    energy_step : float, optional
        Logarithm convolution grid spacing in meV. Default is one quarter
        of the smaller nonzero scale among lt and kB*T.

    Returns
    -------
    V_array : ndarray
        Bias voltages in mV.
    conductance : ndarray
        Differential conductance in S, or dimensionless when norm=True.

    Raises
    ------
    ValueError
        If the ground-state elastic normalization or the requested
        first-point normalization is zero.

    Notes
    -----
    Uses thermal populations and factorized broadened-step times logarithm
    contributions, as in Ternes (2015), Eqs. (24)-(25).
    J=J_tip=0 exactly recovers calc_IETS_2nd.
    Eigenstates and the existing second-order thermal caches are refreshed.
    """
    V, thermal_energy, _, grid, logarithm, paths, tip_paths = _prepare(
        spinsys, V_array, energy_step
    )

    rate_0 = bath.calc_TunnelingMatrixElements(spinsys, "st")[0, 0, 0]
    if rate_0 <= 0:
        raise ValueError("The ground-state elastic normalization is zero.")

    _, conductance = dIdV.calc_IETS_2nd(spinsys, V_array=V, norm=False)
    derivatives = _factorized_conductance(
        spinsys.E_All,
        V,
        paths,
        thermal_energy,
        grid,
        logarithm,
    )

    # The tip-intermediate calculation uses reversed reservoir roles.
    derivatives += _factorized_conductance(
        spinsys.E_All,
        -V,
        tip_paths,
        thermal_energy,
        grid,
        logarithm,
    )[::-1]

    thermal = bath.calc_ThermalPopulations(spinsys)
    correction = np.einsum("i,cvij->v", thermal, derivatives)
    conductance += (1 - spinsys.b0) * spinsys.G_st / rate_0 * correction

    if norm:
        if conductance[0] == 0:
            raise ValueError("Cannot normalize by a zero first conductance value.")
        conductance = conductance / conductance[0]

    return V, conductance


def calc_IETS_3rdR(
    spinsys: SpinSys,
    V_array=None,
    AllowPumping=True,
    norm=False,
    energy_step=None,
):
    """Calculate third-order spectra from corrected rates and populations.

    Parameters
    ----------
    spinsys : SpinSys
        Spin system. J contains signed dimensionless substrate couplings
        per spin; J_tip is the signed dimensionless tip coupling of the
        readout spin. T is in K; w and lt are in meV.
        The Hamiltonian is rebuilt.
    V_array : array_like, optional
        Strictly increasing bias voltages in mV, at least two points.
        Default is 200 points from -30 to 30 mV.
    AllowPumping : bool, optional
        Solve the corrected stationary rate equation if True (default);
        otherwise use thermal populations at every bias.
    norm : bool, optional
        Divide by the first conductance value if True. Default is False.
    energy_step : float, optional
        Internal energy spacing in meV; default as in calc_IETS_3rd.

    Returns
    -------
    V_array : ndarray
        Bias voltages in mV.
    conductance : ndarray
        Differential conductance in S, or dimensionless when norm=True.
    populations : ndarray
        Eigenstate populations, shape (dimensionOfMatrix, len(V_array)).

    Raises
    ------
    ValueError
        If the voltage grid is not strictly increasing with at least two
        points, a normalization is zero, enabled G_tt has nonpositive
        readout G_ss, total rates become negative, or the steady state is not unique.
    """
    V, thermal_energy, step, grid, logarithm, paths, tip_paths = _prepare(
        spinsys, V_array, energy_step
    )
    if len(V) < 2 or np.any(np.diff(V) <= 0):
        raise ValueError(
            "V_array must be strictly increasing with at least two points."
        )

    energies = spinsys.E_All
    dimension = len(energies)
    transport = _phase_integrals(
        energies, V, paths, thermal_energy, grid, logarithm, step
    )
    transport += _phase_integrals(
        energies, -V, tip_paths, thermal_energy, grid, logarithm, step
    )[::-1]

    matrix_st = bath.calc_TunnelingMatrixElements(spinsys, "st")[0]
    matrix_ss = bath.calc_TunnelingMatrixElements(spinsys, "ss")
    matrix_tt = bath.calc_TunnelingMatrixElements(spinsys, "tt")[0]

    rate_0 = matrix_st[0, 0]
    if rate_0 <= 0:
        raise ValueError("The ground-state elastic normalization is zero.")

    factor = 1e-3 / (const.e * rate_0)
    effective = (1 - spinsys.b0) * spinsys.G_st

    # Local substrate corrections, summed incoherently over scattering sites.
    substrate = np.zeros((dimension, dimension))
    for site in range(spinsys.NSpins):
        local_paths = _pathways(
            spinsys,
            site,
            spinsys.SamplePolarization,
            spinsys.SamplePolarization,
            spinsys.J[site],
            potential=False,
        )
        correction = _phase_integrals(
            energies,
            np.array([0.0]),
            local_paths,
            thermal_energy,
            grid,
            logarithm,
            step,
        )[0, 0]
        substrate += spinsys.G_ss[site] * correction

    tip_paths = _pathways(
        spinsys,
        spinsys.ReadoutSpin,
        spinsys.TipPolarization,
        spinsys.TipPolarization,
        spinsys.J_tip,
        potential=False,
    )
    tip_correction = _phase_integrals(
        energies,
        np.array([0.0]),
        tip_paths,
        thermal_energy,
        grid,
        logarithm,
        step,
    )[0, 0]

    tip_scale = 0.0
    if spinsys.G_tt:
        if spinsys.G_ss[spinsys.ReadoutSpin] <= 0:
            raise ValueError("G_tt requires positive G_ss for the readout spin.")
        tip_scale = effective**2 / spinsys.G_ss[spinsys.ReadoutSpin] * spinsys.G_tt

    old_voltage = spinsys.V_DC
    populations = np.zeros((dimension, len(V)))
    current = np.zeros(len(V))
    thermal = bath.calc_ThermalPopulations(spinsys)

    try:
        bath_integral = bath.calc_RateIntegrals(spinsys, "ss")
        fixed_ss = factor * (
            np.einsum("k,kij->ij", spinsys.G_ss, matrix_ss) * bath_integral + substrate
        )
        fixed_tt = factor * tip_scale * (matrix_tt * bath_integral + tip_correction)

        for index, voltage in enumerate(V):
            spinsys.V_DC = voltage
            rates = np.array(
                [
                    factor
                    * effective
                    * (
                        matrix_st.T * bath.calc_RateIntegrals(spinsys, "ts")
                        + transport[0, index]
                    ),
                    factor
                    * effective
                    * (
                        matrix_st * bath.calc_RateIntegrals(spinsys, "st")
                        + transport[1, index]
                    ),
                    fixed_ss,
                    fixed_tt,
                ]
            )

            if np.any(rates < 0):
                raise ValueError(
                    "Negative total rates: reduce |J|/|J_tip| or increase T."
                )

            if AllowPumping:
                total = np.sum(rates, axis=0)
                np.fill_diagonal(total, 0)
                generator = total.T.copy()
                np.fill_diagonal(generator, -np.sum(total, axis=1))
                steady = null_space(generator)

                if steady.shape[1] != 1:
                    raise ValueError("The stationary population is not unique.")

                population = steady[:, 0] / np.sum(steady[:, 0])
            else:
                population = thermal

            populations[:, index] = population
            current[index] = (
                const.e * np.sum(population[:, None] * (rates[0] - rates[1]))
                + spinsys.b0 * spinsys.G_st * voltage * 1e-3
            )
    finally:
        spinsys.V_DC = old_voltage

    conductance = np.gradient(current, V * 1e-3)
    if norm:
        if conductance[0] == 0:
            raise ValueError("Cannot normalize by a zero first conductance value.")
        conductance = conductance / conductance[0]

    return V, conductance, populations

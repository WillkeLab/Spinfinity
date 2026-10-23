"""
Calculating collapse operators and computing the time evolution.

This module provides calculation and visualization helpers for solving
the Lindblad master equation for a given spin system. It includes functions to compute
collapse operators, solve the time evolution of the system under the Lindblad equation,
and visualize the results.

Functions
---------
_calc_elasticTunnelingAmplitudes
    Calculate state-resolved elastic tunneling amplitudes.
calc_elasticCollapseOperators
    Calculate elastic collapse operators for tunneling-induced dephasing.
calc_dephasingOperator
    Calculate a phenomenological local z-dephasing collapse operator of spin_idx.
calc_collapseOperators
    Calculate the collapse operators for the tunnel junction.
initialize_thermal
    Initialize the spin system in a thermal state.
calc_TimeEvolution
    Solve the Lindblad master equation with QuTiP and return mesolve result.
calc_TimeEvolution_Tipfield
    Solve the Lindblad master equation for different BTip.
_calc_TipfieldParallelHelper
    Solve the Lindblad master equation for a BTip value.
calc_DensityMatrix
    Returns selected density matrices from a previously computed time evolution result.
calc_TunnelCurrent_time
    Calculate the tunneling current over time from a QuTiP time-evolution result.
plot_TimeEvolution
    Plot spin expectation values over time from a previously computed time evolution.
plot_TimeEvolution_Tipfield
    Plot time evolution as a function of tip-field strength.
plot_DensityMatrix
    Plot the real part of the density matrix at selected times as a 3D bar chart.
plot_TunnelCurrent_time
    Plot the tunneling current as a function of time.
plot_FourierFreqs
    Plot the frequency spectrum from an existing time evolution result.

Dependencies
------------
- Uses `spinfinity.SpinSys.SpinSys` class to represent the spin system
  and its properties.
- Uses `spinfinity.SpinHamiltonian` for eigenvalue and eigenvector calculations.
- Uses functions from `spinfinity.Bath` for population and rate calculations.

References
----------
- Lambert, N., Giguère, E., Menczel, P. et al. *QuTiP 5: The Quantum Toolbox in Python*.
  Physics Reports 1153,18 1-62 (2026).
  [https://doi.org/10.1016/j.physrep.2025.10.001)

"""

# import python libraries
from copy import deepcopy

import numpy as np
import plotly.graph_objects as go
import scipy
import scipy.constants as const
from joblib import Parallel, delayed
from plotly.subplots import make_subplots
from qutip import Qobj, mesolve

# import other SpinFinity packages
import spinfinity.Bath as bath
import spinfinity.SpinHamiltonian as hamil
from spinfinity import helperfunctions as hf
from spinfinity import plotting
from spinfinity.SpinSys import SpinSys

# region Calculations


def _calc_elasticTunnelingAmplitudes(
    spinsys: SpinSys,
    scattertype: str,
    spin_idx: int = None,
):
    """
    Calculate state-resolved elastic tunneling amplitudes.

    The returned amplitudes retain the electron-spin channel indices that are
    summed over in :func:`spinfinity.Bath.calc_TunnelingMatrixElements`.

    Parameters
    ----------
    spinsys : SpinSys
        Spin system containing eigenvectors and spin operators.
    scattertype : str
        Scattering process: ``"ts"`` (tip to sample), ``"st"`` (sample to
        tip), ``"ss"`` (sample to sample), or ``"tt"`` (tip to tip).
    spin_idx : int, optional
        Index of the system spin involved in the scattering process. By
        default, ``spinsys.ReadoutSpin`` is used.

    Returns
    -------
    numpy.ndarray
        Complex array with shape ``(2, 2, dimensionOfMatrix)``. The first two
        indices identify the electron-spin channel and the final index the
        unchanged spin-system eigenstate.

    Raises
    ------
    ValueError
        If ``scattertype`` or ``spin_idx`` is invalid.
    """
    if scattertype not in ("ts", "st", "ss", "tt"):
        raise ValueError('scattertype must be "ts", "st", "ss", or "tt"')

    if spin_idx is None:
        spin_idx = spinsys.ReadoutSpin

    if spin_idx < 0 or spin_idx >= spinsys.NSpins:
        raise ValueError("spin_idx is outside the spin system")

    electron_scattertype = scattertype
    if scattertype in ("ts", "st"):
        electron_scattertype = "st"

    x_el, y_el, z_el, u_el = bath.calc_TunnelingElectronMatrixElements(
        spinsys,
        electron_scattertype,
    )

    # Bath stores the tip-to-sample amplitudes as <sample|O|tip>.
    # The reverse process is obtained through Hermitian conjugation
    # and exchange of the electron channel indices.
    if scattertype == "st":
        x_el = np.conjugate(x_el.T)
        y_el = np.conjugate(y_el.T)
        z_el = np.conjugate(z_el.T)
        u_el = np.conjugate(u_el.T)

    eigVectors = spinsys.eigVectors

    MatrixX = np.conjugate(eigVectors.T) @ spinsys.Sx[spin_idx, :, :] @ eigVectors
    MatrixY = np.conjugate(eigVectors.T) @ spinsys.Sy[spin_idx, :, :] @ eigVectors
    MatrixZ = np.conjugate(eigVectors.T) @ spinsys.Sz[spin_idx, :, :] @ eigVectors

    diagonalX = np.diag(MatrixX)
    diagonalY = np.diag(MatrixY)
    diagonalZ = np.diag(MatrixZ)

    diagonalU = spinsys.U[spin_idx] * np.ones(
        spinsys.dimensionOfMatrix,
        dtype=complex,
    )

    amplitudes = np.zeros(
        (2, 2, spinsys.dimensionOfMatrix),
        dtype=complex,
    )

    for electron_initial in range(2):
        for electron_final in range(2):
            amplitudes[
                electron_initial,
                electron_final,
                :,
            ] = (
                diagonalX * x_el[electron_initial, electron_final]
                + diagonalY * y_el[electron_initial, electron_final]
                + diagonalZ * z_el[electron_initial, electron_final]
                + diagonalU * u_el[electron_initial, electron_final]
            )

    return amplitudes


def calc_elasticCollapseOperators(
    spinsys: SpinSys,
    recalcRates: bool = False,
):
    """
    Calculate elastic collapse operators for tunneling-induced dephasing.

    The contributions are calculated separately in the following order:
    1. Sample-to-tip scattering
    2. Tip-to-sample scattering
    3. Sample-to-sample scattering
    4. Tip-to-tip scattering

    Parameters
    ----------
    spinsys : SpinSys
        Spin system containing eigenvectors, spin operators, rates, and
        tunneling parameters.
    recalcRates : bool, optional
        Whether to rebuild the Hamiltonian, eigenstates, and tunneling rates.
        Default is False.

    Returns
    -------
    list[Qobj]
        Elastic collapse operators in the original spin-product basis.
        The units are ``ns**(-1/2)``.

    Raises
    ------
    ValueError
        If the tunneling rates are not finite and non-negative.

    Notes
    -----
    The function does not modify ``spinsys.collapse_ops``.

    The generated operators are stored in:

    - ``spinsys.elastic_collapse_ops``
    - ``spinsys.elastic_collapse_ops_by_scattertype``

    The calculated dephasing rates in ``s**(-1)`` are stored in:

    - ``spinsys.elastic_dephasing_rates``
    - ``spinsys.elastic_dephasing_rates_by_scattertype``
    """
    if recalcRates or (
        not hasattr(spinsys, "Rates")
        or spinsys.Rates is None
        or not hasattr(spinsys, "eigVectors")
        or spinsys.eigVectors is None
    ):
        hamil.build_Hamiltonian(spinsys)
        hamil.calc_EigEnergies(spinsys)
        hamil.calc_EigStates(spinsys)
        bath.calc_Rates(spinsys)

    def calculate_rate_scale(
        amplitude_rates,
        elastic_rates,
    ):
        """
        Match the tunneling amplitudes to the rates from Bath.

        Parameters
        ----------
        amplitude_rates : numpy.ndarray
            Array of rates calculated from the tunneling amplitudes.
        elastic_rates : numpy.ndarray
            Array of rates calculated from the Bath module.

        Returns
        -------
        float
            Scaling factor to match the amplitude rates to the elastic rates.

        Raises
        ------
        ValueError
            If the elastic rates are not finite and non-negative.

        """
        nonzero_amplitudes = amplitude_rates > np.finfo(float).eps

        if not np.any(nonzero_amplitudes):
            return 0

        selected_rates = elastic_rates[nonzero_amplitudes]

        if np.any(~np.isfinite(selected_rates)) or np.any(selected_rates < 0):
            raise ValueError("Elastic rates must be finite and non-negative.")

        rate_ratios = selected_rates / amplitude_rates[nonzero_amplitudes]

        return np.sqrt(np.median(rate_ratios))

    def build_channel_operators(
        amplitudes,
        rate_scale,
    ):
        """
        Construct the operators for all electron-spin channels.

        Parameters
        ----------
        amplitudes : numpy.ndarray
            Complex array of tunneling amplitudes with shape
            ``(2, 2, dimensionOfMatrix)``.
        rate_scale : float
            Scaling factor to match the amplitude rates to the elastic rates.

        Returns
        -------
        list[Qobj]
            List of collapse operators for the given channel.
        numpy.ndarray
            Array of dephasing rates with shape
            ``(dimensionOfMatrix, dimensionOfMatrix)``.
        """
        collapse_ops = []

        dephasing_rates = np.zeros(
            (
                spinsys.dimensionOfMatrix,
                spinsys.dimensionOfMatrix,
            )
        )

        for electron_initial in range(2):
            for electron_final in range(2):
                channel_amplitudes = (
                    rate_scale
                    * amplitudes[
                        electron_initial,
                        electron_final,
                        :,
                    ]
                )

                pairwise_rates = (
                    0.5
                    * np.abs(
                        channel_amplitudes[:, np.newaxis]
                        - channel_amplitudes[np.newaxis, :]
                    )
                    ** 2
                )

                # A state-independent operator is proportional to
                # the identity and does not cause dephasing.
                if np.max(pairwise_rates) == 0:
                    continue

                operator_energy_basis = np.diag(channel_amplitudes)

                operator_product_basis = (
                    spinsys.eigVectors
                    @ operator_energy_basis
                    @ np.conjugate(spinsys.eigVectors.T)
                )

                collapse_operator = Qobj(operator_product_basis / np.sqrt(1e9))

                collapse_ops.append(collapse_operator)
                dephasing_rates += pairwise_rates

        return collapse_ops, dephasing_rates

    # -------------------------------------------------------------
    # Sample-to-tip scattering
    # -------------------------------------------------------------

    amplitudes_st = _calc_elasticTunnelingAmplitudes(
        spinsys,
        "st",
    )

    amplitude_rates_st = np.sum(
        np.abs(amplitudes_st) ** 2,
        axis=(0, 1),
    )

    elastic_rates_st = np.real(np.diag(spinsys.Rates[1, :, :]))

    rate_scale_st = calculate_rate_scale(
        amplitude_rates_st,
        elastic_rates_st,
    )

    collapse_ops_st, dephasing_rates_st = build_channel_operators(
        amplitudes_st,
        rate_scale_st,
    )

    # -------------------------------------------------------------
    # Tip-to-sample scattering
    # -------------------------------------------------------------

    amplitudes_ts = _calc_elasticTunnelingAmplitudes(
        spinsys,
        "ts",
    )

    amplitude_rates_ts = np.sum(
        np.abs(amplitudes_ts) ** 2,
        axis=(0, 1),
    )

    elastic_rates_ts = np.real(np.diag(spinsys.Rates[0, :, :]))

    rate_scale_ts = calculate_rate_scale(
        amplitude_rates_ts,
        elastic_rates_ts,
    )

    collapse_ops_ts, dephasing_rates_ts = build_channel_operators(
        amplitudes_ts,
        rate_scale_ts,
    )

    # -------------------------------------------------------------
    # Sample-to-sample scattering
    # -------------------------------------------------------------

    ss_weights = np.asarray(
        spinsys.G_ss,
        dtype=float,
    )

    if (
        len(ss_weights) != spinsys.NSpins
        or np.any(~np.isfinite(ss_weights))
        or np.any(ss_weights < 0)
    ):
        raise ValueError("G_ss must contain one finite, non-negative value per spin.")

    amplitudes_ss = [
        _calc_elasticTunnelingAmplitudes(
            spinsys,
            "ss",
            spin_idx,
        )
        for spin_idx in range(spinsys.NSpins)
    ]

    amplitude_rates_ss = np.zeros(spinsys.dimensionOfMatrix)

    for spin_idx in range(spinsys.NSpins):
        amplitude_rates_ss += ss_weights[spin_idx] * np.sum(
            np.abs(amplitudes_ss[spin_idx]) ** 2,
            axis=(0, 1),
        )

    elastic_rates_ss = np.real(np.diag(spinsys.Rates[2, :, :]))

    rate_scale_ss = calculate_rate_scale(
        amplitude_rates_ss,
        elastic_rates_ss,
    )

    collapse_ops_ss = []

    dephasing_rates_ss = np.zeros(
        (
            spinsys.dimensionOfMatrix,
            spinsys.dimensionOfMatrix,
        )
    )

    for spin_idx in range(spinsys.NSpins):
        spin_rate_scale = rate_scale_ss * np.sqrt(ss_weights[spin_idx])

        spin_collapse_ops, spin_dephasing_rates = build_channel_operators(
            amplitudes_ss[spin_idx],
            spin_rate_scale,
        )

        collapse_ops_ss += spin_collapse_ops
        dephasing_rates_ss += spin_dephasing_rates

    # -------------------------------------------------------------
    # Tip-to-tip scattering
    # -------------------------------------------------------------

    amplitudes_tt = _calc_elasticTunnelingAmplitudes(
        spinsys,
        "tt",
    )

    amplitude_rates_tt = np.sum(
        np.abs(amplitudes_tt) ** 2,
        axis=(0, 1),
    )

    elastic_rates_tt = np.real(np.diag(spinsys.Rates[3, :, :]))

    rate_scale_tt = calculate_rate_scale(
        amplitude_rates_tt,
        elastic_rates_tt,
    )

    collapse_ops_tt, dephasing_rates_tt = build_channel_operators(
        amplitudes_tt,
        rate_scale_tt,
    )

    # -------------------------------------------------------------
    # Collect the separate contributions
    # -------------------------------------------------------------

    elastic_collapse_ops = (
        collapse_ops_st + collapse_ops_ts + collapse_ops_ss + collapse_ops_tt
    )

    elastic_dephasing_rates = (
        dephasing_rates_st
        + dephasing_rates_ts
        + dephasing_rates_ss
        + dephasing_rates_tt
    )

    spinsys.elastic_collapse_ops = elastic_collapse_ops

    spinsys.elastic_collapse_ops_by_scattertype = {
        "st": collapse_ops_st,
        "ts": collapse_ops_ts,
        "ss": collapse_ops_ss,
        "tt": collapse_ops_tt,
    }

    spinsys.elastic_dephasing_rates = elastic_dephasing_rates

    spinsys.elastic_dephasing_rates_by_scattertype = {
        "st": dephasing_rates_st,
        "ts": dephasing_rates_ts,
        "ss": dephasing_rates_ss,
        "tt": dephasing_rates_tt,
    }

    return elastic_collapse_ops


def calc_dephasingOperator(
    spinsys: SpinSys,
    spin_idx: int,
    T_phi: float,
) -> Qobj:
    """
    Calculate a phenomenological local z-dephasing collapse operator of spin_idx.

    Parameters
    ----------
    spinsys : SpinSys
        Spin system containing the spin operators.
    spin_idx : int
        Zero-based index of the spin subject to dephasing.
    T_phi : float
        Pure-dephasing time in ns, not the total transverse time T2.
        Must be positive. Use np.inf to return a zero operator.

    Returns
    -------
    Qobj
        The collapse operator in the product-spin basis, time in ns.

    Raises
    ------
    IndexError
        If spin_idx is outside the range of spins in the system.
    ValueError
        If T_phi is non-positive or NaN.

    Notes
    -----
    The operator is L_phi = sqrt(2 / T_phi) * Sz[spin_idx].
    Spin operators are calculated if missing. The function does not
    modify spinsys.collapse_ops or recalculate the Hamiltonian or rates.

    For spin one-half, this dissipator alone gives off-diagonal decay
    exp(-t / T_phi) in the local z basis. For higher spins, coherences
    between magnetic quantum numbers m and m_prime decay at rate
    (m - m_prime)**2 / T_phi.

    Local z dephasing need not preserve energy-eigenstate populations
    when Sz[spin_idx] does not commute with the full Hamiltonian.
    """
    if not 0 <= spin_idx < spinsys.NSpins:
        raise IndexError("spin_idx is outside the range of spins.")

    if not T_phi > 0:
        raise ValueError("T_phi must be positive or np.inf.")

    if getattr(spinsys, "Sz", None) is None:
        hamil.calc_SpinOperators(spinsys)

    # construct the collapse operator for local z dephasing
    c_op_Tphi = Qobj(np.sqrt(2 / T_phi) * spinsys.Sz[spin_idx])

    return c_op_Tphi


def calc_collapseOperators(
    spinsys: SpinSys,
    threshold_percent: float | None = None,
    recalcRates: bool = True,
):
    """
    Calculate inelastic collapse operators for the tunnel junction.

    Parameters
    ----------
    spinsys : SpinSys
        Spin system containing eigenvectors and tunneling rates.
    threshold_percent : float or None, optional
        Fraction of the largest positive, finite inelastic rate below which
        transitions are omitted. For example, 0.01 corresponds to 1%.
        Default is None, which disables rate-threshold filtering.
    recalcRates : bool, optional
        Recalculate the Hamiltonian, eigenstates and rates. Default is True.

    Returns
    -------
    list[Qobj]
        Collapse operators in the product-spin basis, for time in ns.

    Raises
    ------
    ValueError
        If ``threshold_percent`` is not None and not between 0 and 1.

    Notes
    -----
    Explicit filtering can change steady states, lifetimes and detailed balance.
    Zero, negative and non-finite rates do not produce collapse operators.
    """
    if recalcRates:
        hamil.build_Hamiltonian(spinsys)
        hamil.calc_EigEnergies(spinsys)
        hamil.calc_EigStates(spinsys)
        bath.calc_Rates(spinsys)
    elif not hasattr(spinsys, "Rates_Summed") or spinsys.Rates_Summed is None:
        bath.calc_Rates(spinsys)

    Rates_Summed = spinsys.Rates_Summed
    N_levels = spinsys.dimensionOfMatrix

    # Only determine a cutoff when filtering is explicitly requested
    threshold_value = None

    if threshold_percent is not None:
        if not 0 <= threshold_percent <= 1:
            raise ValueError("threshold_percent must be None or between 0 and 1")

        # Exclude elastic diagonal rates from the reference maximum
        inelastic_mask = ~np.eye(N_levels, dtype=bool)
        positive_rates = Rates_Summed[
            inelastic_mask & np.isfinite(Rates_Summed) & (Rates_Summed > 0)
        ]

        if positive_rates.size > 0:
            threshold_value = threshold_percent * np.max(positive_rates)

    c_ops_T1 = []

    for i_initial in range(N_levels):
        for j_final in range(N_levels):
            if i_initial == j_final:
                continue

            rate = Rates_Summed[i_initial, j_final]

            if not np.isfinite(rate) or rate <= 0:
                continue

            if threshold_value is not None and rate < threshold_value:
                continue

            E_op = (
                Qobj(spinsys.eigVectors[:, j_final])
                * Qobj(spinsys.eigVectors[:, i_initial]).dag()
            )

            # Rates are in 1/s; convert collapse amplitudes to 1/sqrt(ns)
            L_ij = np.sqrt(rate / 1e9) * E_op
            c_ops_T1.append(L_ij)

    return c_ops_T1


def initialize_thermal(spinsys: SpinSys):
    """
    Initialize the spin system in a thermal state.

    This function initializes the spin system in a thermal state based on its
    eigenenergies and temperature using the Boltzmann distribution.

    Parameters
    ----------
    spinsys : SpinSys
        The spin system object containing eigenenergies and temperature.

    Returns
    -------
    Qobj
        The thermal density matrix of the spin system in the original basis.

    Notes
    -----
    The function assigns the thermal density matrix to ``spinsys.psi0`` for later use.
    """
    if not hasattr(spinsys, "E_All") or spinsys.E_All is None:
        hamil.build_Hamiltonian(spinsys)
        hamil.calc_EigEnergies(spinsys)

    if not hasattr(spinsys, "eigVectors") or spinsys.eigVectors is None:
        hamil.calc_EigStates(spinsys)

    probabilities = bath.calc_ThermalPopulations(spinsys)

    # Construct the thermal density matrix in the product-spin basis
    rho_thermal = (
        spinsys.eigVectors @ np.diag(probabilities) @ spinsys.eigVectors.conj().T
    )

    spinsys.psi0 = Qobj(rho_thermal)

    return Qobj(rho_thermal)


def calc_TimeEvolution(
    spinsys: SpinSys,
    psi_0: list,
    time_array: np.ndarray,
    spin_idx_array: list[int] = "all",
    Saxis: str = "z",
    options: dict = None,
):
    """
    Solve the Lindblad master equation with QuTiP and return mesolve result.

    This function computes the time evolution of a given spin system under the Lindblad
    master equation using QuTiP's `mesolve`. It calculates two timesthe expectation
    values of the spin operators for specified spins over a defined time range.

    Parameters
    ----------
    spinsys : SpinSys
        Spin system with Hamiltonian, eigenstates, and spin operators.
    psi_0 : Qobj, array or np.array
        Initial state vector (ket) for the time evolution.
        Can also be a density matrix of the same dimension as the Hamiltonian.
    time_array : np.ndarray
        Array of time values in ns for which the Lindblad equation should be solved.
    spin_idx_array : list[int], optional
        Indices of spins for which the expectation values over time are computed.
        Default is ``"all"``, in which case the expectation values of all spins in
        spinsys are calculated.
    Saxis : {"x", "y", "z"}, optional
        Axis along which the expectation value should be computed. The expectation value
        of Sx, Sy or Sz are computed depending on axis.
        Default is ``"z"``.
    options: dict, optional
        Options to give to the QuTiP mesolve solver.
        See possible options in the QuTiP documentation.

    Returns
    -------
    qutip.solver.Result
        QuTiP result object with ``expect`` and ``states``
        if 'store_states':True in options.

    Raises
    ------
    ValueError
        If ``Saxis`` is not one of ``"x"``, ``"y"``, or ``"z"``
        or spinsys.S(x, y or z) are not defined.

    Notes
    -----
    To avoid 'IntegratorException', 'nsteps' gets set to 20000 and the solver method
    to 'bdf' if such an exception occurs. This can take up
    some calculation time.
    You can also increase 'nsteps' manually with the options parameter
    or increase N_times.

    This function adds the following attributes to the given spinsys:
    - spinsys.timeEvolution_result : the QuTiP result of mesolve
    - spinsys.timeEvolution_spinidx : spin_idx
    - spinsys.timeEvolution_Saxis : Saxis
    """
    # convert psi_0 into Qobj if it is not already one
    if not isinstance(psi_0, Qobj):
        psi_0 = Qobj(psi_0)

    # Ensure the Lindblad collapse operators are calculated before using mesolve
    if not hasattr(spinsys, "collapse_ops") or spinsys.collapse_ops is None:
        spinsys.collapse_ops = calc_collapseOperators(spinsys)

    # set default options for mesolve if not specified otherwise
    if options is None:
        # default options for mesolve. Set here to avoid mutable default argument.
        solver_options = {"store_states": True, "method": "diag"}
    else:
        solver_options = options

    # convert Hamiltonian to Qobj and convert from meV to angular frequency in GHz
    H_q = Qobj(spinsys.H * spinsys.meVtoGHzConversion * 2 * np.pi)

    # calculate list of operators e_ops for which the Lindblad equation will be solved
    S = getattr(spinsys, f"S{Saxis}", None)  # gets Sx, Sy or Sz depending on Saxis
    if S is None:
        raise ValueError(
            'Saxis must be "x", "y" or "z" and the angular momentum '
            "matrices must have been calculated."
        )

    # initialize list of operators for the expectation values
    e_ops = []
    # calculate S matrix for each spin in spin_idx
    if spin_idx_array == "all":
        spin_idx_array = np.arange(0, spinsys.NSpins, 1)  # use all spins in spinsys
    for spin in spin_idx_array:
        S_spin = 2 * S[spin]
        e_ops.append(Qobj(S_spin))

    # solve Lindblad equation, and retry with increased nsteps if needed
    try:
        result = mesolve(
            H=H_q,
            rho0=psi_0,
            tlist=time_array,
            c_ops=spinsys.collapse_ops,
            e_ops=e_ops,
            options=solver_options,
        )
    except Exception as exc:
        if "Excess work done on this call" not in str(exc):
            raise  # show original QuTiP error message

        retry_options = dict(solver_options)
        retry_options["method"] = retry_options.get("method", "bdf")
        # increase nsteps to 20000 to have higher chance to get a result from the solver
        retry_options["nsteps"] = max(int(retry_options.get("nsteps", 1000)), 20000)
        if len(time_array) > 1:
            retry_options.setdefault("max_step", float(time_array[1] - time_array[0]))

        # use the same collapse operators as originally passed (may be sparse)
        result = mesolve(
            H_q,
            psi_0,
            time_array,
            c_ops=spinsys.collapse_ops,
            e_ops=e_ops,
            options=retry_options,
        )

    # add results as attributes to spinsys object
    spinsys.timeEvolution_result = result
    spinsys.timeEvolution_spinidx = spin_idx_array.copy()
    spinsys.timeEvolution_Saxis = Saxis

    return result


def calc_TimeEvolution_Tipfield(
    spinsys: SpinSys,
    psi_0: list,
    time_array: np.ndarray,
    BTip_array: np.ndarray,
    Baxis: str = "z",
    Saxis: str = "z",
    spin_idx: int = 0,
    options: dict = None,
    threshold_percent: float | None = None,
):
    """
    Solve the Lindblad master equation for different BTip.

    This function computes the time evolution of a given spin system under the Lindblad
    master equation using QuTiP's `mesolve` for different values of the tip field along
    a specified axis.

    Parameters
    ----------
    spinsys : SpinSys
        The spin system object
    psi_0 : Qobj, array or np.array
        Initial state of spinsys. Can also be a density matrix.
    time_array : np.ndarray
        Array of time values in ns for which the Lindblad equation should be solved.
    BTip_array : np.ndarray
        Array of tip field values in T for which the Lindblad equation should be solved.
    Baxis: string, optional
        which B axis ("x", "y", or "z") should be swept through.
        Default is "z".
    Saxis: string, optional
        For which S axis ("x", "y", or "z") the expectation value should be calculated.
        Default is "z".
    spin_idx: int, optional
        Which single spin of the spinsys is plotted.
        Default is 0 (which is typically the readout spin)
    options: dict, optional
        Options to give to the QuTiP mesolve solver. See possible options in the QuTiP
        documentation.
    threshold_percent : float or None, optional
        Fraction of the largest positive, finite inelastic rate below which
        transitions are omitted when constructing collapse operators.
        Must be between 0 and 1; for example, 0.01 corresponds to 1%.
        The cutoff is calculated separately at each tip field, excluding
        elastic diagonal rates. Default is None, which disables
        rate-threshold filtering.


    Returns
    -------
    resulti: numpy.ndarray
        2D array of shape (len(BTip_values), len(times)) containing <2S_{Saxis}>(t)
        for each time and tip field value.

    Raises
    ------
    ValueError
        If Baxis is not "x", "y" or "z".
    ValueError
        If Saxis is not "x", "y" or "z" or spinsys.S(x, y or z) are not defined.

    Notes
    -----
    The time evolution results for the different tip field strengths are calculated
    in parallel using _calcDecayParallelHelper.
    """
    # save initial value for later
    BTip_initial = spinsys.BTip.copy()

    # convert psi_0 into Qobj if it is not already one
    if not isinstance(psi_0, Qobj):
        psi_0 = Qobj(psi_0)

    # convert axis to sweep through to numerical value for indexing
    if Baxis == "z":
        Bidx = 2
    elif Baxis == "y":
        Bidx = 1
    elif Baxis == "x":
        Bidx = 0
    else:
        raise ValueError('Baxis must be "x", "y" or "z".')

    # define Sx/Sy/Sz of the spin of which the expectation value will be calculated
    if Saxis == "x":
        S = Qobj(2 * spinsys.Sx[spin_idx])  # *2 to have values from -1 to 1
    elif Saxis == "y":
        S = Qobj(2 * spinsys.Sy[spin_idx])  # *2 to have values from -1 to 1
    elif Saxis == "z":
        S = Qobj(2 * spinsys.Sz[spin_idx])  # *2 to have values from -1 to 1
    else:
        raise ValueError(
            'Saxis must be "x", "y" or "z" and the angular momentum '
            "matrices must have been calculated."
        )

    # set default options for mesolve if not specified otherwise
    if options is None:
        # default options for mesolve.
        solver_options = {"store_states": False, "method": "diag"}
    else:
        solver_options = options

    # compute <Sz(t)> for each tip field strength
    resulti = Parallel(n_jobs=-1)(
        delayed(_calc_TipfieldParallelHelper)(
            spinsys,
            S,
            BTip_value,
            Bidx,
            psi_0,
            time_array,
            solver_options,
            threshold_percent=threshold_percent,
        )
        for BTip_value in BTip_array
    )
    resulti = np.stack(resulti)

    # add results as attributes to spinsys object for later use
    spinsys.timeEvolution_Tipfield_result = resulti
    spinsys.timeEvolution_Tipfield_time_array = time_array
    spinsys.timeEvolution_Tipfield_BTip_array = BTip_array
    spinsys.timeEvolution_Tipfield_spinidx = spin_idx
    spinsys.timeEvolution_Tipfield_Saxis = Saxis
    spinsys.timeEvolution_Tipfield_Baxis = Baxis

    # Restore the initial BTip-Vector
    spinsys.BTip = BTip_initial

    return resulti


def _calc_TipfieldParallelHelper(
    spinsys: SpinSys,
    S: Qobj,
    BTip_value: float,
    Bidx: int,
    psi_0: Qobj,
    time_array: np.ndarray,
    solver_options: dict,
    threshold_percent: float | None,
):
    """
    Solve the Lindblad master equation for a BTip value.

    This function is a helper for `calc_TimeEvolution_Tipfield`. It computes the time
    evolution of a given spin system under the Lindblad master equation using QuTiP's
    `mesolve` for a single value of the tip field along a specified axis.
    It is designed to be used in parallel computation.

    Parameters
    ----------
    spinsys : SpinSys
        The spin system object.
    S: Qobj
        Spin matrix (Sx, Sy or Sz) of the spin of which the expectation value over time
        should be calculated.
    BTip_value: float
        Tip field strength that should be inserted into the component along the axis
        Bidx
    Bidx: int {0, 1, 2}
        Index of the BTip axis along which the value will be changed to BTip_value.
        0 corresponds to x, 1 to y and 2 to z.
    psi_0 : Qobj
        Initial state or density matrix of spinsys.
    time_array : np.ndarray
        Time values in seconds for which the Lindblad equation should be solved.
    solver_options: dict
        Options to give to the QuTiP mesolve solver.

    Returns
    -------
    resulti: numpy.ndarray
        Array containing <S>(t) for each time value.
    """
    # create copy of spinsys to change its BTip only locally
    local_spinsys = deepcopy(spinsys)
    # change BTip value along Bidx axis
    local_spinsys.BTip = np.asarray(local_spinsys.BTip, dtype=float)
    local_spinsys.BTip[Bidx] = BTip_value

    # calculate new Hamiltonian, eigenenergies and eigenstates
    hamil.build_Hamiltonian(local_spinsys)
    hamil.calc_EigEnergies(local_spinsys)
    # calculate new collapse operators with new Hamiltonian
    c_ops = calc_collapseOperators(local_spinsys, threshold_percent=threshold_percent)

    # convert Hamiltonian to Qobj and convert from meV to angular frequency in GHz
    H_q = Qobj(local_spinsys.H * spinsys.meVtoGHzConversion * 2 * np.pi)

    # calculate time evolution for given tip field
    result = mesolve(H_q, psi_0, time_array, c_ops, e_ops=[S], options=solver_options)
    result_expval = np.real(result.expect[0])  # calculated S expectation value

    return result_expval


def calc_DensityMatrix(
    spinsys: SpinSys,
    time_type: str,
    time_index: list[int] = None,
    time_values: list[float] = None,
):
    """
    Return the density matrices at selected times (values or indices) from a result.

    This function retrieves the density matrices corresponding to specified time indices
    or time values from a previously computed time evolution result stored in the
    `Spinsys` object.

    Parameters
    ----------
    spinsys : SpinSys
        Spin system containing time evolution attributes.
    time_type : {"index", "value"}
        Selects whether the time is given by an explicit index or by a
        floating-point time value to be matched to the closest entry in
        ``result.times``.
    time_index : list[int], optional
        Indices to ``result.times`` when ``time_type == "index"``.
    time_values : list[float], optional
        Time values (in ns) used to find the closest entry in
        ``result.times`` when ``time_type == "value"``.

    Returns
    -------
    times_densmats : list[float]
        Selected simulation times in ns, in the order requested.
        These are stored time points and may differ from requested time values.
    densmats : list[numpy.ndarray]
        Density matrices corresponding to times_densmats. Each matrix has
        shape (dimensionOfMatrix, dimensionOfMatrix), in the same basis as
        the stored states. For calc_TimeEvolution, this is the product-spin
        basis. Stored kets are converted to density matrices.
    result_indices : list[int]
        Indices of the selected entries in result.times and result.states.
        For time_type="index", these are the supplied time_index values.

    Raises
    ------
    ValueError
        If ``time_type`` is "index" and ``time_index`` is not provided, or if
        ``time_type`` is "value" and ``time_values`` is not provided.
    ValueError
        If time_type is not "value" or "index".
    AttributeError

    """
    # get time evolution result stored in spinsys
    if not hasattr(spinsys, "timeEvolution_result"):
        raise AttributeError(
            "The SpinSys object is missing the attribute "
            "timeEvolution_result. Run lind.calc_TimeEvolution first."
        )
    result = spinsys.timeEvolution_result
    times_result = result.times

    # initialize list to store time values for which density matrices are calculated
    times_densmats = []
    # initialize list to store density matrices for each requested time value
    densmats = []

    # choose state and time of result corresponding to time_index
    if time_type == "index":
        if time_index is None:  # make sure a time_index is given
            raise ValueError('time_index must be provided when time_type == "index"')
        for time_index_i in time_index:
            state_timestep = result.states[time_index_i]
            if state_timestep.isket:
                densmat = state_timestep.proj().full()
            else:
                densmat = state_timestep.full()
            time = times_result[time_index_i]
            times_densmats.append(time)
            densmats.append(densmat)
        result_indices = time_index

    # choose state and time at the time closest to time_value
    elif time_type == "value":
        # initialize list to store indices corresponding to time_values
        result_indices = []

        if time_values is None:  # make sure a time_value is given
            raise ValueError('time_value must be provided when time_type == "value"')

        # loop through time_values and find closest times in result.times
        for time_value_i in time_values:
            # find indices corresponding to time_value_i in times_result.
            # take first time step if time_value is smaller than the first time
            # in result.times
            if time_value_i < times_result[0]:
                time_index = 0

            elif time_value_i > times_result[-1]:
                # take last time step if time_value is higher than the last time
                # in result.times
                time_index = len(times_result) - 1

            else:
                # time_index is index at which times_result[time_index-1] < time_value_i
                # <= times_result[time_index]
                time_index = np.searchsorted(times_result, time_value_i, side="left")

                diff_current = abs(times_result[time_index] - time_value_i)
                diff_prev = abs(time_value_i - times_result[time_index - 1])
                if diff_current > diff_prev and time_index != 0:
                    # take time_index-1 in case times_result[time_index-1] is closer
                    #  to time_value_i than times_result[time_index]
                    time_index -= 1

            # add density matrix and corresponding time to the lists
            state_timestep = result.states[time_index]
            if state_timestep.isket:
                densmat = state_timestep.proj().full()
            else:
                densmat = state_timestep.full()
            time = times_result[time_index]
            times_densmats.append(time)
            densmats.append(densmat)
            result_indices.append(time_index)

    else:
        raise ValueError('time_type must be "value" or "index"')

    return times_densmats, densmats, result_indices


def calc_TunnelCurrent_time(
    spinsys: SpinSys,
):
    """
    Calculate the tunneling current over time from a QuTiP time-evolution result.

    This function computes the tunneling current at each time point from a previously
    computed time evolution result stored in the `spinsys` object. The current is
    derived from the density matrices and the tunneling rates.

    Parameters
    ----------
    spinsys : SpinSys
        Spin system containing Rates and timeEvolution_result.

    Returns
    -------
    Current : np.ndarray
        Current values at each time point in ``result.times``.

    Raises
    ------
    AttributeError
        If ``spinsys`` does not have ``timeEvolution_result``. Run
        ``calc_TimeEvolution`` first.

    Notes
    -----
    - The function assumes that required attribute `Rates` are available in the provided
      `spinsys` object.
    - The current is calculated from the SpinSys attribute timeEvolution_result that
       needs to be calculated in calc_TimeEvolution first.
    """
    if not hasattr(spinsys, "timeEvolution_result"):
        raise AttributeError(
            "The SpinSys object is missing the attribute timeEvolution_result. "
            "Run lind.calc_TimeEvolution first."
        )

    result = spinsys.timeEvolution_result
    states = result.states  # density matrices at each time

    # Derive the tunneling current at each time step using Equation 41 from Ternes paper
    Current = []

    for state_timestep in states:
        # Convert kets to density matrices in the product-spin basis
        if state_timestep.isket:
            densmat_timestep = state_timestep.proj().full()
        else:
            densmat_timestep = state_timestep.full()

        # Extract populations in the energy eigenbasis
        populations = np.real(
            np.diag(spinsys.eigVectors.conj().T @ densmat_timestep @ spinsys.eigVectors)
        )

        I_timeidx = 0

        for i in range(spinsys.dimensionOfMatrix):
            for j in range(spinsys.dimensionOfMatrix):
                I_temp = (
                    const.e
                    * populations[i]
                    * (spinsys.Rates[0, i, j] - spinsys.Rates[1, i, j])
                )
                I_timeidx += I_temp

        Current.append(I_timeidx)

    Current = np.array(Current)

    # Add the spin-independent background current
    Current = np.add(
        Current,
        spinsys.b0 * spinsys.G_st * spinsys.V_DC * 1e-3,
    )

    return Current


# endregion Calculations

# region Visualization


def plot_TimeEvolution(
    spinsys: SpinSys, spin_idxToPlot: list[int] = "all", show: str | None = None
):
    """
    Plot spin expectation values over time from a previously computed time evolution.

    This function visualizes the time evolution of spin expectation values for a given
    spin system.

    Parameters
    ----------
    spinsys : SpinSys
        Spin system containing ``timeEvolution_result`` produced
        by `calc_TimeEvolution`.
    spin_idxToPlot : {"all"} or sequence of int, optional
        Spin indices to include in the plot. If ``"all"``, all spin indices for which
        the time evolution was computed are plotted.

    show : {"fixed", "responsive", "no"} or None, optional
        Display mode. ``None`` uses the global setting, initially ``"fixed"``.
        ``"fixed"`` uses the preferred width; ``"responsive"`` fills the
        container width. ``"no"`` suppresses display without skipping calculations.

    Returns
    -------
    return_dict : dict
        ``fig``
            Plotly figure object.
        ``time_values``
            Time grid used in the underlying simulation.
        ``expectation_values``
            Spin expectation values.
        ``Saxis``
            Axis along which the expectation value <S_Saxis> was computed.
        ``spin_idx_plotted``
            Positions of the plotted spins within ``spinsys.timeEvolution_spinidx``.

    Raises
    ------
    AttributeError
        If ``spinsys.timeEvolution_result`` is missing.
    ValueError
        If requested ``spin_idxToPlot`` entries are not present in the spins
        ``spinsys.timeEvolution_spinidx`` for which the time evolution was computed.

    Notes
    -----
    This function does not run a new time evolution. It only visualizes data previously
    computed by `calc_TimeEvolution`.
    """
    # check if the time evolution was calculated
    if (
        not hasattr(spinsys, "timeEvolution_result")
        or spinsys.timeEvolution_result is None
    ):
        raise AttributeError(
            "timeEvolution_result is not defined. The time evolution must first be "
            "calculated via calc_TimeEvolution."
        )

    result = spinsys.timeEvolution_result
    # times for which the time evolution was calculated
    times_calc = result.times
    # indices of the spins for which the time evolution was calculated
    spin_idx_calc = spinsys.timeEvolution_spinidx
    Saxis = spinsys.timeEvolution_Saxis

    # select spins to plot and check if their time evolutions were calculated
    if spin_idxToPlot == "all":
        # If not specified, the expectation values of all spins for which the time
        # evolution was calculated are plotted.
        spin_idxToPlot = spin_idx_calc

    # check that requested spin indices exist in result and map to
    # positions in spin_idx_calc
    # list that contains missing spin indices
    missing = [s for s in spin_idxToPlot if s not in spin_idx_calc]
    if missing:  # non-empty list is considered True
        raise ValueError(
            "spin_idxToPlot contains indices not present in "
            f"timeEvolution_spinidx: {missing}"
        )

    # Plot
    fig = go.Figure()
    # indices of spin_idxToPlot values in spin_idx_calc
    plot_idx = [i for i, s in enumerate(spin_idx_calc) if s in spin_idxToPlot]
    for i in plot_idx:  # plot expectation value for each spin
        fig.add_trace(
            go.Scatter(
                x=times_calc,
                y=result.expect[i],
                mode="lines",
                name=f"n={spin_idx_calc[i]}",
            )
        )

    fig.update_layout(
        title="Time evolution",
        xaxis_title="time (ns)",
        yaxis_title=f"<S{Saxis}> (norm.)",
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig.update_xaxes(range=[times_calc[0], times_calc[-1]])  # full time range
    fig.update_yaxes(range=[-1, 1])  # <S> from -1 to 1
    plotting.show_figure(fig, preferred_width=800, show=show)

    return_dict = {
        "fig": fig,
        "time_values": times_calc,
        "expectation_values": result.expect,
        "Saxis": Saxis,
        "spin_idx_plotted": plot_idx,
    }

    return return_dict


def plot_TimeEvolution_Tipfield(spinsys: SpinSys, show: str | None = None):
    """
    Plot time evolution as a function of tip-field strength.

    This function visualizes the time evolution of spin expectation values for a given
    spin system as a 2D colormap, where the x-axis represents time and the y-axis
    represents the tip-field strength. The color intensity indicates the expectation
    value of the spin along a specified axis.

    Parameters
    ----------
    spinsys
        SpinSys instance containing results from calc_TimeEvolution_TipField.
    show : {"fixed", "responsive", "no"} or None, optional
        Display mode. ``None`` uses the global setting, initially ``"fixed"``.
        ``"fixed"`` uses the preferred width; ``"responsive"`` fills the
        container width. ``"no"`` suppresses display without skipping calculations.

    Returns
    -------
    dict
        Dictionary with keys:
        ``fig``: Plotly figure object,
        ``time_values``: array of times used in simulation,
        ``BTip_values``: array of tip-field values (same units as stored),
        ``S``: 2D array of expectation values plotted.

    Raises
    ------
    AttributeError
        If ``spinsys.timeEvolution_Tipfield_result`` is missing.

    Notes
    -----
    This function only visualizes previously computed data from
    ``calc_TimeEvolution_TipField`` and does not perform any new simulations.
    """
    # check if the time evolution was calculated
    if (
        not hasattr(spinsys, "timeEvolution_Tipfield_result")
        or spinsys.timeEvolution_Tipfield_result is None
    ):
        raise AttributeError(
            "timeEvolution_Tipfield result is not defined. The time evolution"
            "must first be calculated via calc_TimeEvolution_TipField."
        )

    resulti = spinsys.timeEvolution_Tipfield_result
    # times for which the time evolution was calculated
    time_array = spinsys.timeEvolution_Tipfield_time_array
    # indices of the spins for which the time evolution was calculated
    BTip_array = spinsys.timeEvolution_Tipfield_BTip_array
    Saxis = spinsys.timeEvolution_Tipfield_Saxis
    # tip field axis used
    Baxis = spinsys.timeEvolution_Tipfield_Baxis

    # Plot as a 2D colormap: x=time, y=tip field values
    fig = go.Figure(
        data=go.Heatmap(
            z=resulti,
            x=time_array,
            y=BTip_array * 1e3,
            zmin=-1,
            zmax=1,
            colorscale="RdBu",
            reversescale=True,
            colorbar=dict(title=f"<S{Saxis}>"),
            zsmooth=False,
        )
    )
    fig.update_layout(
        xaxis_title="time (ns)",
        yaxis_title=f"BTip_{Baxis} (mT)",
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig.update_xaxes(range=[time_array[0], time_array[-1]])  # full time range
    plotting.show_figure(fig, preferred_width=800, show=show)

    returndict = {
        "fig": fig,
        "time_values": time_array,
        "BTip_values": BTip_array,
        "S": resulti,
    }

    return returndict


def plot_DensityMatrix(
    spinsys: SpinSys,
    time_type: str,
    time_index: list[int] = None,
    time_values: list[float] = None,
    show: str | None = None,
):
    """
    Plot the real part of the density matrix at selected times as a 3D bar chart.

    This function visualizes the real part of the density matrix at specified time
    indices or time values from a previously computed time evolution result stored in
    the `spinsys` object. The density matrices are displayed as 3D bar plots, with the
    x and y axes representing the eigenstates of the Hamiltonian and the z axis
    representing the real part of the density matrix elements.

    Parameters
    ----------
    spinsys : SpinSys
        Spin system providing the time evolution result with density matrices and state
        labels used for axis tick text.
    time_type : {"index", "value"}
        Selects whether the time is given by an explicit index or by a
        floating-point time value to be matched to the closest entry in
        ``result.times``.
    time_index : int, optional
        Index to ``result.times`` when ``time_type == "index"`` at which
        ``result.states`` density matrix is plotted.
    time_values : float, optional
        Time values (in ns) used to find the closest entry in
        ``result.times`` when ``time_type == "value"``.

    show : {"fixed", "responsive", "no"} or None, optional
        Display mode. ``None`` uses the global setting, initially ``"fixed"``.
        ``"fixed"`` uses the preferred width; ``"responsive"`` fills the
        container width. ``"no"`` suppresses display without skipping calculations.

    Returns
    -------
    return_dict : dict
    A dictionary containing:
    - 'fig': plotly.graph_objects.Figure
    - 'times': np.ndarray
        The times at which the density matrices are plotted.
    - 'density_matrices': list of np.ndarray
        The density matrices that are plotted.
    - 'result_indices': list[int]
        The indices in ``result.times`` and ``result.states`` corresponding to
        the plotted density matrices.

    Raises
    ------
    AttributeError
        If ``spinsys.timeEvolution_result`` is missing.
    """
    if not hasattr(spinsys, "timeEvolution_result"):
        raise AttributeError(
            "The SpinSys object is missing the attribute "
            "timeEvolution_result. Run lind.calc_TimeEvolution first."
        )

    times, densmats, result_indices = calc_DensityMatrix(
        spinsys, time_type, time_index, time_values
    )

    # Plot as a grid of 3D bar plots.
    n = len(densmats)
    ncols = min(3, n)  # max 3 plots per row
    nrows = int(np.ceil(n / ncols))
    subplot_titles = [f"Density Matrix at t = {time:g} ns" for time in times]
    fig = make_subplots(
        rows=nrows,
        cols=ncols,
        specs=[[{"type": "scene"} for _ in range(ncols)] for _ in range(nrows)],
        subplot_titles=subplot_titles,
    )

    for k, densmat in enumerate(densmats):
        densmat = np.real(densmat)  # take real part of density matrix for plotting
        vertices_x, vertices_y, vertices_z = [], [], []
        triangles_i, triangles_j, triangles_k = [], [], []
        face_colors = []
        hover_data = []
        cube_faces = (
            (0, 1, 2),
            (0, 2, 3),
            (4, 6, 5),
            (4, 7, 6),
            (0, 4, 5),
            (0, 5, 1),
            (1, 5, 6),
            (1, 6, 2),
            (2, 6, 7),
            (2, 7, 3),
            (3, 7, 4),
            (3, 4, 0),
        )
        labels = list(spinsys.statesWithoutE)
        n_y, n_x = densmat.shape

        for y_idx, x_idx in np.ndindex(densmat.shape):
            value = float(densmat[y_idx, x_idx])
            color = "lightgrey" if value == 0 else ("red" if value < 0 else "blue")
            # Leave a small gap between bars while keeping each bar aligned with
            # the matrix cell centred on its integer state index.
            x0, x1 = x_idx - 0.35, x_idx + 0.35
            y0, y1 = y_idx - 0.35, y_idx + 0.35
            z0, z1 = 0.0, value
            cube_vertices = (
                (x0, y0, z0),
                (x1, y0, z0),
                (x1, y1, z0),
                (x0, y1, z0),
                (x0, y0, z1),
                (x1, y0, z1),
                (x1, y1, z1),
                (x0, y1, z1),
            )
            offset = len(vertices_x)
            for x_coord, y_coord, z_coord in cube_vertices:
                vertices_x.append(x_coord)
                vertices_y.append(y_coord)
                vertices_z.append(z_coord)
                hover_data.append((labels[x_idx], labels[y_idx], value))
            for face in cube_faces:
                triangles_i.append(offset + face[0])
                triangles_j.append(offset + face[1])
                triangles_k.append(offset + face[2])
                face_colors.append(color)

        row, col = divmod(k, ncols)
        fig.add_trace(
            go.Mesh3d(
                x=vertices_x,
                y=vertices_y,
                z=vertices_z,
                i=triangles_i,
                j=triangles_j,
                k=triangles_k,
                facecolor=face_colors,
                customdata=hover_data,
                flatshading=True,
                showscale=False,
                hovertemplate=(
                    "State x: %{customdata[0]}<br>"
                    "State y: %{customdata[1]}<br>"
                    "Real density: %{customdata[2]:.6g}<extra></extra>"
                ),
            ),
            row=row + 1,
            col=col + 1,
        )

        scene_name = "scene" if k == 0 else f"scene{k + 1}"
        fig.update_layout(
            **{
                scene_name: dict(
                    xaxis=dict(
                        title="State",
                        tickvals=list(range(len(labels))),
                        ticktext=labels,
                        range=[-0.5, n_x - 0.5],
                    ),
                    yaxis=dict(
                        title="State",
                        tickvals=list(range(len(labels))),
                        ticktext=labels,
                        range=[-0.5, n_y - 0.5],
                    ),
                    zaxis=dict(title="Real Part of Density Matrix", range=[-1, 1]),
                    aspectmode="manual",
                    aspectratio=dict(x=1, y=1, z=0.8),
                )
            }
        )

    fig.update_layout(
        margin=dict(l=20, r=20, t=70, b=20),
        paper_bgcolor="white",
        showlegend=False,
    )
    plotting.show_figure(fig, preferred_width=1000, show=show)

    return_dict = {
        "fig": fig,
        "times": times,
        "density_matrices": densmats,
        "result_indices": result_indices,
    }

    return return_dict


def plot_TunnelCurrent_time(spinsys: SpinSys, show: str | None = None):
    """
    Plot the tunneling current as a function of time.

    This function visualizes the tunneling current over time for a given spin system,
    using the results from a previously computed time evolution.
    The current is calculated using the `calc_TunnelCurrent_time` function

    Parameters
    ----------
    spinsys : SpinSys
        Spin system containing rate matrices, tunneling parameters and
        timeEvolution_result containing a QuTiP result object returned by
        ``mesolve`` with ``states`` and ``times``.

    show : {"fixed", "responsive", "no"} or None, optional
        Display mode. ``None`` uses the global setting, initially ``"fixed"``.
        ``"fixed"`` uses the preferred width; ``"responsive"`` fills the
        container width. ``"no"`` suppresses display without skipping calculations.

    Returns
    -------
    return_dict : dict
        Dictionary containing:
        ``fig`` : plotly.graph_objects.Figure
            The created figure.
        ``time_array`` : numpy.ndarray
            Time values of the whole simulated time interval
        ``Current_array`` : np.ndarray

    Raises
    ------
    AttributeError
        If ``spinsys.timeEvolution_result`` is missing.
        Run ``calc_TimeEvolution`` first.
    """
    if not hasattr(spinsys, "timeEvolution_result"):
        raise AttributeError(
            "The SpinSys object is missing the attribute "
            "timeEvolution_result. Run lind.calc_TimeEvolution first."
        )

    result = spinsys.timeEvolution_result
    time_array = result.times
    Current = calc_TunnelCurrent_time(spinsys)

    fig = go.Figure(
        data=go.Scatter(
            x=time_array,
            y=Current,
            mode="lines",
            name="Tunneling current",
        )
    )
    fig.update_layout(
        title="Tunneling current",
        xaxis_title="time (ns)",
        yaxis_title="I (A)",
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=False,
    )
    fig.update_xaxes(range=[time_array[0], time_array[-1]])  # full time range
    plotting.show_figure(fig, preferred_width=800, show=show)

    return_dict = {
        "fig": fig,
        "time_array": time_array,
        "Current_array": Current,
    }

    return return_dict


def plot_FourierFreqs(
    spinsys: SpinSys, spin_idxToPlot: list[int] = "all", show: str | None = None
):
    """
    Plot the frequency spectrum from an existing time evolution result.

    This function visualizes the frequency spectrum of spin expectation values obtained
    from a previously computed time evolution result. It computes the Fourier transform
    of the expectation values and plots the resulting frequency spectrum for specified

    Parameters
    ----------
    spinsys : SpinSys
        Spin system containing ``timeEvolution_result`` produced by
        ``calc_TimeEvolution``.
    spin_idxToPlot : {"all"} or sequence of int, optional
        Spin indices to plot. If ``"all"``, all spins in
        ``spinsys.timeEvolution_spinidx`` are plotted.

    show : {"fixed", "responsive", "no"} or None, optional
        Display mode. ``None`` uses the global setting, initially ``"fixed"``.
        ``"fixed"`` uses the preferred width; ``"responsive"`` fills the
        container width. ``"no"`` suppresses display without skipping calculations.

    Returns
    -------
    return_dict : dict
        A dictionary containing:
        - ``fig`` : plotly.graph_objects.Figure
            The created figure.
        - ``frequencies`` : list[numpy.ndarray]
            Frequency arrays for each plotted spin (same order as
            ``spin_indices``).
        - ``amplitudes`` : list[numpy.ndarray]
            Complex, one-sided amplitude-scaled FFT coefficients.
            Their units match the stored expectation values.
            The figure displays their magnitudes.
        - ``frequency_peaks`` : list[numpy.ndarray]
            Peak frequencies for each plotted spin. Only peaks with an amplitude of
            at least 1% of the maximum amplitude are considered.
        - ``spin_indices`` : list[int]
            Spin indices corresponding to each spectrum in the plot.

    Raises
    ------
    AttributeError
        If ``spinsys.timeEvolution_result`` is missing. Run
        ``calc_TimeEvolution`` first.
    ValueError
        If requested ``spin_idxToPlot`` entries are not present in the spins
        ``spinsys.timeEvolution_spinidx`` for which the time evolution was computed.

    Notes
    -----
    - Expectation values are real, so a real FFT (``rfft``) is used.
    - The mean value of each spin's <S> is subtracted before the FFT to
      remove the DC offset (otherwise a large peak appears at zero frequency).
    - This function does not recompute time evolution; it uses the cached
      ``spinsys.timeEvolution_result``.
    """
    # calculate the time evolution result to be Fourier transformed if necessary
    if (
        not hasattr(spinsys, "timeEvolution_result")
        or spinsys.timeEvolution_result is None
    ):
        raise AttributeError(
            "The SpinSys object has no attribute `timeEvolution_result`. "
            "Run lind.calc_TimeEvolution first."
        )

    # get data to Fourier transform from time evolution result
    result = spinsys.timeEvolution_result  # result with expectation values
    S_expvals = result.expect
    times = result.times
    spin_idx_array = spinsys.timeEvolution_spinidx

    # select indices of spins to plot if not specified
    if spin_idxToPlot == "all":
        spin_idxToPlot = spin_idx_array

    # check that requested spin indices exist in result and map to positions in spin_idx
    missing = [s for s in spin_idxToPlot if s not in spin_idx_array]
    if missing:  # non-empty list is considered True
        raise ValueError(
            "spin_idxToPlot contains indices not present in "
            f"timeEvolution_spinidx: {missing}"
        )

    # indices of spin_idxToPlot values in spin_idx
    plot_idx = [i for i, s in enumerate(spin_idx_array) if s in spin_idxToPlot]

    # get frequencies from calcFourierFreqs
    freqs = []
    amps = []
    labels = []

    for idxToPlot in plot_idx:
        S_values = np.real(S_expvals[idxToPlot])
        # remove offset (otherwise big peak at zero frequency)
        S_values = S_values - np.mean(S_values)
        freqs_i, amps_i = hf.calc_fft(
            times=times,
            signal=S_values,
        )
        freqs.append(freqs_i)
        amps.append(amps_i)
        labels.append(spin_idx_array[idxToPlot])

    # calculate frequency peaks
    freq_peaks = []
    for freqs_i, amps_i in zip(freqs, amps, strict=True):
        peak_idx, _ = scipy.signal.find_peaks(
            np.abs(amps_i),
            # only consider peaks that are at least 1% of the maximum amplitude
            height=0.01 * np.max(np.abs(amps_i)),
        )
        freq_peaks.append(freqs_i[peak_idx])

    # calculate frequency axis limits for plotting
    # get maximum peak frequencies across all spectra
    max_peak_freq = [max(peaks) for peaks in freq_peaks if len(peaks) > 0]
    if max_peak_freq:
        x_limit = 3 * max(max_peak_freq)  # 3 times the maximum peak frequency
        if x_limit > max(freq[-1] for freq in freqs):
            x_limit = max(freq[-1] for freq in freqs)
    else:  # if no peaks were found
        x_limit = max(freq[-1] for freq in freqs)

    # plot frequency spectrum
    fig = go.Figure()
    for spectrum_idx in range(len(freqs)):
        label = f"n={labels[spectrum_idx]}"
        fig.add_trace(
            go.Scatter(
                x=freqs[spectrum_idx],
                y=np.abs(amps[spectrum_idx]),
                mode="lines",
                name=label,
            )
        )
    fig.update_layout(
        title="Fourier transformation",
        xaxis_title="Frequency (GHz)",
        yaxis_title="Amplitude",
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig.update_xaxes(range=[0, x_limit])
    plotting.show_figure(fig, preferred_width=800, show=show)

    return_dict = {
        "fig": fig,
        "frequencies": freqs,
        "amplitudes": amps,
        "frequency_peaks": freq_peaks,
        "spin_indices": labels,
    }

    return return_dict


# endregion Visualization

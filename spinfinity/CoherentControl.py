"""
Calculating and visualizing pulsed ESR measurements.

This module provides functions for calculating and visualizing pulsed electron spin
resonance (ESR) measurements, including Rabi oscillations, Ramsey fringes, Hahn echo,
relaxometry, and electron spin echo envelope modulation (ESEEM). The calculations are
performed in the rotating frame using the QuTiP library for quantum dynamics
simulations.

Functions
---------
RotFrameSecular
    Rotating-frame Hamiltonian with the secular (RWA) approximation.
calc_RabiOscillation
    Calculate Rabi oscillations as a function of microwave pulse width.
calc_RabiDetuningMap
    Calculate a parallelized Rabi map over pulse widths and microwave frequencies.
calc_Ramsey_Parallel
    Calculate Ramsey signals for multiple free-evolution times in parallel.
calc_RamseyDetuningMap
    Calculate a Ramsey signal map over free-evolution times and frequencies.
plot_RamseyShot
    Simulate and plot the time evolution of a single Ramsey sequence.
calc_HahnEcho_Parallel
    Calculate Hahn-echo signals for multiple delay times in parallel.
calc_HahnEcho_pluspihalf_Parallel
    Calculate parallel Hahn-echo signals with a final pi/2 readout pulse.
plot_HahnEchoShot
    Simulate and plot the time evolution of a single Hahn-echo sequence.
plot_HahnEchoShot_pluspihalf
    Plot a single Hahn-echo sequence with a final pi/2 readout pulse.
calc_Relaxometry
    Calculate spin relaxation following an initial pi pulse.
calc_3pulseESEEM_Parallel
    Calculate three-pulse ESEEM signals for multiple delays in parallel.
calc_3pulseESEEM_pluspihalf_Parallel
    Calculate parallel three-pulse ESEEM signals with a final pi/2 pulse.
calc_3pulseESEEM_pluspihalf_PhaseCycled
    Calculate phase-cycled three-pulse ESEEM signals with a final pi/2 pulse.
calc_3pulseESEEM_pluspihalf_PhaseCycled8
    Calculate rephasing three-pulse ESEEM with an eight-step phase cycle.
plot_3pulseESEEMShot_pluspihalf
    Simulate and plot one three-pulse ESEEM sequence with a final pi/2 pulse.

"""

import numpy as np
import plotly.graph_objects as go
from joblib import Parallel, delayed
from qutip import (
    Bloch,
    Qobj,
    expect,
    liouvillian,
    mesolve,
    operator_to_vector,
    vector_to_operator,
)

from spinfinity import plotting
from spinfinity.SpinSys import SpinSys


def RotFrameSecular(spinsys, f_RF: float):
    """
    Rotating-frame Hamiltonian with the secular (RWA) approximation.

    Steps:
      a) quantization axis of the readout spin  n = B/|B|
         (add BTip here if it acts on the readout spin);
      b) Sz' = n . S_readout;  the frame rotates about n at f_RF;
      c) secular projection: keep only the part of H that commutes with the
         frame rotation, H_sec = sum_m P_m H P_m  (P_m projector on the
         Sz' eigenvalue m of the readout spin);
      d) H_RF = H_sec + 2*pi*f_RF*Sz'  (spinfinity Zeeman sign convention:
         H = -g*muB*B.S, so +f_RF*Sz' cancels the Zeeman term on resonance).

    Parameters
    ----------
    spinsys : SpinSys
        SpinSys object containing the spin system information.
    f_RF : float
        Frequency of the rotating frame in GHz.

    Returns
    -------
    H_RF : Qobj
        Rotating-frame Hamiltonian in angular GHz (rad/ns).

    Notes
    -----
    Also stores the tilted pulse/readout operators on spinsys:
    spinsys.Sz_rot, spinsys.Sx_rot, spinsys.Sy_rot
    Use Sx_rot as the pulse operator and Sy_rot/Sz_rot for detection.
    f_RF in GHz, H_RF returned in angular GHz (rad/ns), like RotFrame.
    """
    i = spinsys.ReadoutSpin

    # First handle the type of the tip
    if spinsys.tip == "a":
        Bvec = np.array(spinsys.B) - np.array(spinsys.BTip)
    else:
        Bvec = np.array(spinsys.B) + np.array(spinsys.BTip)

    n = Bvec / np.linalg.norm(Bvec)

    Sz_r = n[0] * spinsys.Sx[i] + n[1] * spinsys.Sy[i] + n[2] * spinsys.Sz[i]

    # transverse axes of the tilted frame (x' in the plane of lab-x and n)
    ref = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(ref, n)) > 0.99:
        ref = np.array([0.0, 1.0, 0.0])
    x_p = ref - np.dot(ref, n) * n
    x_p /= np.linalg.norm(x_p)
    y_p = np.cross(n, x_p)
    Sx_r = x_p[0] * spinsys.Sx[i] + x_p[1] * spinsys.Sy[i] + x_p[2] * spinsys.Sz[i]
    Sy_r = y_p[0] * spinsys.Sx[i] + y_p[1] * spinsys.Sy[i] + y_p[2] * spinsys.Sz[i]

    # secular projection of the full static Hamiltonian
    H = spinsys.H * spinsys.meVtoGHzConversion * 2 * np.pi
    evals, evecs = np.linalg.eigh(Sz_r)
    H_sec = np.zeros_like(H)
    for m in np.unique(np.round(evals, 6)):
        V = evecs[:, np.abs(evals - m) < 1e-6]
        P = V @ V.conj().T
        H_sec = H_sec + P @ H @ P

    H_RF = H_sec + f_RF * Sz_r * 2 * np.pi

    spinsys.Sz_rot, spinsys.Sx_rot, spinsys.Sy_rot = Sz_r, Sx_r, Sy_r
    spinsys.H_RF = Qobj(H_RF)

    return spinsys.H_RF


# region Rabi


def calc_RabiOscillation(
    spinsys: SpinSys,
    pulsewidth_array: np.ndarray,
    e_ops: list[Qobj] = None,
    solver_options: dict = None,
):
    """
    Calculate Rabi oscillations for a given spin system.

    This function computes the Rabi oscillation of the spinsys.
    The calculation requires ``spinsys`` to contain ``f_RF``, ``collapse_ops``,
    ``RabiRate``, and ``psi0``, as well as a completed Hamiltonian and the
    corresponding spin operators.

    Parameters
    ----------
    spinsys : SpinSys
        SpinSys object containing the spin system information.
    pulsewidth_array : np.ndarray
        Array of pulse widths (in ns) for which to calculate the Rabi oscillation.
    e_ops : list[Qobj], optional
        List of expectation operators to calculate. If None, defaults to the spin
        operators Sx, Sy, and Sz of the readout spin.
    solver_options : dict, optional
        Dictionary of options to pass to the QuTiP mesolve solver. If None, defaults
        to an empty dictionary.

    Returns
    -------
    pulsewidth_array : np.ndarray
        Array of pulse widths (in ns) used for the calculation.
    result : qutip.solver.Result
        Result object containing the expectation values for the specified operators
        at each pulse width.
    """
    if e_ops is None:
        e_ops = [
            Qobj(spinsys.Sx[spinsys.ReadoutSpin]),
            Qobj(spinsys.Sy[spinsys.ReadoutSpin]),
            Qobj(spinsys.Sz[spinsys.ReadoutSpin]),
        ]

    # Add a zero pulse width to the beginning of the array if not already present
    if pulsewidth_array[0] > 0:
        pulsewidth_array = np.concatenate(([0.0], pulsewidth_array))

    # Calculate the Hamiltonian in the rotating frame of the microwave field
    H_rot = RotFrameSecular(spinsys, spinsys.f_RF) + Qobj(
        2 * np.pi * spinsys.RabiRate * spinsys.Sx_rot
    )

    # Option handling
    if solver_options is None:
        solver_options = {}

    result = mesolve(
        H_rot,
        spinsys.psi0,
        pulsewidth_array,
        c_ops=spinsys.collapse_ops,
        e_ops=e_ops,
        options=solver_options,
    )

    return pulsewidth_array, result


def calc_RabiDetuningMap(
    spinsys: SpinSys,
    pulsewidth_array: np.ndarray,
    frequency_array: np.ndarray,
    e_ops: Qobj = None,
    solver_options: dict = None,
    n_jobs: int = -1,
):
    """
    Calculate a Rabi-oscillation map as a function of frequency detuning.

    This function computes the expectation value of a single operator for every
    combination of microwave frequency and pulse width. The calculation requires
    ``spinsys`` to contain ``collapse_ops``, ``RabiRate``, and ``psi0``, as well as
    a completed Hamiltonian and the corresponding spin operators.

    Parameters
    ----------
    spinsys : SpinSys
        SpinSys object containing the spin system information.
    pulsewidth_array : np.ndarray
        Array of pulse widths (in ns) for which to calculate the Rabi oscillation.
    frequency_array : np.ndarray
        Array of microwave frequencies (in GHz) for which to calculate the Rabi
        oscillation.
    e_ops : Qobj, optional
        Expectation operator to calculate. If None, defaults to the Sz operator of
        the readout spin.
    solver_options : dict, optional
        Dictionary of options to pass to the QuTiP mesolve solver. If None, defaults
        to an empty dictionary.
    n_jobs : int, optional
        Number of parallel worker processes used to evaluate the frequencies. ``-1``
        uses all available CPU cores, while ``1`` runs the calculation serially.

    Returns
    -------
    pulsewidth_array : np.ndarray
        Array of pulse widths (in ns) used for the calculation.
    frequency_array : np.ndarray
        Array of microwave frequencies (in GHz) used for the calculation.
    signal_map : np.ndarray
        Two-dimensional array of expectation values with shape
        ``(len(frequency_array), len(pulsewidth_array))``.

    Notes
    -----
    Each microwave frequency is evaluated independently. Parallel calculations use
    Joblib's ``loky`` process backend, and the returned rows retain the order of
    ``frequency_array``.
    """
    if pulsewidth_array[0] > 0:
        pulsewidth_array = np.concatenate(([0.0], pulsewidth_array))
    pulsewidth_array = np.asarray(pulsewidth_array, dtype=float)
    frequency_array = np.asarray(frequency_array, dtype=float)

    if e_ops is None:
        e_ops = Qobj(spinsys.Sz[spinsys.ReadoutSpin])

    if solver_options is None:
        solver_options = {}

    def calc_frequency(frequency):
        # Calculate the Hamiltonian in the rotating frame of the microwave field
        H_rot = RotFrameSecular(spinsys, frequency) + Qobj(
            2 * np.pi * spinsys.RabiRate * spinsys.Sx_rot
        )

        result = mesolve(
            H_rot,
            spinsys.psi0,
            pulsewidth_array,
            c_ops=spinsys.collapse_ops,
            e_ops=[e_ops],
            options=solver_options,
        )

        return np.asarray(result.expect[0])

    if n_jobs == 1 or frequency_array.size <= 1:
        signal_rows = [calc_frequency(frequency) for frequency in frequency_array]
    else:
        signal_rows = Parallel(
            n_jobs=n_jobs,
            backend="loky",
            batch_size="auto",
        )(delayed(calc_frequency)(frequency) for frequency in frequency_array)

    if signal_rows:
        signal_map = np.asarray(signal_rows)
    else:
        dtype = float if e_ops.isherm else complex
        signal_map = np.empty((0, pulsewidth_array.size), dtype=dtype)

    return pulsewidth_array, frequency_array, signal_map


# endregion Rabi

# region Ramsey


def calc_Ramsey_Parallel(
    spinsys: SpinSys,
    tau_array: np.ndarray,
    e_ops: list[Qobj] | None = None,
    options: dict | None = None,
    n_jobs: int = -1,
):
    """
    Calculate a Ramsey sequence for multiple free-evolution times.

    This function computes the final expectation values after a Ramsey sequence
    consisting of a pi/2 pulse, free evolution for ``tau``, and a second pi/2 pulse.
    The calculation requires ``spinsys`` to contain ``f_RF``, ``RabiRate``, ``psi0``,
    and ``collapse_ops``, as well as a completed Hamiltonian and the corresponding
    spin operators.

    Parameters
    ----------
    spinsys : SpinSys
        SpinSys object containing the spin system information.
    tau_array : np.ndarray
        Array of non-negative free-evolution times (in ns) for which to calculate
        the Ramsey signal.
    e_ops : list[Qobj], optional
        List of expectation operators to calculate. If None, defaults to the Sx,
        Sy, and Sz operators of the readout spin.
    options : dict, optional
        Solver options merged with the defaults. store_final_state=True is enforced
        because consecutive pulse-sequence segments require the preceding final state.
        The supplied dictionary is not modified.
    n_jobs : int, optional
        Number of parallel workers used to evaluate the free-evolution times. ``-1``
        uses all available CPU cores, while ``1`` runs the calculation serially.

    Returns
    -------
    tau_array : np.ndarray
        Array of free-evolution times (in ns) used for the calculation.
    results : np.ndarray
        Two-dimensional array of final expectation values with shape
        ``(len(tau_array), len(e_ops))``.

    Raises
    ------
    ValueError
        If any value in ``tau_array`` is negative.

    Notes
    -----
    The first pi/2 pulse is calculated once because it is identical for every
    free-evolution time. The remaining sequence is evaluated independently for each
    value in ``tau_array`` using Joblib.
    """
    tau_array = np.asarray(tau_array, dtype=float)

    if np.any(tau_array < 0):
        raise ValueError("All tau values must be non-negative.")

    if e_ops is None:
        readout = spinsys.ReadoutSpin
        e_ops = [
            Qobj(spinsys.Sx[readout]),
            Qobj(spinsys.Sy[readout]),
            Qobj(spinsys.Sz[readout]),
        ]

    H_free = RotFrameSecular(spinsys, spinsys.f_RF)

    Omega_R = 2.0 * np.pi * spinsys.RabiRate
    tp_90 = np.pi / (2.0 * Omega_R)

    pulse_op = Qobj(spinsys.Sx_rot)
    H_pulse = H_free + Omega_R * pulse_op

    psi0 = spinsys.psi0
    c_ops = spinsys.collapse_ops

    # Only retain the final state from each segment.
    options = {
        "store_states": False,
        "progress_bar": "",
        "method": "diag",
        **({} if options is None else options),
        "store_final_state": True,
    }

    def evolve_final(H, state, duration):
        duration = float(duration)

        if duration == 0.0:
            return state

        result = mesolve(
            H,
            state,
            [0.0, duration],
            c_ops=c_ops,
            e_ops=None,
            options=options,
        )
        return result.final_state

    # This segment is identical for every tau, so calculate it once.
    state_after_first_90 = evolve_final(H_pulse, psi0, tp_90)

    def calc_point(tau):
        state = state_after_first_90

        # Free evolution for tau.
        state = evolve_final(H_free, state, tau)

        # Final pi/2 pulse.
        state = evolve_final(H_pulse, state, tp_90)

        return np.asarray(expect(e_ops, state))

    results = Parallel(n_jobs=n_jobs)(delayed(calc_point)(tau) for tau in tau_array)

    return tau_array, np.asarray(results)


def calc_RamseyDetuningMap(
    spinsys: SpinSys,
    tau_array: np.ndarray,
    frequency_array: np.ndarray,
    e_ops: Qobj | None = None,
    n_jobs: int = -1,
):
    """
    Calculate a Ramsey signal map as a function of microwave frequency.

    This function evaluates a pi/2--tau--pi/2 Ramsey sequence for every
    combination of microwave frequency and free-evolution time. The calculation
    requires ``spinsys`` to contain ``RabiRate``, ``psi0``, and ``collapse_ops``,
    as well as a completed Hamiltonian and the corresponding spin operators.

    Parameters
    ----------
    spinsys : SpinSys
        SpinSys object containing the spin system information.
    tau_array : np.ndarray
        Array of non-negative free-evolution times (in ns).
    frequency_array : np.ndarray
        Array of microwave frequencies (in GHz).
    e_ops : Qobj, optional
        Expectation operator to calculate. If None, defaults to the Sz operator of
        the readout spin.
    n_jobs : int, optional
        Number of parallel workers used to evaluate the free-evolution times at each
        frequency. ``-1`` uses all available CPU cores, while ``1`` runs serially.

    Returns
    -------
    tau_array : np.ndarray
        Array of free-evolution times (in ns) used for the calculation.
    frequency_array : np.ndarray
        Array of microwave frequencies (in GHz) used for the calculation.
    signal_map : np.ndarray
        Two-dimensional array of expectation values with shape
        ``(len(frequency_array), len(tau_array))``.

    Raises
    ------
    ValueError
        If any value in ``tau_array`` is negative.

    Notes
    -----
    Frequencies are processed sequentially. For each frequency, the first pi/2 pulse
    is calculated once and the remaining sequence is evaluated in parallel for the
    values in ``tau_array``.
    """
    tau_array = np.asarray(tau_array, dtype=float)
    frequency_array = np.asarray(frequency_array, dtype=float)

    if np.any(tau_array < 0):
        raise ValueError("All tau values must be non-negative.")

    if e_ops is None:
        e_ops = Qobj(spinsys.Sz[spinsys.ReadoutSpin])

    Omega_R = 2.0 * np.pi * spinsys.RabiRate
    tp_90 = np.pi / (2.0 * Omega_R)
    psi0 = spinsys.psi0
    c_ops = spinsys.collapse_ops

    options = {
        "store_states": False,
        "store_final_state": True,
        "progress_bar": "",
        "method": "diag",
    }

    def evolve_final(H, state, duration):
        duration = float(duration)

        if duration == 0.0:
            return state

        result = mesolve(
            H,
            state,
            [0.0, duration],
            c_ops=c_ops,
            e_ops=None,
            options=options,
        )
        return result.final_state

    signal_rows = []
    for frequency in frequency_array:
        H_free = RotFrameSecular(spinsys, frequency)
        pulse_op = Qobj(spinsys.Sx_rot)
        H_pulse = H_free + Omega_R * pulse_op

        # This first pulse is shared by every delay at the current frequency.
        state_after_first_90 = evolve_final(H_pulse, psi0, tp_90)

        def calc_point(
            tau,
            H_free=H_free,
            state_after_first_90=state_after_first_90,
            H_pulse=H_pulse,
        ):
            state = evolve_final(H_free, state_after_first_90, tau)
            state = evolve_final(H_pulse, state, tp_90)
            return expect(e_ops, state)

        signal_rows.append(
            Parallel(n_jobs=n_jobs)(delayed(calc_point)(tau) for tau in tau_array)
        )

    if signal_rows:
        signal_map = np.asarray(signal_rows)
    else:
        dtype = float if e_ops.isherm else complex
        signal_map = np.empty((0, len(tau_array)), dtype=dtype)

    return tau_array, frequency_array, signal_map


def plot_RamseyShot(spinsys, tau: float, tsteps: int = 500, show: str | None = None):
    """
    Calculate and plot the time evolution of a single Ramsey sequence.

    This function simulates a pi/2--tau--pi/2 Ramsey sequence, displays the Sx, Sy,
    and Sz expectation values in a Plotly figure, and shows the corresponding QuTiP
    Bloch-sphere trajectory.

    Parameters
    ----------
    spinsys : SpinSys
        SpinSys object containing the spin system, initial state, collapse operators,
        Rabi rate, and microwave frequency.
    tau : float
        Free-evolution time between the two pi/2 pulses (in ns).
    tsteps : int, optional
        Number of time points used to simulate and plot the sequence.

    show : {"fixed", "responsive", "no"} or None, optional
        Display mode. ``None`` uses the global setting, initially ``"fixed"``.
        ``"fixed"`` uses the preferred width; ``"responsive"`` fills the
        container width. ``"no"`` suppresses both figures without skipping calculations.

    Returns
    -------
    return_dict : dict
        Dictionary containing ``fig`` (the Plotly figure), ``result`` (the QuTiP
        solver result), ``time_values`` (the time grid in ns), and
        ``expectation_values`` (the Sx, Sy, and Sz expectation-value arrays).

    Notes
    -----
    The Plotly figure and QuTiP Bloch sphere are displayed as side effects
    unless the selected display mode is ``"no"``.
    """
    # fixed e_ops:
    e_ops = [
        Qobj(spinsys.Sx[spinsys.ReadoutSpin]),
        Qobj(spinsys.Sy[spinsys.ReadoutSpin]),
        Qobj(spinsys.Sz[spinsys.ReadoutSpin]),
    ]

    # Calculate the Hamiltonian in the rotating frame of the microwave field
    H_rot = RotFrameSecular(spinsys, spinsys.f_RF)

    # Build Pulse Sequence for Ramsey experiment: pi/2 - tau - pi/2
    Omega_R = 2 * np.pi * spinsys.RabiRate
    tp_90 = (np.pi / 2) / Omega_R
    pulse_op = Qobj(spinsys.Sx_rot)
    solver_options = {"max_step": tp_90 / 10.0}
    t1_start = 0.0
    t1_end = tp_90

    t2_start = t1_end + tau
    t2_end = t2_start + tp_90

    t_final = t2_end
    tlist = np.linspace(0, t_final, tsteps)

    def Ox_coeff(t, Omega_R):
        if t1_start <= t <= t1_end:
            return Omega_R
        if t2_start <= t <= t2_end:
            return Omega_R
        return 0.0

    # Build the time-dependent Hamiltonian for the Ramsey sequence
    H = [
        H_rot,
        [pulse_op, Ox_coeff],
    ]

    # Solve the master equation for the Ramsey sequence
    args = {"Omega_R": Omega_R}
    result = mesolve(
        H,
        spinsys.psi0,
        tlist,
        c_ops=spinsys.collapse_ops,
        e_ops=e_ops,
        args=args,
        options=solver_options,
    )

    # --------------------------------------------------
    # Plotting

    operator_names = [
        "⟨<i>S</i><sub>x</sub>⟩",
        "⟨<i>S</i><sub>y</sub>⟩",
        "⟨<i>S</i><sub>z</sub>⟩",
    ]
    fig = go.Figure()
    for i, _e_op in enumerate(e_ops):
        fig.add_trace(
            go.Scatter(
                x=tlist,
                y=result.expect[i],
                mode="lines",
                name=operator_names[i],
            )
        )
    fig.update_layout(
        xaxis_title="time (ns)",
        yaxis_title="Expectation Value",
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig.update_xaxes(range=[tlist[0], tlist[-1]])
    plotting.show_figure(fig, preferred_width=800, show=show)

    sphere = Bloch()
    sphere.add_points(
        [
            2 * result.expect[0],
            2 * result.expect[1],
            2 * result.expect[2],
        ],
        meth="l",
    )
    # Label the axis according to the expectation values of the spin operators
    sphere.xlabel = [r"$\langle S_x \rangle$", ""]
    sphere.ylabel = [r"$\langle S_y \rangle$", ""]
    sphere.zlabel = [r"$\langle S_z \rangle$", ""]

    if plotting._resolve_display_mode(show) != "no":
        sphere.show()

    return_dict = {
        "fig": fig,
        "result": result,
        "time_values": tlist,
        "expectation_values": result.expect,
    }

    return return_dict


# endregion Ramsey

# region Hahn Echo


def calc_HahnEcho_Parallel(
    spinsys: SpinSys,
    tau_array: np.ndarray,
    e_ops: list[Qobj] | None = None,
    n_jobs: int = -1,
):
    """
    Calculate a Hahn-echo sequence in parallel for multiple delay times.

    This function computes the final expectation values after a
    pi/2--tau--pi--tau pulse sequence. The calculation requires ``spinsys`` to
    contain ``f_RF``, ``RabiRate``, ``psi0``, and ``collapse_ops``, as well as a
    completed Hamiltonian and the corresponding spin operators.

    Parameters
    ----------
    spinsys : SpinSys
        SpinSys object containing the spin system information.
    tau_array : np.ndarray
        Array of non-negative free-evolution times (in ns) on either side of the pi
        pulse.
    e_ops : list[Qobj], optional
        List of expectation operators to calculate. If None, defaults to the Sx,
        Sy, and Sz operators of the readout spin.
    n_jobs : int, optional
        Number of parallel workers used to evaluate the delay times. ``-1`` uses all
        available CPU cores, while ``1`` runs the calculation serially.

    Returns
    -------
    tau_array : np.ndarray
        Array of delay times (in ns) used for the calculation.
    results : np.ndarray
        Two-dimensional array of final expectation values with shape
        ``(len(tau_array), len(e_ops))``.

    Raises
    ------
    ValueError
        If any value in ``tau_array`` is negative.

    Notes
    -----
    The first pi/2 pulse is calculated once. The remainder of the sequence is then
    evaluated independently for each value in ``tau_array`` using Joblib.
    """
    tau_array = np.asarray(tau_array, dtype=float)

    if np.any(tau_array < 0):
        raise ValueError("All tau values must be non-negative.")

    if e_ops is None:
        readout = spinsys.ReadoutSpin
        e_ops = [
            Qobj(spinsys.Sx[readout]),
            Qobj(spinsys.Sy[readout]),
            Qobj(spinsys.Sz[readout]),
        ]

    H_free = RotFrameSecular(spinsys, spinsys.f_RF)

    Omega_R = 2.0 * np.pi * spinsys.RabiRate
    tp_90 = np.pi / (2.0 * Omega_R)
    tp_180 = np.pi / Omega_R

    pulse_op = Qobj(spinsys.Sx_rot)
    H_pulse = H_free + Omega_R * pulse_op

    psi0 = spinsys.psi0
    c_ops = spinsys.collapse_ops

    # Only retain the final state from each segment.
    options = {
        "store_states": False,
        "store_final_state": True,
        "progress_bar": "",
        "method": "diag",
    }

    def evolve_final(H, state, duration):
        duration = float(duration)

        if duration == 0.0:
            return state

        result = mesolve(
            H,
            state,
            [0.0, duration],
            c_ops=c_ops,
            e_ops=None,
            options=options,
        )
        return result.final_state

    # This segment is identical for every tau, so calculate it once.
    state_after_first_90 = evolve_final(H_pulse, psi0, tp_90)

    def calc_point(tau):
        state = state_after_first_90

        # Free evolution for tau.
        state = evolve_final(H_free, state, tau)

        # Pi pulse.
        state = evolve_final(H_pulse, state, tp_180)

        # Second free evolution.
        state = evolve_final(H_free, state, tau)

        return np.asarray(expect(e_ops, state))

    results = Parallel(n_jobs=n_jobs)(delayed(calc_point)(tau) for tau in tau_array)

    return tau_array, np.asarray(results)


def calc_HahnEcho_pluspihalf_Parallel(
    spinsys: SpinSys,
    tau_array: np.ndarray,
    e_ops: list[Qobj] | None = None,
    n_jobs: int = -1,
):
    """
    Calculate a Hahn-echo sequence with a final pi/2 pulse in parallel.

    This function computes the final expectation values after a
    pi/2--tau--pi--tau--pi/2 pulse sequence. The calculation requires ``spinsys``
    to contain ``f_RF``, ``RabiRate``, ``psi0``, and ``collapse_ops``, as well as a
    completed Hamiltonian and the corresponding spin operators.

    Parameters
    ----------
    spinsys : SpinSys
        SpinSys object containing the spin system information.
    tau_array : np.ndarray
        Array of non-negative free-evolution times (in ns) on either side of the pi
        pulse.
    e_ops : list[Qobj], optional
        List of expectation operators to calculate. If None, defaults to the Sx,
        Sy, and Sz operators of the readout spin.
    n_jobs : int, optional
        Number of parallel workers used to evaluate the delay times. ``-1`` uses all
        available CPU cores, while ``1`` runs the calculation serially.

    Returns
    -------
    tau_array : np.ndarray
        Array of delay times (in ns) used for the calculation.
    results : np.ndarray
        Two-dimensional array of final expectation values with shape
        ``(len(tau_array), len(e_ops))``.

    Raises
    ------
    ValueError
        If any value in ``tau_array`` is negative.

    Notes
    -----
    The first pi/2 pulse is calculated once. The remainder of the sequence is then
    evaluated independently for each value in ``tau_array`` using Joblib.
    """
    tau_array = np.asarray(tau_array, dtype=float)

    if np.any(tau_array < 0):
        raise ValueError("All tau values must be non-negative.")

    if e_ops is None:
        readout = spinsys.ReadoutSpin
        e_ops = [
            Qobj(spinsys.Sx[readout]),
            Qobj(spinsys.Sy[readout]),
            Qobj(spinsys.Sz[readout]),
        ]

    H_free = RotFrameSecular(spinsys, spinsys.f_RF)

    Omega_R = 2.0 * np.pi * spinsys.RabiRate
    tp_90 = np.pi / (2.0 * Omega_R)
    tp_180 = np.pi / Omega_R

    pulse_op = Qobj(spinsys.Sx_rot)
    H_pulse = H_free + Omega_R * pulse_op

    psi0 = spinsys.psi0
    c_ops = spinsys.collapse_ops

    # Only retain the final state from each segment.
    options = {
        "store_states": False,
        "store_final_state": True,
        "progress_bar": "",
        "method": "diag",
    }

    def evolve_final(H, state, duration):
        duration = float(duration)

        if duration == 0.0:
            return state

        result = mesolve(
            H,
            state,
            [0.0, duration],
            c_ops=c_ops,
            e_ops=None,
            options=options,
        )
        return result.final_state

    # This segment is identical for every tau, so calculate it once.
    state_after_first_90 = evolve_final(H_pulse, psi0, tp_90)

    def calc_point(tau):
        state = state_after_first_90

        # Free evolution for tau.
        state = evolve_final(H_free, state, tau)

        # Pi pulse.
        state = evolve_final(H_pulse, state, tp_180)

        # Second free evolution.
        state = evolve_final(H_free, state, tau)

        # Final pi/2 pulse.
        state = evolve_final(H_pulse, state, tp_90)

        return np.asarray(expect(e_ops, state))

    results = Parallel(n_jobs=n_jobs)(delayed(calc_point)(tau) for tau in tau_array)

    return tau_array, np.asarray(results)


def plot_HahnEchoShot(
    spinsys: SpinSys, tau: float, tsteps: int = 2500, show: str | None = None
):
    """
    Calculate and plot the time evolution of a single Hahn-echo sequence.

    This function simulates a pi/2--tau--pi--tau sequence, displays the Sx, Sy, and
    Sz expectation values in a Plotly figure, and shows the corresponding QuTiP
    Bloch-sphere trajectory.

    Parameters
    ----------
    spinsys : SpinSys
        SpinSys object containing the spin system, initial state, collapse operators,
        Rabi rate, and microwave frequency.
    tau : float
        Free-evolution time on either side of the pi pulse (in ns).
    tsteps : int, optional
        Number of time points used to simulate and plot the sequence.

    show : {"fixed", "responsive", "no"} or None, optional
        Display mode. ``None`` uses the global setting, initially ``"fixed"``.
        ``"fixed"`` uses the preferred width; ``"responsive"`` fills the
        container width. ``"no"`` suppresses both figures without skipping calculations.

    Returns
    -------
    return_dict : dict
        Dictionary containing ``fig`` (the Plotly figure), ``result`` (the QuTiP
        solver result), ``time_values`` (the time grid in ns), and
        ``expectation_values`` (the Sx, Sy, and Sz expectation-value arrays).

    Notes
    -----
    The Plotly figure and QuTiP Bloch sphere are displayed as side effects
    unless the selected display mode is ``"no"``.
    """
    # fixed e_ops:
    e_ops = [
        Qobj(spinsys.Sx[spinsys.ReadoutSpin]),
        Qobj(spinsys.Sy[spinsys.ReadoutSpin]),
        Qobj(spinsys.Sz[spinsys.ReadoutSpin]),
    ]

    # Calculate the Hamiltonian in the rotating frame of the microwave field
    H_rot = RotFrameSecular(spinsys, spinsys.f_RF)

    # Build Pulse Sequence for Hahn Echo: pi/2 - tau - pi - tau
    Omega_R = 2 * np.pi * spinsys.RabiRate
    tp_90 = (np.pi / 2) / Omega_R
    tp_180 = np.pi / Omega_R
    pulse_op = Qobj(spinsys.Sx_rot)
    solver_options = {"max_step": tp_90 / 10.0}
    t1_start = 0.0
    t1_end = tp_90

    t2_start = t1_end + tau
    t2_end = t2_start + tp_180
    t_final = t2_end + tau

    tlist = np.linspace(0, t_final, tsteps)

    def Ox_coeff(t, Omega_R):
        if t1_start <= t <= t1_end or t2_start <= t <= t2_end:
            return Omega_R
        return 0.0

    # Build the time-dependent Hamiltonian for the Hahn Echo sequence
    H = [
        H_rot,
        [pulse_op, Ox_coeff],
    ]

    # Solve the master equation for the Hahn Echo sequence
    args = {"Omega_R": Omega_R}
    result = mesolve(
        H,
        spinsys.psi0,
        tlist,
        c_ops=spinsys.collapse_ops,
        e_ops=e_ops,
        args=args,
        options=solver_options,
    )

    # --------------------------------------------------
    # Plotting

    operator_names = [
        "⟨<i>S</i><sub>x</sub>⟩",
        "⟨<i>S</i><sub>y</sub>⟩",
        "⟨<i>S</i><sub>z</sub>⟩",
    ]
    fig = go.Figure()
    for i, _e_op in enumerate(e_ops):
        fig.add_trace(
            go.Scatter(
                x=tlist,
                y=result.expect[i],
                mode="lines",
                name=operator_names[i],
            )
        )
    fig.update_layout(
        xaxis_title="time (ns)",
        yaxis_title="Expectation Value",
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig.update_xaxes(range=[tlist[0], tlist[-1]])
    plotting.show_figure(fig, preferred_width=800, show=show)

    sphere = Bloch()
    sphere.add_points(
        [
            2 * result.expect[0],
            2 * result.expect[1],
            2 * result.expect[2],
        ],
        meth="l",
    )
    # Label the axis according to the expectation values of the spin operators
    sphere.xlabel = [r"$\langle S_x \rangle$", ""]
    sphere.ylabel = [r"$\langle S_y \rangle$", ""]
    sphere.zlabel = [r"$\langle S_z \rangle$", ""]

    if plotting._resolve_display_mode(show) != "no":
        sphere.show()

    return_dict = {
        "fig": fig,
        "result": result,
        "time_values": tlist,
        "expectation_values": result.expect,
    }

    return return_dict


def plot_HahnEchoShot_pluspihalf(
    spinsys: SpinSys, tau: float, tsteps: int = 2500, show: str | None = None
):
    """
    Plot a single Hahn-echo sequence followed by a final pi/2 pulse.

    This function simulates a pi/2--tau--pi--tau--pi/2 sequence, displays the Sx,
    Sy, and Sz expectation values in a Plotly figure, and shows the corresponding
    QuTiP Bloch-sphere trajectory.

    Parameters
    ----------
    spinsys : SpinSys
        SpinSys object containing the spin system, initial state, collapse operators,
        Rabi rate, and microwave frequency.
    tau : float
        Free-evolution time on either side of the pi pulse (in ns).
    tsteps : int, optional
        Number of time points used to simulate and plot the sequence.

    show : {"fixed", "responsive", "no"} or None, optional
        Display mode. ``None`` uses the global setting, initially ``"fixed"``.
        ``"fixed"`` uses the preferred width; ``"responsive"`` fills the
        container width. ``"no"`` suppresses both figures without skipping calculations.

    Returns
    -------
    return_dict : dict
        Dictionary containing ``fig`` (the Plotly figure), ``result`` (the QuTiP
        solver result), ``time_values`` (the time grid in ns), and
        ``expectation_values`` (the Sx, Sy, and Sz expectation-value arrays).

    Notes
    -----
    The Plotly figure and QuTiP Bloch sphere are displayed as side effects
    unless the selected display mode is ``"no"``.
    """
    # fixed e_ops:
    e_ops = [
        Qobj(spinsys.Sx[spinsys.ReadoutSpin]),
        Qobj(spinsys.Sy[spinsys.ReadoutSpin]),
        Qobj(spinsys.Sz[spinsys.ReadoutSpin]),
    ]

    # Calculate the Hamiltonian in the rotating frame of the microwave field
    H_rot = RotFrameSecular(spinsys, spinsys.f_RF)

    # Build Pulse Sequence for Hahn Echo: pi/2 - tau - pi - tau
    Omega_R = 2 * np.pi * spinsys.RabiRate
    tp_90 = (np.pi / 2) / Omega_R
    tp_180 = np.pi / Omega_R
    pulse_op = Qobj(spinsys.Sx_rot)
    solver_options = {"max_step": tp_90 / 10.0}
    t1_start = 0.0
    t1_end = tp_90

    t2_start = t1_end + tau
    t2_end = t2_start + tp_180
    t3_start = t2_end + tau
    t3_end = t3_start + tp_90

    t_final = t3_end
    tlist = np.linspace(0, t_final, tsteps)

    def Ox_coeff(t, Omega_R):
        if (
            t1_start <= t <= t1_end
            or t2_start <= t <= t2_end
            or t3_start <= t <= t3_end
        ):
            return Omega_R
        return 0.0

    # Build the time-dependent Hamiltonian for the Hahn Echo sequence
    H = [
        H_rot,
        [pulse_op, Ox_coeff],
    ]

    # Solve the master equation for the Hahn Echo sequence
    args = {"Omega_R": Omega_R}
    result = mesolve(
        H,
        spinsys.psi0,
        tlist,
        c_ops=spinsys.collapse_ops,
        e_ops=e_ops,
        args=args,
        options=solver_options,
    )

    # --------------------------------------------------
    # Plotting

    operator_names = [
        "⟨<i>S</i><sub>x</sub>⟩",
        "⟨<i>S</i><sub>y</sub>⟩",
        "⟨<i>S</i><sub>z</sub>⟩",
    ]
    fig = go.Figure()
    for i, _e_op in enumerate(e_ops):
        fig.add_trace(
            go.Scatter(
                x=tlist,
                y=result.expect[i],
                mode="lines",
                name=operator_names[i],
            )
        )
    fig.update_layout(
        xaxis_title="time (ns)",
        yaxis_title="Expectation Value",
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig.update_xaxes(range=[tlist[0], tlist[-1]])
    plotting.show_figure(fig, preferred_width=800, show=show)

    sphere = Bloch()
    sphere.add_points(
        [
            2 * result.expect[0],
            2 * result.expect[1],
            2 * result.expect[2],
        ],
        meth="l",
    )
    # Label the axis according to the expectation values of the spin operators
    sphere.xlabel = [r"$\langle S_x \rangle$", ""]
    sphere.ylabel = [r"$\langle S_y \rangle$", ""]
    sphere.zlabel = [r"$\langle S_z \rangle$", ""]

    if plotting._resolve_display_mode(show) != "no":
        sphere.show()

    return_dict = {
        "fig": fig,
        "result": result,
        "time_values": tlist,
        "expectation_values": result.expect,
    }

    return return_dict


# endregion Hahn Echo

# region Relaxometry


def calc_Relaxometry(
    spinsys: SpinSys, tau_array: np.ndarray, e_ops: list[Qobj] | None = None
):
    """
    Calculate spin relaxation after an initial pi pulse.

    This function applies a pi pulse to the initial state and then computes the
    expectation values during free evolution over ``tau_array``. The calculation
    requires ``spinsys`` to contain ``f_RF``, ``RabiRate``, ``psi0``, and
    ``collapse_ops``, as well as a completed Hamiltonian and the corresponding spin
    operators.

    Parameters
    ----------
    spinsys : SpinSys
        SpinSys object containing the spin system information.
    tau_array : np.ndarray
        Array of non-negative free-evolution times (in ns).
    e_ops : list[Qobj], optional
        List of expectation operators to calculate. If None, defaults to the Sx,
        Sy, and Sz operators of the readout spin.

    Returns
    -------
    tau_array : np.ndarray
        Array of free-evolution times (in ns) used for the calculation.
    result : qutip.solver.Result
        Result object containing the expectation values for the specified operators
        at each each evolution time.

    Raises
    ------
    ValueError
        If any value in ``tau_array`` is negative.

    Notes
    -----
    The pi-pulse evolution retains only its final state, which is used as the initial
    state for the subsequent free-evolution calculation.
    """
    # Add a zero pulse width to the beginning of the array if not already present
    if tau_array[0] > 0:
        tau_array = np.concatenate(([0.0], tau_array))
    tau_array = np.asarray(tau_array, dtype=float)

    if np.any(tau_array < 0):
        raise ValueError("All tau values must be non-negative.")

    if e_ops is None:
        readout = spinsys.ReadoutSpin
        e_ops = [
            Qobj(spinsys.Sx[readout]),
            Qobj(spinsys.Sy[readout]),
            Qobj(spinsys.Sz[readout]),
        ]

    H_free = RotFrameSecular(spinsys, spinsys.f_RF)

    Omega_R = 2.0 * np.pi * spinsys.RabiRate
    tp_180 = np.pi / Omega_R

    pulse_op = Qobj(spinsys.Sx_rot)
    H_pulse = H_free + Omega_R * pulse_op

    psi0 = spinsys.psi0
    c_ops = spinsys.collapse_ops

    options = {
        "store_states": False,
        "store_final_state": True,
        "progress_bar": "",
        "method": "diag",
    }

    def evolve_final(H, state, duration):
        duration = float(duration)

        if duration == 0.0:
            return state

        result = mesolve(
            H,
            state,
            [0.0, duration],
            c_ops=c_ops,
            e_ops=None,
            options=options,
        )
        return result.final_state

    state_after_pi = evolve_final(H_pulse, psi0, tp_180)

    # Free evolution ater initial pi pulse for each tau

    result = mesolve(
        H_free,
        state_after_pi,
        tau_array,
        c_ops=c_ops,
        e_ops=e_ops,
        options=options,
    )

    return tau_array, result


# endregion Relaxometry


# region ESEEM


def calc_3pulseESEEM_Parallel(
    spinsys: SpinSys,
    T_array: np.ndarray,
    tau: float,
    e_ops: list[Qobj] | None = None,
    n_jobs: int = -1,
):
    """
    Calculate a three-pulse ESEEM sequence in parallel.

    This function computes final expectation values after the sequence
    pi/2--tau--pi/2--T--pi/2--tau for each variable delay in ``T_array``. The
    calculation requires ``spinsys`` to contain ``f_RF``, ``RabiRate``, ``psi0``,
    and ``collapse_ops``, as well as a completed Hamiltonian and the corresponding
    spin operators.

    Parameters
    ----------
    spinsys : SpinSys
        SpinSys object containing the spin system information.
    T_array : np.ndarray
        Array of variable free-evolution times (in ns) between the second and third
        pi/2 pulses.
    tau : float
        Fixed free-evolution time (in ns) following the first and third pi/2 pulses.
    e_ops : list[Qobj], optional
        List of expectation operators to calculate. If None, defaults to the Sx,
        Sy, and Sz operators of the readout spin.
    n_jobs : int, optional
        Number of parallel worker processes used to evaluate the variable delays.
        ``-1`` uses all available CPU cores, while ``1`` runs serially.

    Returns
    -------
    T_array : np.ndarray
        Array of variable free-evolution times (in ns) used for the calculation.
    results : np.ndarray
        Two-dimensional array of final expectation values with shape
        ``(len(T_array), len(e_ops))``.

    Notes
    -----
    The fixed pi/2--tau--pi/2 prefix is calculated once. The remaining sequence is
    evaluated independently for each value in ``T_array`` using Joblib's ``loky``
    process backend.
    """
    T_array = np.asarray(T_array, dtype=float)

    if e_ops is None:
        readout_spin = spinsys.ReadoutSpin
        e_ops = [
            Qobj(spinsys.Sx[readout_spin]),
            Qobj(spinsys.Sy[readout_spin]),
            Qobj(spinsys.Sz[readout_spin]),
        ]
    else:
        e_ops = list(e_ops)

    # Constant free-evolution Hamiltonian.
    H_free = RotFrameSecular(spinsys, spinsys.f_RF)

    # Constant pulse Hamiltonian.
    Omega_R = 2.0 * np.pi * spinsys.RabiRate
    tp_90 = np.pi / (2.0 * Omega_R)

    pulse_op = Qobj(spinsys.Sx_rot)
    H_pulse = H_free + Omega_R * pulse_op

    psi0 = spinsys.psi0
    c_ops = spinsys.collapse_ops

    # Only retain the final state from each segment.
    options = {
        "store_states": False,
        "store_final_state": True,
        "progress_bar": "",
        "method": "diag",
    }

    def evolve_final(H, state, duration: float):
        duration = float(duration)

        if duration == 0.0:
            return state

        result = mesolve(
            H,
            state,
            [0.0, duration],
            c_ops=c_ops,
            e_ops=None,
            options=options,
        )

        return result.final_state

    # --------------------------------------------------------------
    # Fixed prefix:
    #
    #     pi/2 - tau - pi/2
    #
    # This is identical for every T and is therefore calculated once.
    # --------------------------------------------------------------
    state_after_prefix = evolve_final(
        H_pulse,
        psi0,
        tp_90,
    )

    state_after_prefix = evolve_final(
        H_free,
        state_after_prefix,
        tau,
    )

    state_after_prefix = evolve_final(
        H_pulse,
        state_after_prefix,
        tp_90,
    )

    def calc_point(T: float):
        state = state_after_prefix

        # Variable free-evolution interval.
        state = evolve_final(
            H_free,
            state,
            T,
        )

        # Third pi/2 pulse.
        state = evolve_final(
            H_pulse,
            state,
            tp_90,
        )

        # Fixed free-evolution interval.
        state = evolve_final(
            H_free,
            state,
            tau,
        )

        return np.asarray(expect(e_ops, state))

    if n_jobs == 1 or T_array.size <= 1:
        results = [calc_point(T) for T in T_array]
    else:
        results = Parallel(
            n_jobs=n_jobs,
            backend="loky",
            batch_size="auto",
        )(delayed(calc_point)(T) for T in T_array)

    return T_array, np.asarray(results)


def calc_3pulseESEEM_pluspihalf_Parallel(
    spinsys: SpinSys,
    T_array: np.ndarray,
    tau: float,
    e_ops: list[Qobj] | None = None,
    n_jobs: int = -1,
):
    """
    Calculate a three-pulse ESEEM sequence with a final pi/2 pulse in parallel.

    This function computes final expectation values after the sequence
    pi/2--tau--pi/2--T--pi/2--tau--pi/2 for each variable delay in ``tau_array``.
    The calculation requires ``spinsys`` to contain ``f_RF``, ``RabiRate``,
    ``psi0``, and ``collapse_ops``, as well as a completed Hamiltonian and the
    corresponding spin operators.

    Parameters
    ----------
    spinsys : SpinSys
        SpinSys object containing the spin system information.
    T_array : np.ndarray
        Array of variable free-evolution times (in ns) between the second and third
        pi/2 pulses.
    tau : float
        Fixed free-evolution time (in ns) following the first and third pi/2 pulses.
    e_ops : list[Qobj], optional
        List of expectation operators to calculate. If None, defaults to the Sx,
        Sy, and Sz operators of the readout spin.
    n_jobs : int, optional
        Number of parallel worker processes used to evaluate the variable delays.
        ``-1`` uses all available CPU cores, while ``1`` runs serially.

    Returns
    -------
    T_array : np.ndarray
        Array of variable free-evolution times (in ns) used for the calculation.
    results : np.ndarray
        Two-dimensional array of final expectation values with shape
        ``(len(T_array), len(e_ops))``.

    Notes
    -----
    The fixed pi/2--tau--pi/2 prefix is calculated once. The remaining sequence is
    evaluated independently for each value in ``T_array`` using Joblib's ``loky``
    process backend.
    """
    T_array = np.asarray(T_array, dtype=float)

    if e_ops is None:
        readout_spin = spinsys.ReadoutSpin
        e_ops = [
            Qobj(spinsys.Sx[readout_spin]),
            Qobj(spinsys.Sy[readout_spin]),
            Qobj(spinsys.Sz[readout_spin]),
        ]
    else:
        e_ops = list(e_ops)

    # Constant free-evolution Hamiltonian.
    H_free = RotFrameSecular(spinsys, spinsys.f_RF)

    # Constant pulse Hamiltonian.
    Omega_R = 2.0 * np.pi * spinsys.RabiRate
    tp_90 = np.pi / (2.0 * Omega_R)

    pulse_op = Qobj(spinsys.Sx_rot)
    H_pulse = H_free + Omega_R * pulse_op

    psi0 = spinsys.psi0
    c_ops = spinsys.collapse_ops

    # Only retain the final state from each segment.
    options = {
        "store_states": False,
        "store_final_state": True,
        "progress_bar": "",
        "method": "diag",
    }

    def evolve_final(H, state, duration: float):
        duration = float(duration)

        if duration == 0.0:
            return state

        result = mesolve(
            H,
            state,
            [0.0, duration],
            c_ops=c_ops,
            e_ops=None,
            options=options,
        )

        return result.final_state

    # --------------------------------------------------------------
    # Fixed prefix:
    #
    #     pi/2 - tau - pi/2
    #
    # This is identical for every T and is therefore calculated once.
    # --------------------------------------------------------------
    state_after_prefix = evolve_final(
        H_pulse,
        psi0,
        tp_90,
    )

    state_after_prefix = evolve_final(
        H_free,
        state_after_prefix,
        tau,
    )

    state_after_prefix = evolve_final(
        H_pulse,
        state_after_prefix,
        tp_90,
    )

    def calc_point(T: float):
        state = state_after_prefix

        # Variable free-evolution interval.
        state = evolve_final(
            H_free,
            state,
            T,
        )

        # Third pi/2 pulse.
        state = evolve_final(
            H_pulse,
            state,
            tp_90,
        )

        # Fixed free-evolution interval.
        state = evolve_final(
            H_free,
            state,
            tau,
        )

        # Final pi/2 pulse.
        state = evolve_final(
            H_pulse,
            state,
            tp_90,
        )

        return np.asarray(expect(e_ops, state))

    if n_jobs == 1 or T_array.size <= 1:
        results = [calc_point(T) for T in T_array]
    else:
        results = Parallel(
            n_jobs=n_jobs,
            backend="loky",
            batch_size="auto",
        )(delayed(calc_point)(T) for T in T_array)

    return T_array, np.asarray(results)


def calc_3pulseESEEM_pluspihalf_PhaseCycled(
    spinsys: SpinSys,
    T_array: np.ndarray,
    tau: float,
    e_ops: list | None = None,
    n_jobs: int = -1,
):
    """
    Calculate a phase-cycled three-pulse ESEEM sequence with a final pi/2 pulse.

    This function evaluates the sequence
    pi/2--tau--pi/2--T--pi/2--tau--pi/2 using a four-step phase cycle on the first
    two pulses. The third and final pi/2 pulses retain the +x phase. The calculation
    requires ``spinsys`` to contain ``f_RF``, ``RabiRate``, ``psi0``, and
    ``collapse_ops``, as well as a completed Hamiltonian and the corresponding spin
    operators.

    Parameters
    ----------
    spinsys : SpinSys
        SpinSys object containing the spin system information.
    T_array : np.ndarray
        Array of variable free-evolution times (in ns) between the second and third
        pi/2 pulses.
    tau : float
        Fixed free-evolution time (in ns) following the first and third pi/2 pulses.
    e_ops : list[Qobj], optional
        List of expectation operators to calculate. If
        None, defaults to the Sx, Sy, and Sz operators of the readout spin.
    n_jobs : int, optional
        Number of parallel worker processes used to evaluate the variable delays.
        ``-1`` uses all available CPU cores, while ``1`` runs serially.

    Returns
    -------
    T_array : np.ndarray
        Array of variable free-evolution times (in ns) used for the calculation.
    signal : np.ndarray
        Two-dimensional array of phase-cycled expectation values with shape
        ``(len(T_array), len(e_ops))``.

    Notes
    -----
    The four phase combinations are weighted as
    ``(+x, +x) - (-x, +x) - (+x, -x) + (-x, -x)`` and divided by four. Their fixed
    prefixes are calculated once, after which each value in ``T_array`` is
    evaluated independently using Joblib's ``loky`` process backend.
    """
    T_array = np.asarray(T_array, dtype=float)
    if e_ops is None:
        i = spinsys.ReadoutSpin
        e_ops = [Qobj(spinsys.Sx[i]), Qobj(spinsys.Sy[i]), Qobj(spinsys.Sz[i])]
    else:
        e_ops = list(e_ops)

    H_free = RotFrameSecular(spinsys, spinsys.f_RF)
    Omega_R = 2.0 * np.pi * spinsys.RabiRate
    tp_90 = np.pi / (2.0 * Omega_R)

    pulse_px = Qobj(spinsys.Sx_rot)  # +x pulse
    H_px = H_free + Omega_R * pulse_px
    H_mx = H_free - Omega_R * pulse_px  # -x pulse (180 deg phase)

    psi0 = spinsys.psi0
    c_ops = spinsys.collapse_ops
    options = {
        "store_states": False,
        "store_final_state": True,
        "progress_bar": "",
        "method": "diag",
    }

    def evolve_final(H, state, duration):
        if float(duration) == 0.0:
            return state
        res = mesolve(
            H, state, [0.0, float(duration)], c_ops=c_ops, e_ops=None, options=options
        )
        return res.final_state

    # four prefixes pi/2(ph1) - tau - pi/2(ph2), computed once each
    prefixes = {}
    for ph1, Hp1 in (("+x", H_px), ("-x", H_mx)):
        s = evolve_final(Hp1, psi0, tp_90)
        s = evolve_final(H_free, s, tau)
        for ph2, Hp2 in (("+x", H_px), ("-x", H_mx)):
            prefixes[(ph1, ph2)] = evolve_final(Hp2, s, tp_90)

    cycle = [
        (("+x", "+x"), +1.0),
        (("-x", "+x"), -1.0),
        (("+x", "-x"), -1.0),
        (("-x", "-x"), +1.0),
    ]

    def calc_point(T):
        acc = None
        for key, sign in cycle:
            s = evolve_final(H_free, prefixes[key], T)
            s = evolve_final(H_px, s, tp_90)  # 3rd pulse, fixed +x
            s = evolve_final(H_free, s, tau)
            s = evolve_final(H_px, s, tp_90)  # readout pi/2, fixed +x
            vals = np.asarray(expect(e_ops, s), dtype=float)
            acc = sign * vals if acc is None else acc + sign * vals
        return acc / 4.0

    if n_jobs == 1 or T_array.size <= 1:
        results = [calc_point(T) for T in T_array]
    else:
        results = Parallel(n_jobs=n_jobs, backend="loky", batch_size="auto")(
            delayed(calc_point)(T) for T in T_array
        )

    return T_array, np.asarray(results)


def calc_3pulseESEEM_pluspihalf_PhaseCycled8(
    spinsys: SpinSys,
    T_array: np.ndarray,
    tau: float,
    e_ops: list[Qobj] | None = None,
    n_jobs: int = -1,
):
    """
    Calculate three-pulse ESEEM with an eight-step rephasing phase cycle.

    The sequence is pi/2--tau--pi/2--T--pi/2--tau--pi/2. The first and
    final pulses have phase +x. The second and third pulse phases are cycled
    to select the rephasing stimulated-echo contribution.

    Parameters
    ----------
    spinsys : SpinSys
        Spin system with a spin-one-half readout spin, a completed static
        Hamiltonian, and configured f_RF, RabiRate, psi0, and collapse_ops.
        The initial state may be a ket or a density matrix.
    T_array : np.ndarray
        One-dimensional array of finite, non-negative storage delays T in ns.
        The supplied ordering is preserved, and no zero is inserted.
    tau : float
        Finite, non-negative free-evolution delay in ns before the second
        pulse and after the third pulse.
    e_ops : list[Qobj] or None, optional
        Observables measured after the final readout pulse. None uses
        Sz_rot of the readout spin, giving one signal column. Rephasing
        pathway selection assumes observables commuting with Sz_rot.
    n_jobs : int, optional
        Number of parallel workers. The default -1 uses all available CPU
        cores; 1 evaluates the delays serially.

    Returns
    -------
    T_array : np.ndarray
        Storage delays T in ns, in the supplied order.
    signal : np.ndarray
        Weighted mean of the eight expectation-value arrays, with shape
        (len(T_array), len(e_ops)), or (len(T_array), 1) for default readout.
        This is a signed quadrature signal, not a coherence magnitude.

    Raises
    ------
    ValueError
        If the readout spin is not one-half, T_array is not one-dimensional
        or contains negative or non-finite delays, tau is negative or
        non-finite, or RabiRate is non-positive or non-finite.

    Notes
    -----
    The cycle (phase_2, phase_3, weight), with phases in degrees, is
    (0, 0, +1), (180, 0, -1), (90, 90, -1), (270, 90, +1),
    (180, 180, +1), (0, 180, -1), (270, 270, -1), (90, 270, +1).
    Signals are divided by eight, without an additional amplitude rescaling.

    For ideal pulses, electron-secular free evolution, and dissipation
    covariant under rotations about Sz_rot, this selects electron coherence
    orders p_1 = -p_3 = +/-1 and p_2 = 0. Finite pulses include the drift
    Hamiltonian and dissipation; pulse imperfections and nonsecular bath
    terms can leave residual unwanted contributions.

    All Hamiltonians and collapse operators must be time-independent.
    Fixed pulse maps are reused, with one free-evolution map per storage
    delay. Liouville-space propagators are intended for small spin systems.

    The nominal pi/2 duration is 1 / (4 * RabiRate) ns. Total sequence time
    is T + 2*tau + 4*t_pi2, but only T is returned.

    Calls RotFrameSecular, updating H_RF, Sx_rot, Sy_rot, and Sz_rot on
    spinsys. The initial state, physical inputs, and collapse_ops are not
    changed. Rates and collapse operators must already be up to date.
    """
    T_array = np.asarray(T_array, dtype=float)
    tau = float(tau)
    rabi_rate = float(spinsys.RabiRate)

    if spinsys.Spins[spinsys.ReadoutSpin] != 0.5:
        raise ValueError("The eight-step cycle requires a spin-one-half readout.")
    if T_array.ndim != 1 or not np.all(np.isfinite(T_array)) or np.any(T_array < 0):
        raise ValueError("T_array must contain finite, non-negative delays.")
    if not np.isfinite(tau) or tau < 0:
        raise ValueError("tau must be finite and non-negative.")
    if not np.isfinite(rabi_rate) or rabi_rate <= 0:
        raise ValueError("RabiRate must be finite and positive.")

    H_free = RotFrameSecular(spinsys, spinsys.f_RF)
    e_ops = [Qobj(spinsys.Sz_rot)] if e_ops is None else list(e_ops)

    if T_array.size == 0:
        return T_array, np.empty((0, len(e_ops)))

    Omega_R = 2.0 * np.pi * rabi_rate
    t_pi2 = np.pi / (2.0 * Omega_R)
    Sx = Qobj(spinsys.Sx_rot)
    Sy = Qobj(spinsys.Sy_rot)
    pulse_hamiltonians = {
        0: H_free + Omega_R * Sx,
        90: H_free + Omega_R * Sy,
        180: H_free - Omega_R * Sx,
        270: H_free - Omega_R * Sy,
    }

    # All maps act on vectorized density matrices, including unitary cases.
    L_free = liouvillian(H_free, spinsys.collapse_ops)
    free_tau = (L_free * tau).expm()
    pulse_maps = {
        phase: (liouvillian(H, spinsys.collapse_ops) * t_pi2).expm()
        for phase, H in pulse_hamiltonians.items()
    }

    rho0 = spinsys.psi0.proj() if spinsys.psi0.isket else spinsys.psi0
    state_after_tau = free_tau * pulse_maps[0] * operator_to_vector(rho0)
    prefixes = {
        phase: pulse_map * state_after_tau for phase, pulse_map in pulse_maps.items()
    }
    readout_maps = {
        phase: pulse_maps[0] * free_tau * pulse_map
        for phase, pulse_map in pulse_maps.items()
    }

    phase_cycle = [
        (0, 0, +1),
        (180, 0, -1),
        (90, 90, -1),
        (270, 90, +1),
        (180, 180, +1),
        (0, 180, -1),
        (270, 270, -1),
        (90, 270, +1),
    ]

    def calc_point(T):
        free_T = (L_free * float(T)).expm()
        stored_states = {phase: free_T * prefix for phase, prefix in prefixes.items()}

        signal = None
        for phase_2, phase_3, weight in phase_cycle:
            rho_final = vector_to_operator(
                readout_maps[phase_3] * stored_states[phase_2]
            )
            values = np.asarray(expect(e_ops, rho_final))
            signal = weight * values if signal is None else signal + weight * values

        return signal / 8.0

    if n_jobs == 1 or T_array.size <= 1:
        results = [calc_point(T) for T in T_array]
    else:
        results = Parallel(n_jobs=n_jobs, backend="loky", batch_size="auto")(
            delayed(calc_point)(T) for T in T_array
        )

    return T_array, np.asarray(results)


def plot_3pulseESEEMShot_pluspihalf(
    spinsys: SpinSys, T: float, tau: float, tsteps: int = 2500, show: str | None = None
):
    """
    Calculate and plot one three-pulse ESEEM sequence with a final pi/2 pulse.

    This function simulates the complete time evolution of the sequence
    pi/2--tau--pi/2--T--pi/2--tau--pi/2, displays the Sx, Sy, and Sz expectation
    values in a Plotly figure, and shows the corresponding QuTiP Bloch-sphere
    trajectory.

    Parameters
    ----------
    spinsys : SpinSys
        SpinSys object containing the spin system, initial state, collapse operators,
        Rabi rate, and microwave frequency.
    T : float
        Variable free-evolution time between the second and third pi/2 pulses (in ns).
    tau : float
        Fixed free-evolution time following the first and third pi/2 pulses (in ns).
    tsteps : int, optional
        Number of time points used to simulate and plot the sequence.

    show : {"fixed", "responsive", "no"} or None, optional
        Display mode. ``None`` uses the global setting, initially ``"fixed"``.
        ``"fixed"`` uses the preferred width; ``"responsive"`` fills the
        container width. ``"no"`` suppresses both figures without skipping calculations.

    Returns
    -------
    return_dict : dict
        Dictionary containing ``fig`` (the Plotly figure), ``result`` (the QuTiP
        solver result), ``time_values`` (the time grid in ns), and
        ``expectation_values`` (the Sx, Sy, and Sz expectation-value arrays).

    Notes
    -----
    The Plotly figure and QuTiP Bloch sphere are displayed as side effects
    unless the selected display mode is ``"no"``.
    """
    # fixed e_ops:
    e_ops = [
        Qobj(spinsys.Sx[spinsys.ReadoutSpin]),
        Qobj(spinsys.Sy[spinsys.ReadoutSpin]),
        Qobj(spinsys.Sz[spinsys.ReadoutSpin]),
    ]

    # Calculate the Hamiltonian in the rotating frame of the microwave field
    H_rot = RotFrameSecular(spinsys, spinsys.f_RF)

    # Build Pulse Sequence for 3p-ESEEM: pi/2 - tau - pi/2 - T - pi/2 - tau - pi/2
    Omega_R = 2 * np.pi * spinsys.RabiRate
    tp_90 = (np.pi / 2) / Omega_R
    pulse_op = Qobj(spinsys.Sx_rot)
    solver_options = {"max_step": tp_90 / 10.0}

    t1_start = 0.0
    t1_end = tp_90
    t2_start = t1_end + tau
    t2_end = t2_start + tp_90
    t3_start = t2_end + T
    t3_end = t3_start + tp_90
    t4_start = t3_end + tau
    t4_end = t4_start + tp_90

    t_final = t4_end
    tlist = np.linspace(0, t_final, tsteps)

    def Ox_coeff(t, Omega_R):
        if any(
            start <= t <= end
            for start, end in (
                (t1_start, t1_end),
                (t2_start, t2_end),
                (t3_start, t3_end),
                (t4_start, t4_end),
            )
        ):
            return Omega_R
        return 0.0

    H = [
        H_rot,
        [pulse_op, Ox_coeff],
    ]

    args = {"Omega_R": Omega_R}
    result = mesolve(
        H,
        spinsys.psi0,
        tlist,
        c_ops=spinsys.collapse_ops,
        e_ops=e_ops,
        args=args,
        options=solver_options,
    )

    # --------------------------------------------------
    # Plotting

    operator_names = [
        "⟨<i>S</i><sub>x</sub>⟩",
        "⟨<i>S</i><sub>y</sub>⟩",
        "⟨<i>S</i><sub>z</sub>⟩",
    ]
    fig = go.Figure()
    for i, _e_op in enumerate(e_ops):
        fig.add_trace(
            go.Scatter(
                x=tlist,
                y=result.expect[i],
                mode="lines",
                name=operator_names[i],
            )
        )
    fig.update_layout(
        xaxis_title="time (ns)",
        yaxis_title="Expectation Value",
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig.update_xaxes(range=[tlist[0], tlist[-1]])
    plotting.show_figure(fig, preferred_width=800, show=show)

    # Bloch Sphere
    sphere = Bloch()
    sphere.add_points(
        [
            2 * result.expect[0],
            2 * result.expect[1],
            2 * result.expect[2],
        ],
        meth="l",
    )
    # Label the axis according to the expectation values of the spin operators
    sphere.xlabel = [r"$\langle S_x \rangle$", ""]
    sphere.ylabel = [r"$\langle S_y \rangle$", ""]
    sphere.zlabel = [r"$\langle S_z \rangle$", ""]

    if plotting._resolve_display_mode(show) != "no":
        sphere.show()

    return_dict = {
        "fig": fig,
        "result": result,
        "time_values": tlist,
        "expectation_values": result.expect,
    }

    return return_dict


# endregion ESEEM

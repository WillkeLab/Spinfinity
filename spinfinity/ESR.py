"""
Calculating and visualizing Electron Spin Resonance (ESR) spectra.

This module provides calculation and visualization helpers for Electron Spin Resonance
(ESR) spectra of a specified spin systems.

Functions
---------
calc_FreqSweep
    Calculate the Electron Spin Resonance (ESR) spectrum for a given spin system.
calc_RecordedTransitions
    Calculate and record transitions between quantum states within an energy range.
calc_FreqSweep_ESRrate
    Simulate a continuous-wave ESR frequency sweep by solving rate equations.
calc_Populations_ESR
    Calculate the steady-state populations of a spin system under ESR.
add_ESRrate
    Add the ESR transition rates to the given spin system.
plot_FreqSweep
    Plot the ESR spectrum for a given spin system.
plot_FreqSweep_ESRrate
    Compute and plot the ESR spectrum based on the ESR driving rate.
plot_Tipfieldmap
    Plot ESR spectra as a function of tip magnetic field for a given spin system.
plot_ESRPopulations
    Plot the populations of spin states for a given spin system under ESR conditions.

Dependencies
------------
- Uses `spinfinity.SpinSys.SpinSys` class to represent the spin system
  and its properties.
- Uses `spinfinity.SpinHamiltonian` for eigenvalue and eigenvector calculations.
- Uses functions from `spinfinity.Bath` for population and rate calculations.

References
----------
- Yang, K., Willke, P., Bae, Y. et al. *Electrically controlled nuclear polarization of
  individual atoms*. Nature Nanotech 13, 1120–1125 (2018).
  [https://doi.org/10.1038/s41565-018-0296-7](https://doi.org/10.1038/s41565-018-0296-7)
- Yang, K., Phark, SH., Bae, Y. et al. *Probing resonating valence bond states in
  artificial quantum magnets*. Nat Commun 12, 993 (2021).
  [https://doi.org/10.1038/s41467-021-21274-5](https://doi.org/10.1038/s41467-021-21274-5)

"""

import numpy as np
import plotly.colors
import plotly.graph_objects as go
import scipy.constants as const
from matplotlib import cm
from scipy.linalg import null_space

import spinfinity.Bath as bath
import spinfinity.SpinHamiltonian as hamil
from spinfinity import plotting
from spinfinity.SpinSys import SpinSys

# region Calculations


def calc_FreqSweep(
    spinsys: SpinSys,
    freq_array: np.ndarray = None,
    AllowPumping: bool = False,
    lw: float = 0.02,
    norm: bool = True,
    output: bool = True,
):
    """
    Calculate the Electron Spin Resonance (ESR) spectrum for a given spin system.

    This function computes the ESR spectrum by calculating the transition frequencies
    and amplitudes based on the eigenstates of the spin system.
    The ESR signal is constructed as a sum of Lorentzian lines for each allowed
    transition, where the amplitude of each transition depends on the population
    difference and the matrix element of the transition with the tip field operator.

    Parameters
    ----------
    spinsys : SpinSys
        The spin system object containing all relevant spin Hamiltonian parameters
        and state information.
    freq_array : np.ndarray, optional
        An array of frequency points (in GHz) at which the ESR spectrum is calculated.
        If None, a default range from 10 to 20 GHz and 1000 points will be used.
    AllowPumping : bool, optional
        If True, allows for population pumping effects in the calculation.
        Default is False.
    lw : float, optional
        Lorentzian linewidth (full width at half maximum) in GHz. Default is 0.02.
    norm : bool, optional
        If True, scales the signal minimum to zero and maximum to one.
        A constant signal cannot be normalized and raises ValueError.
        If False, returns the unnormalized signal, including zero signals.
        Default is True.
    output : bool, optional
        If True, prints the recorded transitions and their frequencies.
        Default is True. This does not change the returned spectrum.

    Returns
    -------
    freq : ndarray
        Array of frequency values (in GHz) at which the ESR spectrum is calculated.
    ESRsignal : ndarray
        Calculated ESR signal intensity at each frequency point. With norm=True,
        the minimum is zero and the maximum is one.

    Raises
    ------
    ValueError
        If norm=True and the ESR signal is constant.

    Notes
    -----
    The function updates the following attributes of `spinsys`:

    - `RecordedTransitions`: A list of tuples representing the indices of the
        initial and final states for each allowed transition.
    - `ResonanceFrequencies`: An array of resonance frequencies corresponding to the
        recorded transitions.
    """
    # Initialize
    if freq_array is None:
        freq_array = np.linspace(10, 20, 1000, dtype=float)
    N = len(freq_array)
    ESRsignal = np.zeros(N)

    # Lorentzian function equivalent
    def fitlorentz(para, x1):
        return (
            1
            / (para[3] ** 2 + 1)
            * abs(para[1])
            * (1 + (x1 - para[0]) / (0.5 * para[2]) * para[3]) ** 2
            / (1 + ((x1 - para[0]) / (0.5 * para[2])) ** 2)
        )

    # Calculate the Eignevalues and Eignevectors if not already done
    if not hasattr(spinsys, "eigVectors") or spinsys.eigVectors is None:
        hamil.build_Hamiltonian(spinsys)
        hamil.calc_EigEnergies(spinsys)

    bath.calc_Populations(spinsys, AllowPumping=AllowPumping)

    # Look at transitions
    spinsys.RecordedTransitions = calc_RecordedTransitions(
        spinsys, freq_array[0], freq_array[-1]
    )
    spinsys.ResonanceFrequencies = np.zeros(len(spinsys.RecordedTransitions))

    for i in range(len(spinsys.RecordedTransitions)):
        spinsys.ResonanceFrequencies[i] = abs(
            spinsys.E_All_inGHz[spinsys.RecordedTransitions[i][1]]
            - spinsys.E_All_inGHz[spinsys.RecordedTransitions[i][0]]
        )

    if output:
        if len(spinsys.RecordedTransitions) < 300:
            print(f"Found {len(spinsys.RecordedTransitions)} potential transitions:")
            for i in range(len(spinsys.RecordedTransitions)):
                print(
                    f"|{spinsys.RecordedTransitions[i][0]}> "
                    f"-> |{spinsys.RecordedTransitions[i][1]}>; "
                    f"f = {np.round(spinsys.ResonanceFrequencies[i], 4)} GHz"
                )
        else:
            print(
                f"There were {len(spinsys.RecordedTransitions)} "
                "potential transitions in the specified frequency range"
            )

    # Amplitude calculation based on Yang et al. (2021) doi: https://doi.org/10.1038/s41467-021-21274-5
    for k in range(len(spinsys.ResonanceFrequencies)):
        i = spinsys.RecordedTransitions[k][0]  # Initial state index
        j = spinsys.RecordedTransitions[k][1]  # Final state index

        # The signal depends linear on the population difference between the
        # two involved states
        spinsys.dp = abs(spinsys.Populations[i] - spinsys.Populations[j])

        # The drive amplitude is computed by the transition matrix elemeent
        # of the tipfield weighted spin operator vector
        driveamp = (
            abs(
                spinsys.eigVectors[:, j].T.conj()
                @ (
                    spinsys.BTip[0] * spinsys.Sx[spinsys.ReadoutSpin, :, :]
                    + spinsys.BTip[1] * spinsys.Sy[spinsys.ReadoutSpin, :, :]
                    + spinsys.BTip[2] * spinsys.Sz[spinsys.ReadoutSpin, :, :]
                )
                @ spinsys.eigVectors[:, i]
            )
            ** 2
        )

        # The detection amplitude is calculated based on the difference in expectation
        # values of the spin operators for the initial and final states, weighted by
        # the tip field components.
        detectamp = abs(
            spinsys.BTip[0]
            * (
                spinsys.eigVectors[:, i].T.conj()
                @ spinsys.Sx[spinsys.ReadoutSpin, :, :]
                @ spinsys.eigVectors[:, i]
                - spinsys.eigVectors[:, j].T.conj()
                @ spinsys.Sx[spinsys.ReadoutSpin, :, :]
                @ spinsys.eigVectors[:, j]
            )
            + spinsys.BTip[1]
            * (
                spinsys.eigVectors[:, i].T.conj()
                @ spinsys.Sy[spinsys.ReadoutSpin, :, :]
                @ spinsys.eigVectors[:, i]
                - spinsys.eigVectors[:, j].T.conj()
                @ spinsys.Sy[spinsys.ReadoutSpin, :, :]
                @ spinsys.eigVectors[:, j]
            )
            + spinsys.BTip[2]
            * (
                spinsys.eigVectors[:, i].T.conj()
                @ spinsys.Sz[spinsys.ReadoutSpin, :, :]
                @ spinsys.eigVectors[:, i]
                - spinsys.eigVectors[:, j].T.conj()
                @ spinsys.Sz[spinsys.ReadoutSpin, :, :]
                @ spinsys.eigVectors[:, j]
            )
        )

        Amp = spinsys.dp * driveamp * detectamp

        ESRsignal += fitlorentz(
            [spinsys.ResonanceFrequencies[k], Amp, lw, 0], freq_array
        )

    # Normalization
    if norm:
        signal_min = ESRsignal.min()
        signal_range = ESRsignal.max() - signal_min
        if signal_range == 0:
            raise ValueError("Cannot normalize a constant ESR signal; use norm=False.")
        ESRsignal = (ESRsignal - signal_min) / signal_range

    return freq_array, ESRsignal


def calc_RecordedTransitions(spinsys: SpinSys, MinE: float, MaxE: float):
    """
    Calculate and record transitions between quantum states within an energy range.

    This function calculates the recorded transitions based on the occupation
    and the energy differences between states. It records transitions where the state
    has sufficiently high occupation and the energy difference is within the range.

    Parameters
    ----------
    spinsys : SpinSys
        The spin system object containing state populations and energy levels.
    MinE : float
        The minimum allowed energy difference (in GHz) for a transition to be recorded.
    MaxE : float
        The maximum allowed energy difference (in GHz) for a transition to be recorded.

    Returns
    -------
    RecordedTransitions : list of tuple of int
        A list of tuples, each representing a pair of state indices (i, j)
        corresponding to allowed transitions.

    Notes
    -----
    Here no selection rules are applied, transitions within the specified energy range
    which initial states fulfill Population > 10e-5 are recorded.
    """
    # Initialize RecordedTransitions with a dummy value
    # to avoid issues when appending later
    RecordedTransitions = [(0, 0)]

    # Loop over all possible state pairs
    for i in range(spinsys.dimensionOfMatrix):
        # Check if the state has sufficiently high occupation
        #  based on the Boltzmann distribution
        if spinsys.Populations[i] > 10 ** (-5):
            for j in range(i + 1, spinsys.dimensionOfMatrix):
                # Calculate the energy difference
                deltaE = spinsys.E_All_inGHz[j] - spinsys.E_All_inGHz[i]
                if MinE < deltaE < MaxE:
                    RecordedTransitions.append((i, j))

    # Remove the initial dummy (0, 0) element
    RecordedTransitions.pop(0)

    return RecordedTransitions


def calc_FreqSweep_ESRrate(
    spinsys: SpinSys,
    freq_array: np.ndarray = None,
    B1: float = 1e-3,
    T2: float = 200,
    norm: bool = True,
):
    """
    Calculate the ESR spectrum for a given spin system via ESR-Rates.

    This function calculates the ESR signal as a function of frequency
    by solving the rate equations for the spin system, including the
    effects of Rabi driving and relaxation. The ESR signal is computed
    as the difference in tunneling current with and without RF excitation.

    Parameters
    ----------
    spinsys : SpinSys
        The spin system object containing Hamiltonian, rates,
        eigenvectors, and other relevant properties.
    freq_array : np.ndarray, optional
        An array of frequency points (in GHz) at which the ESR spectrum is calculated.
        If None, a default range from 10 to 20 GHz and 200 points will be used.
    B1 : float, optional
        The amplitude of the RF driving field (in T) for the ESR transition
        (default is 1e-3).
    T2 : float, optional
        The total coherence time (in ns) for the ESR transition (default is 200).
        It technically includes T1 contributions from the inelastic tunneling as well
        as pure phase decoherence.
    norm : bool, optional
        If True, normalizes the ESR signal to its maximum value (default is True).

    Returns
    -------
    freq : ndarray
        Array of frequency points (in GHz) at which the ESR signal is calculated.
    ESRsignal : ndarray
        Array of calculated ESR signal values corresponding to each frequency point.
    I_off : float
        The tunneling current without RF excitation,
        used as a reference for the ESR signal.

    Raises
    ------
    ValueError
        If `V_DC` is not set to a non-zero value in the `SpinSys` object,
        as it is required for the tunneling current calculation.

    Notes
    -----
    The function updates the following attributes of `spinsys`:

    - `ESRRates`: A matrix of ESR-induced transition rates between states.
    """
    if freq_array is None:
        freq_array = np.linspace(10, 20, 200, dtype=float)
    N = len(freq_array)
    ESRsignal = np.zeros(N)

    # Calculate the Eignevalues and Eignevectors if not already done
    if not hasattr(spinsys, "eigVectors") or spinsys.eigVectors is None:
        hamil.build_Hamiltonian(spinsys)
        hamil.calc_EigEnergies(spinsys)

    # Check if V_DC is non-zero, as it is required for the tunneling current calculation
    if not hasattr(spinsys, "V_DC") or spinsys.V_DC is None or spinsys.V_DC == 0:
        raise ValueError(
            "V_DC must be set to a non-zero value in the SpinSys object "
            "to calculate the tunneling current."
        )

    # if not hasattr(spinsys, 'Populations') or spinsys.Populations is None:
    bath.calc_Populations(spinsys)

    # The subsequent calculation is based on Yang et al. (2018) doi: https://doi.org/10.1038/s41565-018-0296-7

    # Calculate the RF off tunnel current via the rates.
    I_off = spinsys.b0 * spinsys.G_st * spinsys.V_DC * 1e-3
    for i in range(spinsys.dimensionOfMatrix):
        for j in range(spinsys.dimensionOfMatrix):
            I_temp = (
                const.e
                * spinsys.Populations[i]
                * (spinsys.Rates[0, i, j] - spinsys.Rates[1, i, j])
            )
            I_off += I_temp

    # Initialize the ESR Rates
    spinsys.ESRRates = np.zeros((spinsys.dimensionOfMatrix, spinsys.dimensionOfMatrix))

    # Extract the g-factor and calculate the gyromagnetic ratio for the readout spin
    k = spinsys.ReadoutSpin
    g_k = spinsys.gfactorvector[k]
    gamma_k = g_k * spinsys.muBHz / const.hbar * 1e-9  # rad/ns/T

    # Calculate the ESR Transition Matrix Elements
    TransitionMatrixElement = (
        np.conjugate(np.transpose(spinsys.eigVectors))
        @ spinsys.Sx[spinsys.ReadoutSpin, :, :]
        @ spinsys.eigVectors
    )

    # Calculate the cw current for each frequency based on Yang
    for freq_index in range(N):
        # calculate the ESR rate matrix
        for i in range(spinsys.dimensionOfMatrix):
            for j in range(spinsys.dimensionOfMatrix):
                if i != j:
                    f_ij = abs(spinsys.E_All_inGHz[i] - spinsys.E_All_inGHz[j])

                    delta = 2 * np.pi * (freq_array[freq_index] - f_ij)  # rad/ns

                    Omega_ij = (
                        gamma_k * B1 * abs(TransitionMatrixElement[i, j])
                    )  # rad/ns

                    spinsys.ESRRates[i, j] = (
                        1e9 * 0.5 * Omega_ij**2 * T2 / (1 + (delta * T2) ** 2)
                    )

        # Sum the rates including the ESR rates
        Rates_Summed = (
            spinsys.Rates[0, :, :]
            + spinsys.Rates[1, :, :]
            + spinsys.Rates[2, :, :]
            + spinsys.Rates[3, :, :]
            + spinsys.ESRRates[:, :]
        )

        # Build the Rate-Equation matrix
        K = np.zeros((spinsys.dimensionOfMatrix, spinsys.dimensionOfMatrix))
        for i in range(spinsys.dimensionOfMatrix):
            for j in range(spinsys.dimensionOfMatrix):
                if i != j:
                    K[i, i] -= Rates_Summed[i, j]  # Rates leaving state i
                    K[i, j] = Rates_Summed[j, i]  # Rates leading to state i

        # Solve the Rate-Equation if non-thermal population is desired

        p = null_space(K)
        if p.size == 0:  # If no valid solution, use thermal distribution
            Populations_ESR = spinsys.pop_0
            print("Rate equation has no solution, using thermal distribution")
        else:
            Populations_ESR = p[:, 0] / np.sum(p[:, 0])

        # Deriving the tunneling current
        I_cw = spinsys.b0 * spinsys.G_st * spinsys.V_DC * 1e-3
        for i in range(spinsys.dimensionOfMatrix):
            for j in range(spinsys.dimensionOfMatrix):
                I_temp = (
                    const.e
                    * Populations_ESR[i]
                    * (spinsys.Rates[0, i, j] - spinsys.Rates[1, i, j])
                )
                I_cw += I_temp
        # Delta I
        ESRsignal[freq_index] = I_cw - I_off

    # Normalization
    if norm:
        amplitude = np.max(np.abs(ESRsignal))
        if amplitude > 0:
            ESRsignal = np.abs(ESRsignal) / amplitude

    return freq_array, ESRsignal, I_off


def calc_Populations_ESR(
    spinsys: SpinSys, freq: float = 14, B1: float = 1e-3, T2: float = 200
):
    """
    Calculate the steady-state populations of a spin system under ESR.

    This function updates the populations of the given spin system by
    including ESR-induced transitions, solving the rate equations for
    the steady-state populations. If the rate equation has no solution,
    the thermal distribution is used as a fallback.

    Parameters
    ----------
    spinsys : SpinSys
        The spin system object containing the system's states, rates, and populations.
    freq : float, optional
        The frequency (in GHz) of the ESR driving field (default is 14).
    B1 : float, optional
        The amplitude of the RF driving field (in T) for the ESR transition
        (default is 1e-3).
    T2 : float, optional
        The coherence time (in ns) for the ESR transition (default is 200).

    Returns
    -------
    np.ndarray
        The steady-state populations of the spin system under ESR conditions.

    Raises
    ------
    ValueError
        If `V_DC` is not set to a non-zero value in the `SpinSys` object,
        as it is required for the tunneling current calculation.

    Notes
    -----
    The function updates the following attributes of `spinsys`:

    - `Populations_ESR`: An array containing the steady-state populations of the spin
        system under ESR conditions.
    """
    # Check if V_DC is non-zero, as it is required for the tunneling current calculation
    if not hasattr(spinsys, "V_DC") or spinsys.V_DC is None or spinsys.V_DC == 0:
        raise ValueError(
            "V_DC must be set to a non-zero value in the SpinSys object "
            "to calculate the tunneling current."
        )
    bath.calc_Populations(spinsys, AllowPumping=True)
    add_ESRrate(spinsys, freq=freq, B1=B1, T2=T2)

    K = np.zeros((spinsys.dimensionOfMatrix, spinsys.dimensionOfMatrix))
    Rates_Summed = np.zeros((spinsys.dimensionOfMatrix, spinsys.dimensionOfMatrix))
    Rates_Summed = spinsys.Rates_Summed + spinsys.ESRRates

    for i in range(spinsys.dimensionOfMatrix):
        for j in range(spinsys.dimensionOfMatrix):
            if i != j:
                K[i, i] -= Rates_Summed[i, j]  # Rates leaving state i
                K[i, j] = Rates_Summed[j, i]  # Rates leading to state i
    # Solve the Rate equation to get the steady-state populations
    p = null_space(K)
    if p.size == 0:  # If no valid solution, use thermal distribution
        spinsys.Populations_ESR = spinsys.pop_0
        print("Rate equation has no solution, using thermal distribution")
    else:
        spinsys.Populations_ESR = p[:, 0] / np.sum(p[:, 0])

    return spinsys.Populations_ESR


def add_ESRrate(spinsys: SpinSys, freq: float = 14, B1: float = 1e-3, T2: float = 200):
    """
    Add the ESR (Electron Spin Resonance) transition rates to the given spin system.

    This function calculates the ESR-induced transition rates between
    energy levels of the spin system. The rates are
    computed based on the Rabi rate, coherence time (T2), and the
    frequency of the driving field.

    Parameters
    ----------
    spinsys : SpinSys
        The spin system object to which the ESR rates will be added.
        Must have attributes such as `eigVectors`, `Sx`,
        `ReadoutSpin`, `E_All_inGHz`, `dimensionOfMatrix`, and `Rates`.
    freq : float, optional
        The frequency (in GHz) of the driving field (default is 14).
    B1 : float, optional
        The amplitude of the RF driving field (in T) for the ESR transition
        (default is 1e-3).
    T2 : float, optional
        The coherence time (in ns) for the ESR transition (default is 200).
    """
    if not hasattr(spinsys, "Rates") or spinsys.Rates is None:
        bath.calc_Rates(spinsys)

    # Initialize the ESR Rates
    spinsys.ESRRates = np.zeros((spinsys.dimensionOfMatrix, spinsys.dimensionOfMatrix))

    # Extract the g-factor and calculate the gyromagnetic ratio for the readout spin
    k = spinsys.ReadoutSpin
    g_k = spinsys.gfactorvector[k]
    mu_B = const.physical_constants["Bohr magneton"][0]  # J/T
    gamma_k = g_k * mu_B / const.hbar * 1e-9  # rad/ns/T

    # Calculate the ESR Transition Matrix Elements
    TransitionMatrixElement = (
        np.conjugate(np.transpose(spinsys.eigVectors))
        @ spinsys.Sx[spinsys.ReadoutSpin, :, :]
        @ spinsys.eigVectors
    )

    # calculate the ESR rate matrix
    for i in range(spinsys.dimensionOfMatrix):
        for j in range(spinsys.dimensionOfMatrix):
            if i != j:
                f_ij = abs(spinsys.E_All_inGHz[i] - spinsys.E_All_inGHz[j])

                delta = 2 * np.pi * (freq - f_ij)  # rad/ns

                Omega_ij = gamma_k * B1 * abs(TransitionMatrixElement[i, j])  # rad/ns

                spinsys.ESRRates[i, j] = (
                    1e9 * 0.5 * Omega_ij**2 * T2 / (1 + (delta * T2) ** 2)
                )

    return


# endregion Calculations

# region Visualization


def plot_FreqSweep(
    spinsys: SpinSys,
    freq_array: np.ndarray = None,
    AllowPumping: bool = False,
    lw: float = 0.02,
    norm: bool = True,
    output: bool = False,
    show: str | None = None,
):
    """
    Plot the ESR spectrum for a given spin system.

    This function computes the ESR spectrum using the provided spin
    system using `calc_FreqSweep` and plots the resulting signal.
    It also marks the resonance frequencies corresponding to
    recorded transitions in the spin system.

    Parameters
    ----------
    spinsys : SpinSys
        The spin system object containing resonance frequencies and
        recorded transitions.
    freq_array : np.ndarray, optional
        An array of frequency points (in GHz) at which the ESR spectrum is calculated.
        If None, a default range from 10 to 20 GHz and 1000 points will be used.
    AllowPumping : bool, optional
        Whether to allow pumping effects in the ESR calculation (default is False).
    lw : float, optional
        Linewidth for the ESR calculation (default is 0.02).
    norm : bool, optional
        Whether to normalize the ESR signal (default is True).
    output : bool, optional
        If True, prints the recorded transitions and their frequencies.
        Default is False. This is independent of the figure display mode.
    show : {"fixed", "responsive", "no"} or None, optional
        Display mode. ``None`` uses the global setting, initially ``"fixed"``.
        ``"fixed"`` uses the preferred width; ``"responsive"`` fills the
        container width. ``"no"`` suppresses display without skipping calculations.

    Returns
    -------
    return_dict : dict
        A dictionary containing:
        - 'fig': plotly.graph_objects.Figure
            The plotly figure object containing the ESR spectrum plot.
        - 'Freq_array': numpy.ndarray
            The array of frequency points in GHz.
        - 'ESRsignal': numpy.ndarray
            The calculated ESR signal corresponding to the frequency points.
    """
    # calculate the ESR spectrum
    Freq_array, ESRsignal = calc_FreqSweep(
        spinsys,
        freq_array=freq_array,
        AllowPumping=AllowPumping,
        lw=lw,
        norm=norm,
        output=output,
    )

    # Make the plot
    Nc = len(spinsys.ResonanceFrequencies)
    color_list = cm.jet(np.linspace(0, 1, Nc))
    color_list = [
        f"rgb({int(c[0] * 255)},{int(c[1] * 255)},{int(c[2] * 255)})"
        for c in color_list
    ]

    # Create the main ESR signal trace
    fig = go.Figure()
    fig.update_layout(plot_bgcolor="white", paper_bgcolor="white")
    fig.add_trace(
        go.Scatter(
            x=Freq_array,
            y=ESRsignal,
            mode="lines",
            line=dict(color="black", width=2),
            name="ESR Signal",
        )
    )

    # Mark resonances
    for i, freq_res in enumerate(spinsys.ResonanceFrequencies):
        index = np.argmin(np.abs(Freq_array - freq_res))
        amp = ESRsignal[index]
        if abs(amp) > 0.01 * np.max(np.abs(ESRsignal)):  # Filter very small peaks
            transition_str = (
                f"|{spinsys.RecordedTransitions[i][0]}> "
                f"→ |{spinsys.RecordedTransitions[i][1]}>; "
                f"f = {np.round(freq_res, 4)} GHz"
            )
            fig.add_trace(
                go.Scatter(
                    x=[freq_res],
                    y=[amp],
                    mode="markers",
                    marker=dict(color=color_list[i], size=12),
                    name=transition_str,
                )
            )

    fig.update_layout(
        xaxis_title="Freq (GHz)",
        yaxis_title="ΔI (a.u.)",
        title="ESR-Simulation",
        legend_title="Transitions",
        showlegend=True,
    )
    fig.update_xaxes(range=[Freq_array[0], Freq_array[-1]])
    plotting.show_figure(fig, preferred_width=800, show=show)

    return_dict = {"fig": fig, "Freq_array": Freq_array, "ESRsignal": ESRsignal}

    return return_dict


def plot_FreqSweep_ESRrate(
    spinsys: SpinSys,
    freq_array: np.ndarray = None,
    B1: float = 0.001,
    T2: float = 200,
    norm: bool = True,
    show: str | None = None,
):
    """
    Plot the ESR spectrum based on the ESR driving rate.

    This function computes the ESR spectrum based on an ESR driving rate
    for a given spin system using `calc_FreqSweep_ESRrate` and plots
    the resulting ESR signal. It also marks the resonance frequencies
    corresponding to allowed transitions in the spin system.

    Parameters
    ----------
    spinsys : SpinSys
        The spin system object containing energy levels and transition information.
    freq_array : np.ndarray, optional
        An array of frequency points (in GHz) at which the ESR spectrum is calculated.
        If None, a default range from 10 to 20 GHz and 200 points will be used.
    B1 : float, optional
        The amplitude of the RF driving field (in T) for the ESR transition
        (default is 1e-3).
    T2 : float, optional
        The coherence time (in ns) for the ESR transition (default is 200).
    norm : bool, optional
        Whether to normalize the ESR signal (default is True).

    show : {"fixed", "responsive", "no"} or None, optional
        Display mode. ``None`` uses the global setting, initially ``"fixed"``.
        ``"fixed"`` uses the preferred width; ``"responsive"`` fills the
        container width. ``"no"`` suppresses display without skipping calculations.

    Returns
    -------
    return_dict : dict
        A dictionary containing:
        - 'fig': plotly.graph_objects.Figure
            The plotly figure object containing the ESR spectrum plot.
        - 'Freq_array': numpy.ndarray
            The array of frequency points in GHz.
        - 'ESRsignal': numpy.ndarray
            The calculated ESR signal corresponding to the frequency points.
    """
    # Calculate the ESR spectrum using the specified parameters
    Freq_array, ESRsignal, _ = calc_FreqSweep_ESRrate(
        spinsys,
        freq_array=freq_array,
        B1=B1,
        T2=T2,
        norm=norm,
    )

    # Record the transitions
    spinsys.RecordedTransitions = calc_RecordedTransitions(
        spinsys,
        Freq_array[0],
        Freq_array[-1],
    )
    spinsys.ResonanceFrequencies = np.zeros(len(spinsys.RecordedTransitions))
    for i in range(len(spinsys.RecordedTransitions)):
        spinsys.ResonanceFrequencies[i] = abs(
            spinsys.E_All_inGHz[spinsys.RecordedTransitions[i][1]]
            - spinsys.E_All_inGHz[spinsys.RecordedTransitions[i][0]]
        )

    # Make the plot
    color_list = plotly.colors.qualitative.Plotly[: len(spinsys.ResonanceFrequencies)]

    # Create the main ESR signal trace
    fig = go.Figure()
    fig.update_layout(plot_bgcolor="white", paper_bgcolor="white")
    fig.add_trace(
        go.Scatter(
            x=Freq_array,
            y=ESRsignal,
            mode="lines",
            line=dict(color="black", width=2),
            name="ESR Signal",
        )
    )

    # Mark resonances
    for i, freq_res in enumerate(spinsys.ResonanceFrequencies):
        index = np.argmin(np.abs(Freq_array - freq_res))
        amp = ESRsignal[index]
        if abs(amp) > 0.01 * np.max(np.abs(ESRsignal)):  # Filter very small peaks
            transition_str = (
                f"|{spinsys.RecordedTransitions[i][0]}> "
                f"→ |{spinsys.RecordedTransitions[i][1]}>; "
                f"f = {np.round(freq_res, 4)} GHz"
            )
            fig.add_trace(
                go.Scatter(
                    x=[freq_res],
                    y=[amp],
                    mode="markers",
                    marker=dict(color=color_list[i % len(color_list)], size=12),
                    name=transition_str,
                )
            )

    yaxis_title = "Normalized ΔI (a.u.)" if norm else "ΔI (A)"

    fig.update_layout(
        xaxis_title="Freq (GHz)",
        yaxis_title=yaxis_title,
        title="ESR-Simulation",
        legend_title="Transitions",
        showlegend=True,
    )
    fig.update_xaxes(range=[Freq_array[0], Freq_array[-1]])
    plotting.show_figure(fig, preferred_width=800, show=show)

    return_dict = {"fig": fig, "Freq_array": Freq_array, "ESRsignal": ESRsignal}

    return return_dict


def plot_Tipfieldmap(
    spinsys: SpinSys,
    freq_array: np.ndarray = None,
    BTip_array: np.ndarray = None,
    lw: float = 0.02,
    norm: bool = True,
    AllowPumping: bool = False,
    Angle: float = 0,
    show: str | None = None,
):
    """
    Plot ESR spectra as a function of tip magnetic field for a given spin system.

    The function caluclates the ESR spectrum for a set of tipfields unsing
    `calc_FreqSweep` and plots the resulting ESR signal as a color map.
    The x-axis represents the frequency, the y-axis represents the tip magnetic field,
    and the color intensity represents the ESR signal strength. Additionally,
    a secondary y-axis is added to show the detuning from the tip field in GHz.

    Parameters
    ----------
    spinsys : SpinSys
        The spin system object containing all relevant spin and field parameters.
    freq_array : np.ndarray, optional
        An array of frequency points (in GHz) at which the ESR spectrum is calculated.
        If None, a default range from 10 to 20 GHz and 500 points will be used.
    BTip_array : np.ndarray, optional
        An array of tip magnetic field values (T) at which to calculate the spectrum.
        If None, a default range from -0.1 to 0.1 T and 100 points will be used.
    lw : float, optional
        Linewidth for the ESR peaks (default is 0.02).
    norm : bool, optional
        Whether to normalize the ESR signal (default is True).
    AllowPumping : bool, optional
        Whether to allow pumping effects in the ESR calculation (default is False).
    Angle : float, optional
        Angle (in degrees) of the tip field with respect to the z-axis (default is 0).

    show : {"fixed", "responsive", "no"} or None, optional
        Display mode. ``None`` uses the global setting, initially ``"fixed"``.
        ``"fixed"`` uses the preferred width; ``"responsive"`` fills the
        container width. ``"no"`` suppresses display without skipping calculations.

    Returns
    -------
    return_dict : dict
        A dictionary containing:
        - 'fig': plotly.graph_objects.Figure
            The plotly figure object containing the ESR spectrum plot.
        - 'Freq_array': numpy.ndarray
            The array of frequency points in GHz.
        - 'BTip_array': numpy.ndarray
            The array of tip magnetic field values in Tesla.
        - 'ESRsignal': numpy.ndarray
            The calculated ESR signal corresponding to the frequency points.
    """
    # Save previous BTip value
    BTip_initial = spinsys.BTip.copy()

    # Predefine Arrays
    if freq_array is None:
        freq_array = np.linspace(10, 20, 500, dtype=float)
    if BTip_array is None:
        BTip_array = np.linspace(-0.1, 0.1, 100, dtype=float)
    N_B = len(BTip_array)
    N_Freq = len(freq_array)

    ESRsignal = np.zeros((N_B, N_Freq), dtype=float)

    # calculating the ESR signal for each Field
    for i in range(N_B):
        spinsys.BTip[2] = BTip_array[i] * np.cos(Angle * np.pi / 180)
        spinsys.BTip[0] = BTip_array[i] * np.sin(Angle * np.pi / 180) + 1e-9
        spinsys.BTip[1] = 0  # BTip[0] is the y-component of the tip field
        hamil.build_Hamiltonian(spinsys)
        hamil.calc_EigEnergies(spinsys)
        if AllowPumping:
            bath.calc_Rates(spinsys)
        f, ESRsignal[i][:] = calc_FreqSweep(
            spinsys,
            norm=norm,
            lw=lw,
            freq_array=freq_array,
            AllowPumping=AllowPumping,
            output=False,
        )

    # Calculating the Detuning from the Tipfield
    Detuning = (
        spinsys.gfactorvector[spinsys.ReadoutSpin]
        * spinsys.muB
        * (BTip_array + spinsys.B[2])
        * spinsys.meVtoGHzConversion
        - spinsys.gfactorvector[spinsys.ReadoutSpin]
        * spinsys.muB
        * spinsys.B[2]
        * spinsys.meVtoGHzConversion
    )

    # Plot the ESR Signal as a color plot
    fig = go.Figure(
        data=go.Heatmap(
            z=ESRsignal,
            x=freq_array,
            y=BTip_array * 1e3,  # Convert to mT for y-axis
            colorscale="Blues",
            colorbar=dict(title="ESR Signal"),
            zsmooth=False,
        )
    )

    # Add labels and title
    fig.update_layout(
        xaxis_title="Freq (GHz)",
        yaxis_title="B_Tip (mT)",
        title=f"Tipfield dependent ESR (Angle = {Angle}°)",
    )

    # Add secondary y-axis for Detuning (GHz)
    # We'll add a custom axis on the right with detuning values
    # Calculate detuning for each BTip value
    detuning_labels = np.round(Detuning, 2)
    fig.update_layout(
        yaxis=dict(title="B_Tip (mT)", side="left"),
        yaxis2=dict(
            title="Detuning (GHz)",
            overlaying="y",
            side="right",
            tickvals=BTip_array * 1e3,
            ticktext=detuning_labels,
        ),
    )

    plotting.show_figure(fig, preferred_width=800, show=show)

    # Restore the initial BTip value
    spinsys.BTip = BTip_initial
    hamil.build_Hamiltonian(spinsys)
    hamil.calc_EigEnergies(spinsys)

    return_dict = {
        "fig": fig,
        "Freq_array": freq_array,
        "BTip_array": BTip_array,
        "ESRsignal": ESRsignal,
    }

    return return_dict


def plot_ESRPopulations(
    spinsys: SpinSys,
    freq: float = 14,
    B1: float = 0.001,
    T2: float = 200,
    filter: float = None,
    show: str | None = None,
):
    """
    Plot the populations of spin states for a given spin system under ESR conditions.

    This function calculates and visualizes the populations of spin
    states, including Boltzmann, tunneling, and ESR-driven populations,
    for a provided spin system. The populations are displayed as
    horizontal bar plots. Uses `calc_Populations_ESR` to compute the ESR-driven
    populations.

    Parameters
    ----------
    spinsys : SpinSys
        The spin system object containing the relevant state and population information.
    freq : float, optional
        The frequency of the ESR field in the same units as used in the
        simulation, by default 14.
    B1 : float, optional
        The amplitude of the RF driving field (in T) for the ESR transition
        (default is 1e-3).
    T2 : float, optional
        The coherence time (in ns) for the ESR transition (default is 200).
    filter : float, optional
        If provided, only states with populations greater than this
        value in any of the three population arrays will be displayed.
        Default is None, which means all states are shown.

    show : {"fixed", "responsive", "no"} or None, optional
        Display mode. ``None`` uses the global setting, initially ``"fixed"``.
        ``"fixed"`` uses the preferred width; ``"responsive"`` fills the
        container width. ``"no"`` suppresses display without skipping calculations.

    Returns
    -------
    return_dict : dict
        A dictionary containing:
        - 'fig': plotly.graph_objects.Figure
            The plotly figure object containing the plot.
        - 'Pop_Boltzmann': numpy.ndarray
            The Boltzmann populations of the spin states.
        - 'Pop_Tunneling': numpy.ndarray
            The steady state populations under the influence of the
            tunneling rates of the spin states.
        - 'Pop_ESR': numpy.ndarray
            The ESR-driven populations of the spin states.
    """
    calc_Populations_ESR(spinsys, freq=freq, B1=B1, T2=T2)
    if not hasattr(spinsys, "basisSates") or spinsys.basisStates is None:
        hamil.calc_EigStates(spinsys)

    width = 0.2

    if filter is None:
        # Plot all states with plotly
        labels = spinsys.statesWithoutE
        positions = np.arange(len(spinsys.Populations))
        pop0 = np.array(spinsys.pop_0)
        popT = np.array(spinsys.Populations)
        popESR = np.array(spinsys.Populations_ESR)

        fig = go.Figure()
        fig.update_layout(plot_bgcolor="white", paper_bgcolor="white")
        fig.add_trace(
            go.Bar(
                x=pop0,
                y=labels,
                orientation="h",
                name="Boltzmann Populations",
                marker=dict(color="blue"),
                opacity=0.5,
            )
        )
        fig.add_trace(
            go.Bar(
                x=popT,
                y=labels,
                orientation="h",
                name="Tunneling Populations",
                marker=dict(color="red"),
                opacity=0.5,
            )
        )
        fig.add_trace(
            go.Bar(
                x=popESR,
                y=labels,
                orientation="h",
                name="ESR Populations",
                marker=dict(color="green"),
                opacity=0.5,
            )
        )

        fig.update_layout(
            barmode="group",
            xaxis_title="Population",
            yaxis=dict(
                title="Basis States",
                tickmode="array",
                tickvals=positions,
                ticktext=labels,
            ),
            title="Population of Spin States",
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
        )

        plotting.show_figure(fig, preferred_width=800, show=show)
    else:
        # Only display states with population > filter in any of the three populations
        pop_arrays = [spinsys.pop_0, spinsys.Populations, spinsys.Populations_ESR]
        mask = np.any([np.array(p) >= filter for p in pop_arrays], axis=0)

        filtered_indices = np.where(mask)[0]
        if len(filtered_indices) == 0:
            print(f"No states with population greater than {filter * 100}%.")
        else:
            labels = np.array(spinsys.statesWithoutE)[mask]
            positions = np.arange(len(filtered_indices))
            pop0 = np.array(spinsys.pop_0)[mask]
            popT = np.array(spinsys.Populations)[mask]
            popESR = np.array(spinsys.Populations_ESR)[mask]

            fig = go.Figure()
            fig.update_layout(plot_bgcolor="white", paper_bgcolor="white")
            fig.add_trace(
                go.Bar(
                    x=pop0,
                    y=positions,
                    name="Boltzmann Populations",
                    orientation="h",
                    marker=dict(color="blue"),
                    opacity=0.5,
                    width=width,
                )
            )
            fig.add_trace(
                go.Bar(
                    x=popT,
                    y=positions,
                    name="Tunneling Populations",
                    orientation="h",
                    marker=dict(color="red"),
                    opacity=0.5,
                    width=width,
                )
            )
            fig.add_trace(
                go.Bar(
                    x=popESR,
                    y=positions,
                    name="ESR Populations",
                    orientation="h",
                    marker=dict(color="green"),
                    opacity=0.5,
                    width=width,
                )
            )

            fig.update_layout(
                barmode="group",
                xaxis_title="Population",
                yaxis=dict(
                    title="Basis States",
                    tickmode="array",
                    tickvals=positions,
                    ticktext=labels,
                ),
                title=f"Population of Spin States (> {filter * 100}%)",
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
                ),
            )

            plotting.show_figure(fig, preferred_width=800, show=show)

    return_dict = {
        "fig": fig,
        "Pop_Boltzmann": spinsys.pop_0,
        "Pop_Tunneling": spinsys.Populations,
        "Pop_ESR": spinsys.Populations_ESR,
    }

    return return_dict


# endregion Visualization

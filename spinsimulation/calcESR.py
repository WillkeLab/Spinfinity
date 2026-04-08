import numpy as np
import plotly.colors
import plotly.graph_objects as go
import scipy.constants as const
from scipy.linalg import null_space

import spinsimulation.calcBath as bath
import spinsimulation.SpinHamiltonian as hamil
from spinsimulation.SpinSys import SpinSys


def calcESR_FreqSweep(spinsys: SpinSys, FreqRange: list = [10, 20], AllowPumping: bool = False, N: int = 1000, lw: float = 0.02, norm: bool = True, output: bool = True):
    """
    Calculate the Electron Spin Resonance (ESR) spectrum for a given spin system.

    Parameters
    ----------
    spinsys : SpinSys
        The spin system object containing all relevant spin Hamiltonian parameters and state information.
    FreqRange : list of float, optional
        The frequency range [start, stop] in GHz over which to calculate the ESR spectrum. Default is [10, 20].
    AllowPumping : bool, optional
        If True, allows for population pumping effects in the calculation. Default is False.
    N : int, optional
        Number of frequency points in the calculated spectrum. Default is 1000.
    lw : float, optional
        Lorentzian linewidth (full width at half maximum) in GHz. Default is 0.02.
    norm : bool, optional
        If True, normalizes the ESR signal to the range [0, 1]. Default is True.

    Returns
    -------
    freq : ndarray
        Array of frequency values (in GHz) at which the ESR spectrum is calculated.
    ESRsignal : ndarray
        Calculated ESR signal intensity at each frequency point.

    Notes
    -----
    - The function computes transition frequencies and amplitudes based on the eigenstates of the spin system.
    - If the number of transitions is less than 300, details of each transition are printed.
    - The ESR signal is constructed as a sum of Lorentzian lines for each allowed transition.
    """
    # Initialize
    spinsys.ESRshift = 0  # Dont remember what thuis is for
    freq = np.linspace(FreqRange[0], FreqRange[1], N)  # Frequency array
    ESRsignal = np.zeros(len(freq))

    # Lorentzian function equivalent
    fitlorentz = lambda para, x1: (1 / (para[3]**2 + 1) * abs(para[1]) * 
                                    (1 + (x1 - para[0]) / (0.5 * para[2]) * para[3])**2 / 
                                    (1 + ((x1 - para[0]) / (0.5 * para[2]))**2))

    # Calculate the Eignevalues and Eignevectors if not already done
    if not hasattr(spinsys, 'eigVectors') or spinsys.eigVectors is None:
        hamil.calcEigEnergies(spinsys)

    bath.calcPopulations(spinsys, AllowPumping=AllowPumping)

    # Look at transitions
    spinsys.RecordedTransitions = calcRecordedTransitions(spinsys, freq[0], freq[-1])
    spinsys.ResonanceFrequencies = np.zeros(len(spinsys.RecordedTransitions))

    for i in range(len(spinsys.RecordedTransitions)):
        spinsys.ResonanceFrequencies[i] = abs(spinsys.E_All_inGHz[spinsys.RecordedTransitions[i][1]] - 
                                        spinsys.E_All_inGHz[spinsys.RecordedTransitions[i][0]])

    # Optionally print the transitions
    if output:
        if len(spinsys.RecordedTransitions) < 300:
            print(f"Found {len(spinsys.RecordedTransitions)} potential transitions:")
            for i in range(len(spinsys.RecordedTransitions)):
                print(f"|{spinsys.RecordedTransitions[i][0]}> -> |{spinsys.RecordedTransitions[i][1]}>; "
                    f"f = {np.round(spinsys.ResonanceFrequencies[i], 4)} GHz")
        else:
            print(f"There were {len(spinsys.RecordedTransitions)} potential transitions in the specified frequency range")

    # Amplitude calculation
    for k in range(len(spinsys.ResonanceFrequencies)):
        i = spinsys.RecordedTransitions[k][0]  # Initial state index
        j = spinsys.RecordedTransitions[k][1]  # Final state index

        spinsys.dp = abs(spinsys.Populations[i] - spinsys.Populations[j])

        # Rabi rate factor
        rabiRateFactor = 1

        spinsys.M_x = rabiRateFactor * (spinsys.eigVectors[:, j].T.conj() @ spinsys.Sx[spinsys.ReadoutSpin, :, :] @ spinsys.eigVectors[:, i])
        spinsys.M_y = rabiRateFactor * (spinsys.eigVectors[:, j].T.conj() @ spinsys.Sy[spinsys.ReadoutSpin, :, :] @ spinsys.eigVectors[:, i])
        spinsys.M_z = rabiRateFactor * (spinsys.eigVectors[:, j].T.conj() @ spinsys.Sz[spinsys.ReadoutSpin, :, :] @ spinsys.eigVectors[:, i])

        spinsys.Amp = spinsys.dp * (spinsys.BTip[0] * spinsys.M_x + spinsys.BTip[1] * spinsys.M_y + spinsys.BTip[2] * spinsys.M_z)**2

        ESRsignal += fitlorentz([spinsys.ResonanceFrequencies[k], spinsys.Amp, lw, 0], freq)

    # Normalization
    if norm == 1:
        ESRsignal = (ESRsignal - ESRsignal.min()) / (ESRsignal.max() - ESRsignal.min())

    return freq, ESRsignal


def calcRecordedTransitions(spinsys: SpinSys, MinE: float, MaxE: float):
    """
    Calculate and record all transitions between quantum states within a given energy range. 
    
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
        A list of tuples, each representing a pair of state indices (i, j) corresponding to allowed transitions.

    Notes
    -----
    Here no selection rules are applied, so all transitions within the specified energy range are recorded.
    """
    # This function calculates the recorded transitions based on the occupation probability
    # and the energy differences between states. It records transitions where the state
    # has sufficiently high occupation and the energy difference is within the specified range.

    # Initialize RecordedTransitions with a dummy value to avoid issues when appending later
    RecordedTransitions = [(0, 0)]

    # Loop over all possible state pairs
    for i in range(spinsys.dimensionOfMatrix):
    # Check if the state has sufficiently high occupation based on the Boltzmann distribution
        if spinsys.Populations[i] > 10**(-5):
            for j in range(i + 1, spinsys.dimensionOfMatrix):
            # Calculate the energy difference
                deltaE = spinsys.E_All_inGHz[j] - spinsys.E_All_inGHz[i]
                if MinE < deltaE < MaxE:
                    RecordedTransitions.append((i, j))
            
    # Remove the initial dummy (0, 0) element
    RecordedTransitions.pop(0)

    return RecordedTransitions 


def calcESR_FreqSweep_ESRrate(
        spinsys: SpinSys, 
        FreqRange: list = [10, 20], 
        N: int = 200, 
        norm: bool = True):
    """
    Simulate a continuous-wave ESR (Electron Spin Resonance) frequency sweep for a given spin system.

    This function calculates the ESR signal as a function of frequency by solving the rate equations
    for the spin system, including the effects of Rabi driving and relaxation.
    The ESR signal is computed as the difference in tunneling current with and without RF excitation.

    Parameters
    ----------
    spinsys : SpinSys
        The spin system object containing Hamiltonian, rates, eigenvectors, and other relevant properties.
    FreqRange : list of float, optional
        The frequency range [start, stop] in GHz over which to perform the sweep (default is [10, 20]).
    N : int, optional
        Number of frequency points in the sweep (default is 200).
    norm : bool, optional
        If True, normalizes the ESR signal to its maximum value (default is True).

    Returns
    -------
    freq : ndarray
        Array of frequency points (in GHz) at which the ESR signal is calculated.
    ESRsignal : ndarray
        Array of calculated ESR signal values corresponding to each frequency point.

    Notes
    -----
    - The function assumes that the `spinsys` object provides all necessary matrices and properties,
        including eigenvectors, rates, and population vectors.
    - The ESR signal is calculated as the difference in tunneling current with and without RF excitation.
    - The rate equations are solved for each frequency point.
    """
    freq = np.linspace(FreqRange[0], FreqRange[1], N)  # Frequency array
    ESRsignal = np.zeros(len(freq))

    # Calculate the Eignevalues and Eignevectors if not already done
    if not hasattr(spinsys, 'eigVectors') or spinsys.eigVectors is None:
        hamil.calcEigEnergies(spinsys)

    # Calculate the RF off tunnel current via the rates

    # if not hasattr(spinsys, 'Populations') or spinsys.Populations is None:
    bath.calcPopulations(spinsys)
    
    I_off = 0
    for i in range(spinsys.dimensionOfMatrix):
        for j in range(spinsys.dimensionOfMatrix):
            I_temp = const.e * spinsys.Populations[i] * (spinsys.Rates[0, i, j] - spinsys.Rates[1, i, j])
            I_off += I_temp
    
    # Initialize the ESR Rates
    spinsys.ESRRates = np.zeros((spinsys.dimensionOfMatrix, spinsys.dimensionOfMatrix))

    # Calculate the ESR Transition Matrix Elements
    TransitionMatrixElement = np.conjugate(np.transpose(spinsys.eigVectors)) @ spinsys.Sx[spinsys.ReadoutSpin, :, :] @ spinsys.eigVectors
    
    # Calculate the cw current for each frequency
    for freq_index in range(N):
        # calculate the ESR rate matrix
        for i in range(spinsys.dimensionOfMatrix):
            for j in range(spinsys.dimensionOfMatrix):
                if i != j:
                    spinsys.ESRRates[i, j] = 0.5 * spinsys.RabiRate**2 * abs(TransitionMatrixElement[i, j])**2 * spinsys.T2 / ((freq[freq_index] - abs(spinsys.E_All_inGHz[i] - spinsys.E_All_inGHz[j]))**2 * spinsys.T2**2 + 1)

        # Sum the rates including the ESR rates
        Rates_Summed = (
            spinsys.Rates[0, :, :] + spinsys.Rates[1, :, :] +
            spinsys.Rates[2, :, :] + spinsys.Rates[3, :, :] +
            spinsys.ESRRates[:, :]
        )

        # Build the Rate-Equation matrix
        K = np.zeros((spinsys.dimensionOfMatrix, spinsys.dimensionOfMatrix))
        for i in range(spinsys.dimensionOfMatrix):
            for j in range(spinsys.dimensionOfMatrix):
                if i != j:
                    K[i, i] -= Rates_Summed[i, j]  # Rates leaving state i
                    K[i, j] = Rates_Summed[j, i]   # Rates leading to state i

        # Solve the Rate-Equation if non-thermal population is desired
        
        p = null_space(K)
        if p.size == 0:  # If no valid solution, use thermal distribution
            Populations_ESR = spinsys.pop_0
            print("Rate equation has no solution, using thermal distribution")
        else:
            Populations_ESR = p[:, 0] / np.sum(p[:, 0])

        # Deriving the tunneling current 
        I_cw = 0
        for i in range(spinsys.dimensionOfMatrix):
            for j in range(spinsys.dimensionOfMatrix):
                I_temp = const.e * Populations_ESR[i] * (spinsys.Rates[0, i, j] - spinsys.Rates[1, i, j])
                I_cw += I_temp

        ESRsignal[freq_index] = (I_cw - I_off) / I_off  # Calculate the ESR signal as the relative change in current 

    # Normalization
    if norm:
        ESRsignal = abs(ESRsignal) / abs(ESRsignal).max()

    return freq, ESRsignal


def calcPopulations_ESR(spinsys: SpinSys, freq: float = 14):
    """
    Calculate the steady-state populations of a spin system under ESR (Electron Spin Resonance) conditions.

    This function updates the populations of the given spin system by including ESR-induced transitions,
    solving the rate equations for the steady-state populations. If the rate equation has no solution,
    the thermal distribution is used as a fallback.

    Parameters
    ----------
    spinsys : SpinSys
        The spin system object containing the system's states, rates, and populations.
    freq : float, optional
        The frequency (in GHz) of the ESR driving field (default is 14).

    Returns
    -------
    np.ndarray
        The steady-state populations of the spin system under ESR conditions.

    Notes
    -----
    - The function modifies the `spinsys` object in-place, updating its `Populations_ESR` attribute.
    - If the rate matrix has no null space (i.e., no steady-state solution), the thermal population (`spinsys.pop_0`) is used.
    """
    bath.calcPopulations(spinsys, AllowPumping=True)
    addESRrate(spinsys, freq=freq)

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


def addESRrate(spinsys: SpinSys, freq: float = 14):
    """
    Add the ESR (Electron Spin Resonance) transition rates to the given spin system.

    This function calculates the ESR-induced transition rates between energy levels of the spin system
    and updates the `Rates` and `Rates_Summed` attributes of the `spinsys` object. The rates are computed
    based on the Rabi rate, coherence time (T2), and the frequency of the driving field.

    Parameters
    ----------
    spinsys : SpinSys
        The spin system object to which the ESR rates will be added. Must have attributes such as
        `eigVectors`, `Sx`, `ReadoutSpin`, `E_All_inGHz`, `dimensionOfMatrix`, and `Rates`.
    freq : float, optional
        The frequency (in GHz) of the driving field (default is 14).
        
    Notes
    -----
    - If the `Rates` attribute of `spinsys` is not present or is `None`, the function calls `bath.calcRates(spinsys)`
      to initialize it.
    - The ESR rates are stored in the 5th index (index 4) of the `Rates` array.
    - The total rates are summed and stored in `spinsys.Rates_Summed`.
    """
    if not hasattr(spinsys, 'Rates') or spinsys.Rates is None:
        bath.calcRates(spinsys)

    # Initialize the ESR rates if not already present
    spinsys.ESRRates = np.zeros((spinsys.dimensionOfMatrix, spinsys.dimensionOfMatrix))

    # Calculate the matrix element for the ESR rate
    TransitionMatrixElement = np.conjugate(np.transpose(spinsys.eigVectors)) @ spinsys.Sx[spinsys.ReadoutSpin, :, :] @ spinsys.eigVectors

    # Calculate each entry individually
    for i in range(spinsys.dimensionOfMatrix):
        for j in range(spinsys.dimensionOfMatrix):
            if i != j:
                spinsys.ESRRates[i, j] = 0.5 * spinsys.RabiRate**2 * abs(TransitionMatrixElement[i, j])**2 * spinsys.T2 / ((freq - abs(spinsys.E_All_inGHz[i] - spinsys.E_All_inGHz[j]))**2 * spinsys.T2**2 + 1)

    
# region Visualization
def plotESR_FreqSweep(spinsys: SpinSys, FreqRange: list = [10, 20], AllowPumping: bool = False, N: int = 1000, lw: float = 0.02, norm: bool = True, output: bool = False):
    """
    Plot the ESR (Electron Spin Resonance) spectrum for a given spin system over a specified frequency range.

    This function computes the ESR spectrum using the provided spin system and plots the resulting signal.
    It also marks the resonance frequencies corresponding to recorded transitions in the spin system.

    Parameters
    ----------
    spinsys : SpinSys
        The spin system object containing resonance frequencies and recorded transitions.
    FreqRange : list of float, optional
        The frequency range [start, stop] in GHz over which to compute the ESR spectrum (default is [10, 20]).
    AllowPumping : bool, optional
        Whether to allow pumping effects in the ESR calculation (default is False).
    N : int, optional
        Number of frequency points in the sweep (default is 1000).
    lw : float, optional
        Linewidth for the ESR calculation (default is 0.02).
    norm : bool, optional
        Whether to normalize the ESR signal (default is True).

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

    Notes
    -----
    - The ESR spectrum is calculated using `calcESR_FreqSweep`.
    - Adds ResonanceFrequencies and RecordedTransitions to the spinsys object.
    """
    # calculate the ESR spectrum
    Freq_array, ESRsignal = calcESR_FreqSweep(spinsys, FreqRange=FreqRange, AllowPumping=AllowPumping, N=N, lw=lw, norm=norm, output=output)

    # Make the plot
    Nc = len(spinsys.ResonanceFrequencies)
    color_list = cm.jet(np.linspace(0, 1, Nc))
    color_list = [f'rgb({int(c[0] * 255)},{int(c[1] * 255)},{int(c[2] * 255)})' for c in color_list]

    # Create the main ESR signal trace
    fig = go.Figure()
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    fig.add_trace(go.Scatter(
        x=Freq_array,
        y=ESRsignal,
        mode='lines',
        line=dict(color='black', width=2),
        name='ESR Signal'
    ))

    # Mark resonances
    for i, freq_res in enumerate(spinsys.ResonanceFrequencies):
        index = np.argmin(np.abs(Freq_array - freq_res))
        amp = ESRsignal[index]
        if amp > 0.01 * np.max(ESRsignal):  # Filter very small peaks
            transition_str = (f"|{spinsys.RecordedTransitions[i][0]}> → |{spinsys.RecordedTransitions[i][1]}>; "
                             f"f = {np.round(freq_res, 4)} GHz")
            fig.add_trace(go.Scatter(
                x=[freq_res],
                y=[amp],
                mode='markers',
                marker=dict(color=color_list[i], size=12),
                name=transition_str
            ))

    fig.update_layout(
        xaxis_title='Freq (GHz)',
        yaxis_title='ΔI (a.u.)',
        title='ESR-Simulation',
        legend_title='Transitions',
        showlegend=True
    )
    fig.update_xaxes(range=[Freq_array[0], Freq_array[-1]])
    fig.show()

    return_dict = {
        'fig': fig,
        'Freq_array': Freq_array,
        'ESRsignal': ESRsignal}
    
    return return_dict


def plotESR_FreqSweep_ESRrate(spinsys: SpinSys, FreqRange=[10, 20], N: int = 200, norm: bool = True):
    """
    Compute and plot the ESR spectrum based on the ESR driving rate for a given spin system over a specified frequency range.

    This function computes the ESR spectrum based on an ESR driving rate for a given spin system over a specified frequency range
    and plots the resulting ESR signal. It also marks the resonance frequencies corresponding to allowed transitions in the spin
    system.

    Parameters
    ----------
    spinsys : SpinSys
        The spin system object containing energy levels and transition information.
    FreqRange : list of float, optional
        The frequency range [start, stop] in GHz over which to perform the sweep (default is [10, 20]).
    N : int, optional
        Number of frequency points in the sweep (default is 1000).
    norm : bool, optional
        Whether to normalize the ESR signal (default is True).

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
        
    Notes
    -----
    - The ESR spectrum is calculated using `calcESR_FreqSweep_ESRrate`.
    - Adds ResonanceFrequencies and RecordedTransitions to the spinsys object.
    """
    # Calculate the ESR spectrum using the specified parameters
    Freq_array, ESRsignal = calcESR_FreqSweep_ESRrate(spinsys, FreqRange=FreqRange, N=N, norm=norm)
    
    # Record the transitions
    spinsys.RecordedTransitions = calcRecordedTransitions(spinsys, Freq_array[0], Freq_array[-1])
    spinsys.ResonanceFrequencies = np.zeros(len(spinsys.RecordedTransitions))
    for i in range(len(spinsys.RecordedTransitions)):
        spinsys.ResonanceFrequencies[i] = abs(spinsys.E_All_inGHz[spinsys.RecordedTransitions[i][1]] - 
                                        spinsys.E_All_inGHz[spinsys.RecordedTransitions[i][0]])

    # Make the plot
    color_list = plotly.colors.qualitative.Plotly[:len(spinsys.ResonanceFrequencies)]

    # Create the main ESR signal trace
    fig = go.Figure()
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    fig.add_trace(go.Scatter(
        x=Freq_array,
        y=ESRsignal,
        mode='lines',
        line=dict(color='black', width=2),
        name='ESR Signal'
    ))

    # Mark resonances
    for i, freq_res in enumerate(spinsys.ResonanceFrequencies):
        index = np.argmin(np.abs(Freq_array - freq_res))
        amp = ESRsignal[index]
        if amp > 0.01 * np.max(ESRsignal):  # Filter very small peaks
            transition_str = (f"|{spinsys.RecordedTransitions[i][0]}> → |{spinsys.RecordedTransitions[i][1]}>; "
                             f"f = {np.round(freq_res, 4)} GHz")
            fig.add_trace(go.Scatter(
                x=[freq_res],
                y=[amp],
                mode='markers',
                marker=dict(color=color_list[i], size=12),
                name=transition_str
            ))

    fig.update_layout(
        xaxis_title='Freq (GHz)',
        yaxis_title='ΔI (a.u.)',
        title='ESR-Simulation',
        legend_title='Transitions',
        showlegend=True
    )
    fig.update_xaxes(range=[Freq_array[0], Freq_array[-1]])
    fig.show()

    return_dict = {
        'fig': fig,
        'Freq_array': Freq_array,
        'ESRsignal': ESRsignal}
    
    return return_dict


def plotESR_Tipfieldsweep(spinsys: SpinSys, FreqRange: list = [10, 20], N_Freq: int = 500, lw: float = 0.02, norm: bool = True, BTipRange: list = [-0.05, 0.05], N_B: int = 100, AllowPumping: bool = False, Angle: float = 0):  
    """
    Plot ESR spectra as a function of tip magnetic field for a given spin system.
    
    Parameters
    ----------
    spinsys : SpinSys
        The spin system object containing all relevant spin and field parameters.
    FreqRange : list of float, optional
        Frequency range [min, max] in GHz for the ESR spectrum (default is [10, 20]).
    N_Freq : int, optional
        Number of frequency points in the ESR spectrum (default is 500).
    lw : float, optional
        Linewidth for the ESR peaks (default is 0.02).
    norm : bool, optional
        Whether to normalize the ESR signal (default is True).
    BTipRange : list of float, optional
        Range [min, max] of tip magnetic field values in Tesla (default is [-0.05, 0.05]).
    N_B : int, optional
        Number of tip field values to compute (default is 100).
    AllowPumping : bool, optional
        Whether to allow pumping effects in the ESR calculation (default is False).
    Angle : float, optional
        Angle (in degrees) of the tip field with respect to the z-axis (default is 0).

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
        
    Notes
    -----
    - The initial tip magnetic field of the spin system is restored after the calculation.
    - Adds ResonanceFrequencies and RecordedTransitions to the spinsys object.
    - The ESR spectrum is calculated using `calcESR_FreqSweep`.
    """
    # Save previous BTip value
    BTip_initial = spinsys.BTip.copy()

    # Predefine Arrays
    BTip_array = np.linspace(BTipRange[0], BTipRange[1], N_B, dtype=float)
    Freq_array = np.linspace(FreqRange[0], FreqRange[1], N_Freq, dtype=float)

    ESRsignal = np.zeros((N_B, N_Freq), dtype=float)
    
    # calculating the ESR signal for each Field 
    for i in range(N_B):
        spinsys.BTip[2] = BTip_array[i] * np.cos(Angle * np.pi / 180)  # BTip[2] is the z-component of the tip field
        spinsys.BTip[0] = BTip_array[i] * np.sin(Angle * np.pi / 180) + 1e-6  # BTip[0] is the x-component of the tip field
        spinsys.BTip[1] = 0  # BTip[0] is the y-component of the tip field
        hamil.calcEigEnergies(spinsys)
        f, ESRsignal[i][:] = calcESR_FreqSweep(spinsys, norm=norm, N=N_Freq, lw=lw, FreqRange=FreqRange, AllowPumping=AllowPumping, output=False)
        
    # Calculating the Detuning from the Tipfield
    Detuning = spinsys.gfactorvector[spinsys.ReadoutSpin] * spinsys.muB * (BTip_array + spinsys.B[2]) * spinsys.meVtoGHzConversion - spinsys.gfactorvector[spinsys.ReadoutSpin] * spinsys.muB * spinsys.B[2] * spinsys.meVtoGHzConversion

    # Plot the ESR Signal as a color plot
    fig = go.Figure(data=go.Heatmap(
        z=ESRsignal,
        x=Freq_array,
        y=BTip_array * 1e3,  # Convert to mT for y-axis
        colorscale='Blues',
        colorbar=dict(title='ESR Signal'),
        zsmooth=False
    ))

    # Add labels and title
    fig.update_layout(
        height=600,
        width=800,
        xaxis_title='Freq (GHz)',
        yaxis_title='B_Tip (mT)',
        title=f'Tipfield dependent ESR (Angle = {Angle}°)'
    )

    # Add secondary y-axis for Detuning (GHz)
    # We'll add a custom axis on the right with detuning values
    # Calculate detuning for each BTip value
    detuning_labels = np.round(Detuning, 2)
    fig.update_layout(
        yaxis=dict(
            title='B_Tip (mT)',
            side='left'
        ),
        yaxis2=dict(
            title='Detuning (GHz)',
            overlaying='y',
            side='right',
            tickvals=BTip_array * 1e3,
            ticktext=detuning_labels
        )
    )

    fig.show()

    # Restore the initial BTip value
    spinsys.BTip = BTip_initial
    hamil.calcEigEnergies(spinsys)

    return_dict = {
        'fig': fig,
        'Freq_array': Freq_array,
        'BTip_array': BTip_array,
        'ESRsignal': ESRsignal}
    
    return return_dict


def plotESR_Populations(spinsys: SpinSys, freq: float = 14, filter: float = None):
    """
    Plot the populations of spin states for a given spin system under ESR conditions.

    This function calculates and visualizes the populations of spin states, including Boltzmann, tunneling, 
    and ESR-driven populations, for a provided spin system. The populations are displayed as horizontal bar plots.

    Parameters
    ----------
    spinsys : SpinSys
        The spin system object containing the relevant state and population information.
    freq : float, optional
        The frequency of the ESR field (in the same units as used in the simulation), by default 14.
    filter : float, optional
        If provided, only states with populations greater than this value in any of the three population arrays
        will be displayed. Default is None, which means all states are shown.

    Returns
    -------
    return_dict : dict
        A dictionary containing:
        - 'fig': plotly.graph_objects.Figure
            The plotly figure object containing the ESR spectrum plot.
        - 'Pop_Boltzmann': numpy.ndarray
            The Boltzmann populations of the spin states.
        - 'Pop_Tunneling': numpy.ndarray
            The steady state populations under the influence of the tunneling rates of the spin states.
        - 'Pop_ESR': numpy.ndarray
            The ESR-driven populations of the spin states.

    Notes
    -----
    - The function assumes that the `spinsys` object has attributes `pop_0`, `Populations`, `Populations_ESR`, and `basisStates`.
    - If the eigenstates are not already calculated, the function will compute them.
    - Adds Populations_ESR to the spinsys object.
    """
    calcPopulations_ESR(spinsys, freq=freq)
    if not hasattr(spinsys, 'basisSates') or spinsys.basisStates is None:
        hamil.calcEigStates(spinsys)
    
    width = 0.2

    if filter is None:
        # Plot all states with plotly
        labels = spinsys.statesWithoutE
        positions = np.arange(len(spinsys.Populations))
        pop0 = np.array(spinsys.pop_0)
        popT = np.array(spinsys.Populations)
        popESR = np.array(spinsys.Populations_ESR)

        fig = go.Figure()
        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        fig.add_trace(go.Bar(
            x=pop0,
            y=labels,
            orientation='h',
            name='Boltzmann Populations',
            marker=dict(color='blue'),
            opacity=0.5
        ))
        fig.add_trace(go.Bar(
            x=popT,
            y=labels,
            orientation='h',
            name='Tunneling Populations',
            marker=dict(color='red'),
            opacity=0.5
        ))
        fig.add_trace(go.Bar(
            x=popESR,
            y=labels,
            orientation='h',
            name='ESR Populations',
            marker=dict(color='green'),
            opacity=0.5
        ))

        fig.update_layout(
                barmode='group',
                xaxis_title='Population',
                yaxis=dict(
                    title='Basis States',
                    tickmode='array',
                    tickvals=positions,
                    ticktext=labels
                ),
                title='Population of Spin States',
                legend=dict(orientation='h', yanchor='bottom', y=1.02,
                             xanchor='right', x=1)
        )

        fig.show()
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
            fig.update_layout(
                plot_bgcolor='white',
                paper_bgcolor='white'
            )
            fig.add_trace(go.Bar(
                x=pop0,
                y=positions,
                name='Boltzmann Populations',
                orientation='h',
                marker=dict(color='blue'),
                opacity=0.5,
                width=width
            ))
            fig.add_trace(go.Bar(
                x=popT,
                y=positions,
                name='Tunneling Populations',
                orientation='h',
                marker=dict(color='red'),
                opacity=0.5,
                width=width
            ))
            fig.add_trace(go.Bar(
                x=popESR,
                y=positions,
                name='ESR Populations',
                orientation='h',
                marker=dict(color='green'),
                opacity=0.5,
                width=width
            ))

            fig.update_layout(
                barmode='group',
                xaxis_title='Population',
                yaxis=dict(
                    title='Basis States',
                    tickmode='array',
                    tickvals=positions,
                    ticktext=labels
                ),
                title=f'Population of Spin States (> {filter * 100}%)',
                legend=dict(orientation='h', yanchor='bottom', y=1.02,
                             xanchor='right', x=1)
            )

            fig.show()

    return_dict = {
        'fig': fig,
        'Pop_Boltzmann': spinsys.pop_0,
        'Pop_Tunneling': spinsys.Populations,
        'Pop_ESR': spinsys.Populations_ESR}
    
    return return_dict

# endregion Visualization
    
import numpy as np
import scipy as sp
import scipy.constants as const
from scipy.linalg import null_space

from spinsimulation.SpinSys import SpinSys
import spinsimulation.SpinHamiltonian as hamil 
import spinsimulation.calcBath as bath


def calcESR_FreqSweep_Benjamin(spinsys: SpinSys, FreqRange: list = [10, 20], AllowPumping: bool = False, N: int = 1000, lw: float = 0.02, norm: bool = True, output: bool = True):
    """
    Calculates the Electron Spin Resonance (ESR) spectrum for a given spin system.

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
    spinsys.ESRshift = 0 # Dont remember what thuis is for
    freq = np.linspace(FreqRange[0], FreqRange[1], N)  # Frequency array
    ESRsignal = np.zeros(len(freq))

    # Lorentzian function equivalent
    fitlorentz = lambda para, x1: (1 / (para[3]**2 + 1) * abs(para[1]) * 
                                    (1 + (x1 - para[0]) / (0.5 * para[2]) * para[3])**2 / 
                                    (1 + ((x1 - para[0]) / (0.5 * para[2]))**2))


    # Calculate the Eignevalues and Eignevectors if not already done
    if not hasattr(spinsys, 'eigVectors') or spinsys.eigVectors is None:
        hamil.calcEigEnergies(spinsys)

    bath.calcPopulations(spinsys, AllowPumping=AllowPumping, Approach="Loth")

    # Look at transitions
    spinsys.RecordedTransitions = calcRecordedTransitions(spinsys,freq[0],freq[-1])
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
                    f"f = {np.round(spinsys.ResonanceFrequencies[i],4)} GHz")
        else:
            print(f"There were {len(spinsys.RecordedTransitions)} potential transitions in the specified frequency range")

    # Amplitude calculation
    for k in range(len(spinsys.ResonanceFrequencies)):
        i = spinsys.RecordedTransitions[k][0]  # Initial state index
        j = spinsys.RecordedTransitions[k][1]  # Final state index

        spinsys.dp = abs(spinsys.Populations[i] - spinsys.Populations[j])

        # Rabi rate factor
        rabiRateFactor = 1

        spinsys.M_x = rabiRateFactor * (spinsys.eigVectors[:, j].T.conj() @ spinsys.Sx[spinsys.ReadoutSpin,:, :] @ spinsys.eigVectors[:, i])
        spinsys.M_y = rabiRateFactor * (spinsys.eigVectors[:, j].T.conj() @ spinsys.Sy[spinsys.ReadoutSpin,:, :] @ spinsys.eigVectors[:, i])
        spinsys.M_z = rabiRateFactor * (spinsys.eigVectors[:, j].T.conj() @ spinsys.Sz[spinsys.ReadoutSpin,:, :] @ spinsys.eigVectors[:, i])

        spinsys.Amp = spinsys.dp * (spinsys.BTip[0] * spinsys.M_x + spinsys.BTip[1] * spinsys.M_y + spinsys.BTip[2] * spinsys.M_z)**2

        ESRsignal += fitlorentz([spinsys.ResonanceFrequencies[k], spinsys.Amp, lw, 0], freq)

    # Normalization
    if norm == 1:
        ESRsignal = (ESRsignal - ESRsignal.min()) / (ESRsignal.max() - ESRsignal.min())


    return freq, ESRsignal

def calcRecordedTransitions(spinsys: SpinSys, MinE: float, MaxE: float):
    """
    Calculates and records all transitions between quantum states within a given energy range. 
    
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
    Simulates a continuous-wave ESR (Electron Spin Resonance) frequency sweep for a given spin system.
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

    #if not hasattr(spinsys, 'Populations') or spinsys.Populations is None:
    bath.calcPopulations(spinsys, AllowPumping = True) # Here we do not care about the approach yet
    
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
                    spinsys.ESRRates[i, j] = 0.5 * spinsys.RabiRate**2 * abs(TransitionMatrixElement[i, j])**2 * spinsys.T2 / ((freq[freq_index]-abs(spinsys.E_All_inGHz[i] - spinsys.E_All_inGHz[j]))**2 * spinsys.T2**2 + 1)

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
    Adds the ESR (Electron Spin Resonance) transition rates to the given spin system.
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
                spinsys.ESRRates[i, j] = 0.5 * spinsys.RabiRate**2 * abs(TransitionMatrixElement[i, j])**2 * spinsys.T2 / ((freq-abs(spinsys.E_All_inGHz[i] - spinsys.E_All_inGHz[j]))**2 * spinsys.T2**2 + 1)
    

    
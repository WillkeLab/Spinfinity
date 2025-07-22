import numpy as np
import scipy as sp
import scipy.constants as const
from scipy.linalg import null_space

from spinsimulation.SpinSys import SpinSys
import spinsimulation.SpinHamiltonian as hamil 
import spinsimulation.calcBath as bath


def calcESR_Benjamin(spinsys: SpinSys, FreqRange: list = [10, 20], AllowPumping: bool = False, N: int = 1000, lw: float = 0.02, norm: bool = True, output: bool = True):
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

def calcRecordedTransitions(spinsys: SpinSys,MinE: float,MaxE: float):
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

#----------------------------------------------------------------------------------------------------------
## Bath

import numpy as np
import scipy as sp
import scipy.constants as const
from scipy.linalg import null_space
import plotly.graph_objects as go
import plotly.colors
from plotly.subplots import make_subplots

from spinfinity.SpinSys import SpinSys
import spinfinity.SpinHamiltonian as hamil 

# region Physical Quantities

def calcMagnetization(spinsys: SpinSys, type: str='z', AllowPumping: bool = True):
    """
    Calculate the magnetization of a spin system along a specified axis.

    Parameters
    ----------
    spinsys : SpinSys
        The spin system object containing eigenvectors, spin operators, and populations.
    type : str, optional
        The axis along which to calculate magnetization ('x', 'y', or 'z'). Default is 'z'.
    AllowPumping : bool, optional
        Whether to allow population pumping during calculation. Default is True.
    Approach : str, optional
        The approach used for population calculation. Default is "Loth".

    Raises
    ------
    ValueError
        If `type` is not one of 'x', 'y', or 'z'.

    Notes
    -----
    The function updates the following attributes of `spinsys`:
        - Mag_tot: Total magnetization of the system.
        - Mag_tot_max: Maximum possible total magnetization.
        - Mag: Magnetization for each individual spin.
        - Mag_max: Maximum possible magnetization for each spin.
        - Magtype: The axis along which magnetization was calculated.

    Returns
    -------
    None
        The results are stored in the `spinsys` object.
    """
    if not hasattr(spinsys, 'Populations') or spinsys.Populations is None:
        calcPopulations(spinsys, AllowPumping=True, Approach=Approach)
    
    if AllowPumping:
        Populations=spinsys.Populations
    else:
        Populations=spinsys.pop_0

    # Choose the Operator based on the type of magnetization
    if type == 'z':
        S = spinsys.Sz
    elif type == 'y':
        S = spinsys.Sy
    elif type == 'x':
        S = spinsys.Sx
    else:
        raise ValueError("Invalid type for magnetization. Choose 'x', 'y', or 'z'.")
    
    S_Operator = np.sum(S, axis=0)
    
    # Initialize the magnetization arrays
    S_elements = np.zeros((spinsys.NSpins, spinsys.dimensionOfMatrix)) #Individual projection 
    S_elements_tot = np.zeros(spinsys.dimensionOfMatrix) # Projection of the total spin system
    Mag = np.zeros((spinsys.NSpins))
    Mag_tot = 0

    # Itterate through every eigenstate and each spin 
    for i in range(spinsys.dimensionOfMatrix):
            S_elements_tot[i] = np.real(spinsys.eigVectors[:, i].T.conj() @ S_Operator @ spinsys.eigVectors[:, i])
            Mag_tot += S_elements_tot[i] * Populations[i]
            for j in range(spinsys.NSpins):
                S_elements[j, i] = np.real(spinsys.eigVectors[:, i].T.conj() @ S[j, :, :] @ spinsys.eigVectors[:, i])
                Mag[j] += S_elements[j,i] * Populations[i]
    
    # Remember the maximum value for each spin to have the opportunity to normalize
    Mag_tot_max = S_elements_tot.max()
    Mag_max = S_elements.max(axis=1)

    # Store the magnetization in the spinsys object
    spinsys.Mag_tot = Mag_tot  # Store the total magnetization in the spinsys object
    spinsys.Mag_tot_max = Mag_tot_max  # Store the maximum magnetization in the spinsys object
    spinsys.Mag = Mag  # Store the individual magnetizations in the spinsys object
    spinsys.Mag_max = Mag_max  # Store the maximum magnetization for each spin in
    spinsys.Magtype = type  # Store the type of magnetization calculated

    return Mag

def calcLifetimes(spinsys: SpinSys, AllowPumping: bool = True, Approach: str = "Loth"):
    """
    Calculate the lifetimes (T1) for each state in a spin system.

    This function computes the total and natural lifetimes for each state in the given spin system
    by summing the transition rates out of each state. If the rate matrix is not already calculated,
    it will be computed using the specified approach.

    Parameters
    ----------
    spinsys : SpinSys
        The spin system object containing state information and rate matrices.
    AllowPumping : bool, optional
        Whether to allow pumping effects in the calculation (default is True).
    Approach : str, optional
        The approach used to calculate the rate matrix if not already present (default is "Loth").

    Returns
    -------
    None
        The results are stored as attributes (`T1_tot` and `T1_ss`) in the `spinsys` object.

    Notes
    -----
    - `T1_tot` is the total lifetime for each state, considering all possible transitions.
    - `T1_ss` is the natural lifetime for each state, considering only sample scattering transitions.
    - Requires the `Rates` and `Rates_Summed` attributes in the `spinsys` object.
    """
    # Check whether the RateMatrix is already calculated
    if not hasattr(spinsys, 'Rates') or spinsys.Rates is None:
        calcRates(spinsys, Approach=Approach)

    # Initialize the Rate array that somes up all rates that leave the state
    R1_tot = np.zeros(spinsys.dimensionOfMatrix)
    R1_ss = np.zeros(spinsys.dimensionOfMatrix)

    # Summing up all rates that leave the state
    for i in range(spinsys.dimensionOfMatrix): # i is the initial state
        for j in range(spinsys.dimensionOfMatrix): # j is the final state
            if i != j:
                R1_tot[i] += spinsys.Rates_Summed[i, j]  # Summing up all the rates
                R1_ss[i] = spinsys.Rates[2, i, j] # Just sample sample scattering

    spinsys.T1_tot = 1 / R1_tot  # Total lifetime
    spinsys.T1_ss = 1 / R1_ss  # natural lifetime

    return

def calcSplitLifetimes(spinsys: SpinSys, subspin: SpinSys):
    """
    Compute lifetimes for subsystem eigenstates.
    
    Parameters
    ----------
    spinsys : SpinSys
        Full system object.
    subspin : SpinSys
        Subsystem object defining the eigenstates onto which rates are projected.
    
    Returns
    -------
    None
        The function mutates `spinsys` in-place. After the call the following
        attributes on `spinsys` will be set to 1D numpy arrays of length
        subspin.dimensionOfMatrix:
        - T1_A_tot : total lifetimes for overlap type A
        - T1_B_tot : total lifetimes for overlap type B
        - T1_A_ss  : sample-sample lifetimes for overlap type A
        - T1_B_ss  : sample-sample lifetimes for overlap type B
    
    Notes
    -----
    - The function calls calcRates(spinsys) at the start to ensure rate arrays are
      populated.
    """
    # Calculate the Rates of the SpinSystem
    calcRates(spinsys)
    # Initialize the Rate arrays
    R_A_tot = np.zeros(subspin.dimensionOfMatrix)
    R_B_tot = np.zeros(subspin.dimensionOfMatrix)
    R_A_ss = np.zeros(subspin.dimensionOfMatrix)
    R_B_ss = np.zeros(subspin.dimensionOfMatrix)

    # Iterate over the states of the subspin system
    for idx in range(subspin.dimensionOfMatrix):
        overlap_A = hamil.calcOverlap(spinsys, subspin.dimensionOfMatrix, subspin.dimensionOfMatrix, subspin.eigVectors[:, idx], order='BA')
        overlap_B = hamil.calcOverlap(spinsys, subspin.dimensionOfMatrix, subspin.dimensionOfMatrix, subspin.eigVectors[:, idx], order='AB')
        # Compute the rates projected onto the overlaps
        for i in range(spinsys.dimensionOfMatrix): # i is the initial state
            for j in range(spinsys.dimensionOfMatrix): # j is the final state
                if i != j:
                    R_A_tot[idx] += overlap_A[i] * spinsys.Rates_Summed[i, j] * (1-overlap_A[j])  # Summing up all the rates
                    R_B_tot[idx] += overlap_B[i] * spinsys.Rates_Summed[i, j] * (1-overlap_B[j])  # Summing up all the rates
                    R_A_ss[idx] += overlap_A[i] * spinsys.Rates[2, i, j] * (1-overlap_A[j])  # Summing up sample-sample rates
                    R_B_ss[idx] += overlap_B[i] * spinsys.Rates[2, i, j] * (1-overlap_B[j])  # Summing up sample-sample rates

    spinsys.T1_A_tot = 1 / R_A_tot  # Total lifetime
    spinsys.T1_B_tot = 1 / R_B_tot  # Total lifetime
    spinsys.T1_A_ss = 1 / R_A_ss  # Substrate lifetime
    spinsys.T1_B_ss = 1 / R_B_ss  # Substrate lifetime

    return

def calcTunnelCurrent(spinsys: SpinSys, AllowPumping: bool = True, Approach: str = "Loth"):
    """
    Calculate the tunneling current for the given spin system.

    Parameters
    ----------
    spinsys : SpinSys
        The spin system object containing energy levels, populations, rates, and other relevant parameters.
    AllowPumping : bool, optional
        If True, allows for non-thermal population effects (default is True).
    Approach : str, optional
        The approach used for calculating rates and current, e.g., "Loth" or "Ternes" (default is "Loth").

    Returns
    -------
    I : float
        The calculated tunneling current.

    Notes
    -----
    - The function updates the `I` attribute of the `spinsys` object in-place.
    - The current is calculated using the populations and transition rates between spin states.
    - If populations are not available, they are calculated using `calcPopulations`.
    - The calculation method depends on the specified approach.
    """
    if not hasattr(spinsys, 'Populations') or spinsys.Populations is None:
            calcPopulations(spinsys, AllowPumping=AllowPumping, Approach=Approach)

    # Deriving the tunneling current using Equation 41 from Ternes paper
    spinsys.I = 0
    for i in range(spinsys.dimensionOfMatrix):
        for j in range(spinsys.dimensionOfMatrix):
            I_temp = const.e * spinsys.Populations[i] * (spinsys.Rates[0, i, j] - spinsys.Rates[1, i, j])
            spinsys.I += I_temp

    # Adjust current calculation based on approach
    if Approach == "Ternes":
        spinsys.I = spinsys.I / (const.e**2 * spinsys.T02 / spinsys.h_meV)
    if Approach == "Loth":
        spinsys.I += spinsys.b0 * spinsys.G * spinsys.V_DC

    return spinsys.I

def calcPopulations(spinsys: SpinSys, AllowPumping:bool = True, Approach: str = "Loth"):
    """
    Calculate the population distribution of spin states in a given spin system.

    Computes the thermal population distribution based on Boltzmann statistics.
    If pumping is allowed, calculates the non-thermal population by solving the steady-state
    of the rate equations using the specified approach.

    Parameters
    ----------
    spinsys : SpinSys
        The spin system object containing energy levels, temperature, and other relevant parameters.
    AllowPumping : bool, optional
        If True, calculates the non-thermal population considering pumping effects (default is True).
    Approach : str, optional
        The approach used for calculating rates, e.g., "Loth" (default is "Loth").

    Returns
    -------
    None
        The function updates the `Populations` attribute of the `spinsys` object in-place.

    Notes
    -----
    - If pumping is not allowed or no valid solution is found for the non-thermal population,
        the thermal population is used.
    - The function assumes that the `spinsys` object has the necessary attributes and methods
        for rate calculations.
    """
    if not hasattr(spinsys, 'E_All') or spinsys.E_All is None:
        hamil.calcEigEnergies(spinsys)
    
    # Calculate the thermal population in any case
    p0 = np.exp(-spinsys.E_All / (spinsys.kB * spinsys.T))
    pTot = np.sum(p0)
    spinsys.pop_0 = p0 / pTot  # Normalized thermal population

    # If pumping is allowed, calculate the non-thermal population
    if AllowPumping:
        # Calculate the rates if not already done
        if not hasattr(spinsys, 'Rates') or spinsys.Rates is None:
            calcRates(spinsys, Approach=Approach)
        spinsys.K = np.zeros((spinsys.dimensionOfMatrix, spinsys.dimensionOfMatrix))

        for i in range(spinsys.dimensionOfMatrix):
            for j in range(spinsys.dimensionOfMatrix):
                if i != j:
                    spinsys.K[i, i] -= spinsys.Rates_Summed[i, j]  # Rates leaving state i
                    spinsys.K[i, j] = spinsys.Rates_Summed[j, i]  # Rates leading to state i
        # Solve the Rate equation to get the steady-state populations
        if AllowPumping:
            try:
                p = null_space(spinsys.K)
            except Exception as e:
                print("Warning: null_space failed, using thermal population. Error:", e)
                p = np.empty((0, 0))
        if p.size == 0:  # If no valid solution, use thermal distribution
            spinsys.Populations = spinsys.pop_0
        else:
            spinsys.Populations = p[:, 0] / np.sum(p[:, 0])
    else: 
        spinsys.Populations = spinsys.pop_0

    return 

# endregion Physical Quantities

# region Rate Calculations

def calcRates(spinsys: SpinSys, Approach: str = "Loth"):
    """
    Calculate transition rates for a given spin system.

    Parameters
    ----------
    spinsys : SpinSys
        The spin system object containing all relevant parameters and matrices.
    Approach : str, optional
        The approach used for rate calculation. Can be "Loth" (default) or "Ternes".

    Returns
    -------
    None
        The function updates the following attributes of `spinsys` in-place:
        - spinsys.Rates : ndarray
            A 3D array containing the calculated rates for different processes.
        - spinsys.Rates_Summed : ndarray
            A 2D array containing the summed rates over all processes.

    Notes
    -----
    - The function initializes and calculates various tunneling matrix elements and rate integrals.
    - The rate factors are computed based on the selected approach.
    - The rates are calculated for different tunneling and scattering processes and stored in `spinsys.Rates`.
    - The summed rates are stored in `spinsys.Rates_Summed`.
    - The function assumes that required attributes and methods (such as `calcEigEnergies`, `calcTunnelingMatrixElements`, and `calcRateIntegrals`) are available in the provided `spinsys` object.
    """
    # Initialize the Rates matrix
    spinsys.Rates = np.zeros((4,spinsys.dimensionOfMatrix, spinsys.dimensionOfMatrix))

     # Calculate the Eignevalues and Eignevectors if not already done
    if not hasattr(spinsys, 'eigVectors') or spinsys.eigVectors is None:
        hamil.calcEigEnergies(spinsys)

    # Getting the relevant matrix elements
    spinsys.MatrixST = calcTunnelingMatrixElements(spinsys,"st")[0]
    spinsys.MatrixTS = spinsys.MatrixST.T
    spinsys.MatrixSS = calcTunnelingMatrixElements(spinsys,"ss")
    spinsys.MatrixTT = calcTunnelingMatrixElements(spinsys,"tt")[0]
    
    MatrixSS_summed = np.zeros((spinsys.dimensionOfMatrix, spinsys.dimensionOfMatrix))

    for i in range(spinsys.NSpins):
        MatrixSS_summed += spinsys.MatrixSS[i, :, :] * spinsys.G_ss[i]

    # Calculating the 3 different integrals
    RateIntegrals_ts = calcRateIntegrals(spinsys, 'ts')
    RateIntegrals_st = calcRateIntegrals(spinsys, 'st')
    RateIntegrals_ss = calcRateIntegrals(spinsys, 'ss')
    #RateIntegrals_tt = spinsys.calcRateIntegrals(spinsys.V_DC, 'tt')

    Rate_0 = spinsys.MatrixST[0][0]  # Transition matrix element of the ground state

    ### Rate-Factors using Loth 2010
    if Approach == "Loth":
        spinsys.G_st = (1 - spinsys.b0) * spinsys.G

        RateFactor_ts = spinsys.G_st / (const.e * Rate_0) / 1e3
        RateFactor_st = spinsys.G_st / (const.e * Rate_0) / 1e3
        # The G_ss value might be different for every spin thats why we previously incorpareted it when we summed the Matrix
        RateFactor_ss = 1 / (const.e * Rate_0) / 1e3
        RateFactor_tt = spinsys.G_st**2/spinsys.G_ss[spinsys.ReadoutSpin]/(const.e*Rate_0)/1e3*spinsys.G_tt; #Gtt is actually calculated from Gss and Gst, we use the Gtt value 0,1 here just o allow it or not

    ### Rate-Factors using Ternes 2010 (not default)
    if Approach == "Ternes":
        RateFactor_ts = 2 * np.pi / spinsys.hbar_meV * spinsys.T02
        RateFactor_st = 2 * np.pi / spinsys.hbar_meV * spinsys.T02
        RateFactor_ss = 2 * np.pi / spinsys.hbar_meV * spinsys.Jrs**2
        RateFactor_tt = 2 * np.pi / spinsys.hbar_meV * spinsys.T02**2 / (spinsys.Jrs**2) * spinsys.G_tt

    # Calculate rates based on rate factors and integrals
    spinsys.Rates[0, :, :] = RateFactor_ts * spinsys.MatrixTS *  RateIntegrals_ts
    spinsys.Rates[1, :, :] = RateFactor_st * spinsys.MatrixST *  RateIntegrals_st
    spinsys.Rates[2, :, :] = RateFactor_ss * MatrixSS_summed *  RateIntegrals_ss
    spinsys.Rates[3, :, :] = RateFactor_tt * spinsys.MatrixTT *  RateIntegrals_ss
    
    # Sum the rates
    spinsys.Rates_Summed = (spinsys.Rates[0,:, :] + spinsys.Rates[1, :, :] + 
                        spinsys.Rates[2, :, :] + spinsys.Rates[3, :, :])

    return 

def calcRateIntegrals(spinsys: SpinSys, type: str):
    """
    Calculate rate integrals for a given spin system and bias voltage.
    
    Parameters
    ----------
    spinsys : SpinSys
        An instance of the SpinSys class containing system parameters such as energy levels,
        Boltzmann constant, temperature, and matrix dimension.
    type : str
        The type of transition to calculate. Must be one of:
        - 'ts': tip to sample
        - 'st': sample to tip
        - 'ss': sample to sample
        - 'tt': tip to tip

    Returns
    -------
    RateIntegrals : np.ndarray
        A 2D numpy array of shape (dimensionOfMatrix, dimensionOfMatrix) containing the
        calculated rate integrals for each pair of energy levels.

    Notes
    -----
    - The function avoids division by zero by substituting a small value for V_DC if it is zero.
    """
    # Create the three nxn matrices [I_ts, I_st, I_ss]
    RateIntegrals = np.zeros((spinsys.dimensionOfMatrix, spinsys.dimensionOfMatrix))

    # Check if V_DC is zero, if so, set it to a small value to avoid division by zero
    V_DC = spinsys.V_DC
    if V_DC == 0:
        V_DC = 1e-9

    for i in range(spinsys.dimensionOfMatrix):
        for j in range(spinsys.dimensionOfMatrix):
            delta_E = spinsys.E_All[j] - spinsys.E_All[i]
            if type == 'ts':
                RateIntegrals[i, j] = abs((delta_E - V_DC) / (np.exp((delta_E - V_DC) / (spinsys.kB * spinsys.T)) - 1))
            elif type == 'st':
                RateIntegrals[i, j] = abs((delta_E + V_DC) / (np.exp((delta_E + V_DC) / (spinsys.kB * spinsys.T)) - 1))
            elif type == 'ss':
                RateIntegrals[i, j] = abs(delta_E / (np.exp(delta_E / (spinsys.kB * spinsys.T)) - 1))
            elif type == 'tt':
                RateIntegrals[i, j] = abs(delta_E / (np.exp(delta_E / (spinsys.kB * spinsys.T)) - 1))

    return RateIntegrals

def calcTunnelingMatrixElements(spinsys: SpinSys, scattertype: str):
    """
    Calculate the tunneling matrix elements for a given spin system and scattering type.

    Parameters
    ----------
    spinsys : SpinSys
        An instance of the SpinSys class containing the spin system properties, eigenvectors,
        and spin operators (Sx, Sy, Sz, U).
    scattertype : str
        Type of scattering process. If "ss", calculations are performed for all spins;
        otherwise, calculations are performed for the readout spin only.

    Returns
    -------
    RateMatrix : np.ndarray
        Array containing the calculated tunneling rate matrix elements. The shape is
        (NSpins, dimensionOfMatrix, dimensionOfMatrix) for "ss" scattertype, and
        (1, dimensionOfMatrix, dimensionOfMatrix) otherwise.
        
    Notes
    -----
    The function computes the tunneling matrix elements by projecting spin operators
    into the eigenbasis and combining them with electron tunneling matrix elements.
    The resulting rate matrix is used in further calculations of tunneling rates.
    """
    # Get the electron tunneling matrix elements
    x_el, y_el, z_el, u_el = calcTunnelingElectronMatrixElements(spinsys, scattertype)

    # Initialize RateMatrix
    if scattertype == "ss":
        RateMatrix = np.zeros((spinsys.NSpins, spinsys.dimensionOfMatrix, spinsys.dimensionOfMatrix))
        N = spinsys.NSpins
        Index = np.arange(N)
    else:
        RateMatrix = np.zeros((1, spinsys.dimensionOfMatrix, spinsys.dimensionOfMatrix)) 
        N = 1
        Index = np.array([spinsys.ReadoutSpin])

    # x_el, y_el, z_el, and u_el are assumed to be 2x2 matrices (flattened to 4 elements)
    for k in range(N):
        MatrixX = np.conjugate(np.transpose(spinsys.eigVectors)) @ spinsys.Sx[Index[k], :, :] @ spinsys.eigVectors
        MatrixY = np.conjugate(np.transpose(spinsys.eigVectors)) @ spinsys.Sy[Index[k], :, :] @ spinsys.eigVectors
        MatrixZ = np.conjugate(np.transpose(spinsys.eigVectors)) @ spinsys.Sz[Index[k], :, :] @ spinsys.eigVectors
        MatrixU = spinsys.U[Index[k]] * np.conjugate(np.transpose(spinsys.eigVectors)) @ spinsys.eigVectors

        for i in range(2):
            for j in range(2):
                RateMatrix[k][:][:] += np.abs(MatrixX * x_el[i][j] + MatrixY * y_el[i][j] + MatrixZ * z_el[i][j] + MatrixU * u_el[i][j]) ** 2

    return RateMatrix

def calcTunnelingElectronMatrixElements(spinsys: SpinSys, scattertype: str):
    """
    Calculate tunneling electron matrix elements for a given spin system and scattering type.

    This function computes the matrix elements associated with tunneling electrons between tip and sample,
    considering their spin polarizations. It normalizes the tip and sample polarizations if necessary,
    constructs the corresponding density matrices, and calculates the matrix elements for the specified
    scattering type ('st', 'tt', or 'ss').

    Parameters
    ----------
    spinsys : SpinSys
        An object representing the spin system, containing tip and sample polarizations.
    scattertype : str
        The type of scattering to calculate matrix elements for. Options are:
        - 'st': tip-to-sample scattering
        - 'tt': tip-to-tip scattering
        - 'ss': sample-to-sample scattering

    Returns
    -------
    x : ndarray
        Matrix elements for the x-component of spin.
    y : ndarray
        Matrix elements for the y-component of spin.
    z : ndarray
        Matrix elements for the z-component of spin.
    u : ndarray
        Matrix elements for the identity component.

    Notes
    -----
    If an invalid `scattertype` is provided, the function defaults to 'st' (tip-to-sample scattering).
    The function uses Pauli matrices for spin-1/2 electrons and computes the matrix elements in the eigenbasis
    of the tip and sample density matrices.
    """
    # Normalize the Tip and Sample Polarizations if their norms are greater than 1
    if np.linalg.norm(spinsys.TipPolarization) > 1:
        spinsys.TipPolarization = spinsys.TipPolarization / np.linalg.norm(spinsys.TipPolarization)
    if np.linalg.norm(spinsys.SamplePolarization) > 1:
        spinsys.SamplePolarization = spinsys.SamplePolarization / np.linalg.norm(spinsys.SamplePolarization)

    # Get Spin-1/2 matrices (Pauli matrices) for electrons
    S_x, S_y, S_z = hamil.calcAngMomMatrices(0.5)

    # Calculate the density matrices for tip and sample
    densityTip = 0.5 * np.eye(2) + spinsys.TipPolarization[0] * S_x + spinsys.TipPolarization[1] * S_y + spinsys.TipPolarization[2] * S_z
    densitySample = 0.5 * np.eye(2) + spinsys.SamplePolarization[0] * S_x + spinsys.SamplePolarization[1] * S_y + spinsys.SamplePolarization[2] * S_z

    # Calculate the eigenvalues and eigenvectors of the density matrices
    eigValTip_temp, eigVecTip = np.linalg.eig(densityTip)
    eigValSample_temp, eigVecSample = np.linalg.eig(densitySample)
    eigValTip = np.diag(eigValTip_temp)
    eigValSample = np.diag(eigValSample_temp)

    # Swap the eigen value order
    eigValTip[[0, 1], [0, 1]] = eigValTip[[1, 0], [1, 0]]
    eigVecTip[:, [0, 1]] = eigVecTip[:, [1, 0]]

    # Calculate the weights matrix as sqrt(diag(eigValSample) * diag(eigValTip))
    Weights_st = np.sqrt(np.outer(np.diag(eigValSample), np.diag(eigValTip)))
    Weights_tt = np.sqrt(np.outer(np.diag(eigValTip), np.diag(eigValTip)))
    Weights_ss = np.sqrt(np.outer(np.diag(eigValSample), np.diag(eigValSample)))

    x_st = np.multiply((np.conjugate(eigVecSample.T) @ (2 * S_x) @ eigVecTip), Weights_st)
    y_st = np.multiply((np.conjugate(eigVecSample.T) @ (2 * S_y) @ eigVecTip), Weights_st)
    z_st = np.multiply((np.conjugate(eigVecSample.T) @ (2 * S_z) @ eigVecTip), Weights_st)
    u_st = np.multiply((2 * np.conjugate(eigVecSample.T) @ eigVecTip), Weights_st)

    x_tt = np.multiply((np.conjugate(eigVecTip.T) @ (2 * S_x) @ eigVecTip), Weights_tt)
    y_tt = np.multiply((np.conjugate(eigVecTip.T) @ (2 * S_y) @ eigVecTip), Weights_tt)
    z_tt = np.multiply((np.conjugate(eigVecTip.T) @ (2 * S_z) @ eigVecTip), Weights_tt)
    u_tt = np.multiply((2 * np.conjugate(eigVecTip.T) @ eigVecTip), Weights_tt)

    x_ss = np.multiply((np.conjugate(eigVecSample.T) @ (2 * S_x) @ eigVecSample), Weights_ss)
    y_ss = np.multiply((np.conjugate(eigVecSample.T) @ (2 * S_y) @ eigVecSample), Weights_ss)
    z_ss = np.multiply((np.conjugate(eigVecSample.T) @ (2 * S_z) @ eigVecSample), Weights_ss)
    u_ss = np.multiply((2 * np.conjugate(eigVecSample.T) @ eigVecSample), Weights_ss)

    if scattertype == "st":
        return x_st, y_st, z_st, u_st
    elif scattertype == "tt":
        return x_tt, y_tt, z_tt, u_tt
    elif scattertype == "ss":
        return x_ss, y_ss, z_ss, u_ss
    else:
        return x_st, y_st, z_st, u_st

# endregion Rate Calculations

# region Visualization
def plotPopulations(spinsys: SpinSys, AllowPumping: bool = True, Approach: str = "Loth"):
    """
    Plot the thermal populations of the spin system's states.

    Parameters
    ----------
    spinsys : SpinSys
        The spin system object containing the necessary parameters and methods.
    AllowPumping : bool, optional
        Whether to allow pumping effects in the calculation (default is True).
    Approach : str, optional
        The approach to use for calculating populations (default is "Loth").

    Returns
    -------
    return_dict : dict
        A dictionary containing:
        - 'fig': plotly.graph_objects.Figure
            The plotly figure object containing the ESR spectrum plot.
    
    Notes
    -----
    - The function assumes that the `spinsys` object has the necessary attributes and methods for rate calculations.
    """
    if not hasattr(spinsys, 'Populations') or spinsys.Populations is None:
            calcPopulations(spinsys,AllowPumping=AllowPumping, Approach=Approach)
    if not hasattr(spinsys, 'basisSates') or spinsys.basisStates is None:
            hamil.calcEigStates(spinsys)
    
    x = np.arange(len(spinsys.Populations))
    width = 0.35

    # Plot with Plotly (white background)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=spinsys.pop_0,
        y=spinsys.statesWithoutE,
        orientation='h',
        name='Boltzmann Populations',
        marker=dict(color='blue'),
        opacity=0.5
    ))
    fig.add_trace(go.Bar(
        x=spinsys.Populations,
        y=spinsys.statesWithoutE,
        orientation='h',
        name='Tunneling Populations',
        marker=dict(color='red'),
        opacity=0.5
    ))

    fig.update_layout(
        barmode='group',
        title='Population of Spin States',
        xaxis_title='Population',
        yaxis_title='Basis States',
        plot_bgcolor='white',
        paper_bgcolor='white',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        height=400,
        width=700,
        margin=dict(l=150, r=20, t=50, b=20)
    )
    fig.update_xaxes(showgrid=True, gridcolor='lightgray')
    # show first state at top (matches typical horizontal-bar plotting)
    #fig.update_yaxes(autorange='reversed')

    fig.show()

    return_dict = {
        'fig': fig}

    return return_dict

def plotRateContributions(spinsys: SpinSys, yscale: str='linear', N=None, ylim: list=None):
    """
    Plot the contributions of the highest N transition rates between spin states.

    Parameters
    ----------
    spinsys : SpinSys
        The spin system object containing the states, rates, and matrix dimension.
        It is expected to have the following attributes:
            - dimensionOfMatrix: int, the number of spin states.
            - statesWithoutE: list of str, labels for the spin states.
            - Rates: list of 2D numpy arrays, each representing the rate matrix for a process.
    yscale : str, optional
        The scale for the y-axis ('linear' or 'log'). Default is 'linear'.
    N : int, optional
        The number of highest rate transitions to display. If None, all transitions are shown (default is None).
    ylim : list of float, optional
        The limits for the y-axis as [ymin, ymax]. If None, no limits are set (default is None).

    Returns
    -------
    return_dict : dict
        A dictionary containing:
        - 'fig': plotly.graph_objects.Figure
            The plotly figure object containing the ESR spectrum plot.
    """
    hamil.calcEigStates(spinsys)
    calcRates(spinsys)

    labels = ['ts', 'st', 'ss','tt']
    colors = ['blue', 'orange', 'green', 'red']

    # Build a list of all transitions (i -> f) for i != f
    all_transitions = [(i, f) for i in range(spinsys.dimensionOfMatrix) for f in range(spinsys.dimensionOfMatrix) if i != f]
    transition_labels_all = [f'{spinsys.statesWithoutE[i]} → {spinsys.statesWithoutE[f]}' for (i, f) in all_transitions]

    def safe_rate(r):
        return 0.0 if (r is None or (isinstance(r, float) and np.isnan(r))) else float(r)

    if N is None:
        # Prepare y-values for each rate contribution type across all transitions
        data = []
        for idx in range(4):
            y_vals = [safe_rate(spinsys.Rates[idx][i, f]) for (i, f) in all_transitions]
            data.append(go.Bar(
                name=labels[idx],
                x=transition_labels_all,
                y=y_vals,
                marker_color=colors[idx],
                opacity=0.7,
                hovertemplate='%{x}<br>' + labels[idx] + ': %{y}<extra></extra>'
            ))

        fig = go.Figure(data=data)
        fig.update_layout(
            barmode='stack',
            title='Rate Contributions',
            xaxis=dict(title='Transitions', tickangle=45),
            yaxis=dict(title='Rate', type='linear' if yscale == 'linear' else yscale, autorange=True),
            plot_bgcolor='white',
            paper_bgcolor='white',
            width=1200,
            height=500,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        fig.update_yaxes(
            exponentformat='e',
            showexponent='all'
        )
        if ylim is not None:
            fig.update_yaxes(range=ylim)
        fig.show()

    else:
        # Collect all transitions and their total rates, ignoring NaNs
        transitions = []
        for i in range(spinsys.dimensionOfMatrix):
            for f in range(spinsys.dimensionOfMatrix):
                if i != f:
                    total_rate = sum([spinsys.Rates[idx][i, f] for idx in range(4)])
                    if not (isinstance(total_rate, float) and np.isnan(total_rate)):
                        transitions.append(((i, f), total_rate))

        # Sort by total_rate descending and keep top N
        transitions = sorted(transitions, key=lambda x: x[1], reverse=True)[:N]
        if len(transitions) == 0:
            fig = go.Figure()
            fig.update_layout(
                title=f'Top {N} Rate Contributions (no valid transitions)',
                plot_bgcolor='white',
                paper_bgcolor='white'
            )
            fig.update_yaxes(
                exponentformat='e',
                showexponent='all'
            )
            fig.show()
            return

        selected_pairs = [t[0] for t in transitions]
        transition_labels = [f'{spinsys.statesWithoutE[i]} → {spinsys.statesWithoutE[f]}' for (i, f) in selected_pairs]

        data = []
        for idx in range(4):
            y_vals = [safe_rate(spinsys.Rates[idx][i, f]) for (i, f) in selected_pairs]
            data.append(go.Bar(
                name=labels[idx],
                x=transition_labels,
                y=y_vals,
                marker_color=colors[idx],
                opacity=0.7,
                hovertemplate='%{x}<br>' + labels[idx] + ': %{y}<extra></extra>'
            ))

        fig = go.Figure(data=data)
        fig.update_layout(
            barmode='stack',
            title=f'Top {N} Rate Contributions',
            xaxis=dict(title='Transitions', tickangle=45),
            yaxis=dict(title='Rate', type='linear' if yscale == 'linear' else yscale, autorange=True),
            plot_bgcolor='white',
            paper_bgcolor='white',
            width=1200,
            height=500,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        fig.update_yaxes(
            exponentformat='e',
            showexponent='all'
        )
        if ylim is not None:
            fig.update_yaxes(range=ylim)
        fig.show()

    return_dict = {
        'fig': fig}
    
    return return_dict

def plotRatesTo(spinsys: SpinSys, state: int, yscale: str='linear', N: int=None):
    """
    Plot the rate contributions to a specified state in a spin system.

    This function calculates and visualizes the transition rates from all other states to a given target state
    in the provided spin system. The rates are displayed as stacked bar plots, with each bar representing
    the contributions from different rate components (e.g., 'ts', 'st', 'ss', 'tt').

    Parameters
    ----------
    spinsys : SpinSys
        The spin system object containing state information and rate matrices.
    state : int
        The index of the target state to which rates are plotted.
    yscale : str, optional
        The scale for the y-axis of the plot. Can be 'linear', 'log', etc. Default is 'linear'.
    N : int, optional
        The number of highest rate transitions to display. If None, all transitions are shown (default is None).

    Returns
    -------
    return_dict : dict
        A dictionary containing:
        - 'fig': plotly.graph_objects.Figure
            The plotly figure object containing the ESR spectrum plot.

    Notes
    -----
    - The function first calculates the eigenstates and rates for the spin system.
    """
    # Calculate the eigenstates and rates first
    hamil.calcEigStates(spinsys)
    calcRates(spinsys)

    def safe_rate(r):
        return 0.0 if (r is None or (isinstance(r, float) and np.isnan(r))) else float(r)

    # Plotly implementation of plotRatesTo
    if N is not None:
        # Collect all transitions and their total rates to the target state
        transitions = []
        for i in range(spinsys.dimensionOfMatrix):
            if i != state:
                total_rate = sum([spinsys.Rates[idx][i, state] for idx in range(4)])
                if not (isinstance(total_rate, float) and np.isnan(total_rate)):
                    transitions.append(((i, state), total_rate))

        # Sort by total_rate descending and take top N
        transitions = sorted(transitions, key=lambda x: x[1], reverse=True)[:N]
        if len(transitions) == 0:
            fig = go.Figure()
            fig.update_layout(title=f'Top {N} Rate Contributions to {spinsys.statesWithoutE[state]} (no valid transitions)')
            fig.show()

        x_labels = [f'{spinsys.statesWithoutE[i]} → {spinsys.statesWithoutE[state]}' for (i, _), _ in transitions]
        # Build stacked traces per rate type
        data = []
        for idx in range(4):
            y_vals = [safe_rate(spinsys.Rates[idx][i, state]) for (i, _), _ in transitions]
            data.append(go.Bar(
                name=['ts', 'st', 'ss', 'tt'][idx],
                x=x_labels,
                y=y_vals,
                marker_color=['blue', 'orange', 'green', 'red'][idx],
                opacity=0.8,
                hovertemplate='%{x}<br>' + ['ts', 'st', 'ss', 'tt'][idx] + ': %{y}<extra></extra>'
            ))

        fig = go.Figure(data=data)
        fig.update_layout(
            barmode='stack',
            title=f'Top {N} Rate Contributions to {spinsys.statesWithoutE[state]}',
            xaxis=dict(title='Transitions', tickangle=45),
            yaxis=dict(title='Rate', type=yscale),
            plot_bgcolor='white',
            paper_bgcolor='white',
            width=1000,
            height=500,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        fig.update_yaxes(
            exponentformat='e',
            showexponent='all'
        )
        fig.show()

    else:
        # All transitions to the target state
        x_labels = []
        for i in range(spinsys.dimensionOfMatrix):
            if i != state:
                x_labels.append(f'{spinsys.statesWithoutE[i]} → {spinsys.statesWithoutE[state]}')

        data = []
        for idx in range(4):
            y_vals = []
            for i in range(spinsys.dimensionOfMatrix):
                if i != state:
                    y_vals.append(safe_rate(spinsys.Rates[idx][i, state]))
            data.append(go.Bar(
                name=['ts', 'st', 'ss', 'tt'][idx],
                x=x_labels,
                y=y_vals,
                marker_color=['blue', 'orange', 'green', 'red'][idx],
                opacity=0.8,
                hovertemplate='%{x}<br>' + ['ts', 'st', 'ss', 'tt'][idx] + ': %{y}<extra></extra>'
            ))

        fig = go.Figure(data=data)
        fig.update_layout(
            barmode='stack',
            title=f'Rate Contributions to {spinsys.statesWithoutE[state]}',
            xaxis=dict(title='Transitions', tickangle=45),
            yaxis=dict(title='Rate', type=yscale),
            plot_bgcolor='white',
            paper_bgcolor='white',
            width=1000,
            height=500,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        fig.update_yaxes(
            exponentformat='e',
            showexponent='all'
        )
        fig.show()

        return_dict = { 
            'fig': fig}

        return return_dict

def plotRatesFrom(spinsys: SpinSys, state: int, yscale: str='linear', N: int=None):
    """
    Plot the rate contributions from a specified state in a spin system.

    This function calculates and visualizes the transition rates to all other states from a given initial state
    in the provided spin system. The rates are displayed as stacked bar plots, with each bar representing
    the contributions from different rate components (e.g., 'ts', 'st', 'ss', 'tt').

    Parameters
    ----------
    spinsys : SpinSys
        The spin system object containing state information and rate matrices.
    state : int
        The index of the target state to which rates are plotted.
    yscale : str, optional
        The scale for the y-axis of the plot. Can be 'linear', 'log', etc. Default is 'linear'.
    N : int, optional
        The number of highest rate transitions to display. If None, all transitions are shown (default is None).

    Returns
    -------
    return_dict : dict
        A dictionary containing:
        - 'fig': plotly.graph_objects.Figure
            The plotly figure object containing the ESR spectrum plot.

    Notes
    -----
    - The function first calculates the eigenstates and rates for the spin system.
    """
    hamil.calcEigStates(spinsys)
    calcRates(spinsys)

    labels = ['ts', 'st', 'ss','tt']
    colors = ['blue', 'orange', 'green', 'red']

    def safe_rate(r):
        return 0.0 if (r is None or (isinstance(r, float) and np.isnan(r))) else float(r)

    if N is not None:
        # Collect all transitions and their total rates from the given state
        transitions = []
        for f in range(spinsys.dimensionOfMatrix):
            if f != state:
                total_rate = sum([spinsys.Rates[idx][state, f] for idx in range(4)])
                if not (isinstance(total_rate, float) and np.isnan(total_rate)):
                    transitions.append(((state, f), total_rate))
        transitions = sorted(transitions, key=lambda x: x[1], reverse=True)[:N]
        if len(transitions) == 0:
            fig = go.Figure()
            fig.update_layout(title=f'Top {N} Rate Contributions from {spinsys.statesWithoutE[state]} (no valid transitions)')
            fig.show()

        x_labels = [f'{spinsys.statesWithoutE[s]} → {spinsys.statesWithoutE[f]}' for (s, f), _ in transitions]
        data = []
        for idx in range(4):
            y_vals = [safe_rate(spinsys.Rates[idx][s, f]) for (s, f), _ in transitions]
            data.append(go.Bar(
                name=labels[idx],
                x=x_labels,
                y=y_vals,
                marker_color=colors[idx],
                opacity=0.8,
                hovertemplate='%{x}<br>' + labels[idx] + ': %{y}<extra></extra>'
            ))

        fig = go.Figure(data=data)
        fig.update_layout(
            barmode='stack',
            title=f'Top {N} Rate Contributions from {spinsys.statesWithoutE[state]}',
            xaxis=dict(title='Transitions', tickangle=45),
            yaxis=dict(title='Rate', type=yscale),
            plot_bgcolor='white',
            paper_bgcolor='white',
            width=1000,
            height=500,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        fig.update_yaxes(
            exponentformat='e',
            showexponent='all'
        )
        fig.show()

    else:
        # All transitions from the given state
        x_labels = [f'{spinsys.statesWithoutE[state]} → {spinsys.statesWithoutE[f]}' for f in range(spinsys.dimensionOfMatrix) if f != state]
        data = []
        for idx in range(4):
            y_vals = [safe_rate(spinsys.Rates[idx][state, f]) for f in range(spinsys.dimensionOfMatrix) if f != state]
            data.append(go.Bar(
                name=labels[idx],
                x=x_labels,
                y=y_vals,
                marker_color=colors[idx],
                opacity=0.8,
                hovertemplate='%{x}<br>' + labels[idx] + ': %{y}<extra></extra>'
            ))

        fig = go.Figure(data=data)
        fig.update_layout(
            barmode='stack',
            title=f'Rate Contributions from {spinsys.statesWithoutE[state]}',
            xaxis=dict(title='Transitions', tickangle=45),
            yaxis=dict(title='Rate', type=yscale),
            plot_bgcolor='white',
            paper_bgcolor='white',
            width=1000,
            height=500,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        fig.update_yaxes(
            exponentformat='e',
            showexponent='all'
        )
        fig.show()

    return_dict = {
        'fig': fig}

    return return_dict

# endregion Visualization

#----------------------------------------------------------------------------------------------------------
## dIdV
import numpy as np
import scipy as sp
import scipy.constants as const
from scipy.linalg import null_space
import plotly.graph_objects as go
import plotly.colors
from plotly.subplots import make_subplots

from spinfinity.SpinSys import SpinSys
import spinfinity.SpinHamiltonian as hamil 
import spinfinity.Bath as bath

# region 2nd Order

def calcIETS_2nd(spinsys: SpinSys, Vrange: list = [-30,30], N: int = 100, norm: bool = False):
    """
    Calculate the second derivative of the tunneling current (dI/dV) for a given spin system  in 2nd Order.

    Parameters
    ----------
    spinsys : SpinSys
        The spin system object containing eigenvalues, eigenvectors, temperature, and other relevant parameters.
    Vrange : list of float, optional
        The voltage range [V_min, V_max] over which to calculate dI/dV. Default is [-30, 30].
    N : int, optional
        Number of voltage points in the output array. Default is 100.
    norm : bool, optional
        If True, normalize the output dI/dV array to the range [0, 1]. Default is False.

    Returns
    -------
    V_array : ndarray
        Array of voltage values over the specified range.
    dIdV_array : ndarray
        Calculated dI/dV values corresponding to each voltage in V_array.

    Notes
    -----
    - The function assumes that the spin system object contains all necessary attributes.
    - Eigenvalues and eigenvectors are calculated if not already present in the spin system.
    - The calculation uses a Boltzmann distribution for thermal population.
    - The tunneling rate matrix is computed using the `calcTunnelingMatrixElements` function.
    - The output can be normalized for easier comparison between different systems or parameters.
    """
    # Initialize the arrays
    V_array = np.linspace(Vrange[0], Vrange[1], N)
    dIdV_array = np.zeros(N, dtype=float)

    # Calculate the Eignevalues and Eignevectors if not already done
    if not hasattr(spinsys, 'eigVectors') or spinsys.eigVectors is None:
        hamil.calcEigEnergies(spinsys)

    # Calculate the thermal population (Boltzmann distribution)
    bath.calcPopulations(spinsys, AllowPumping=False)

    # Calculate the Rate Matrix     
    RateMatrix = bath.calcTunnelingMatrixElements(spinsys, "st")[0]

    for i in range(spinsys.dimensionOfMatrix):
        if spinsys.pop_0[i] > 10**(-6):
            for j in range(spinsys.dimensionOfMatrix):
                ep = tunnelingBroadenedStepFunction(np.array((spinsys.E_All[j] - spinsys.E_All[i] - V_array) / (spinsys.T * spinsys.kB)))
                en = tunnelingBroadenedStepFunction(np.array((spinsys.E_All[j] - spinsys.E_All[i] + V_array) / (spinsys.T * spinsys.kB)))
                ytemp = spinsys.pop_0[i] * (RateMatrix[j, i] * ep + RateMatrix[i, j] * en)
                dIdV_array = dIdV_array + ytemp

    # Normalize with Rate_00 to get correct units
    Rate_0 = RateMatrix[0][0]  # Transition matrix element of the ground state
    dIdV_array = dIdV_array * spinsys.G / Rate_0
    
    # Normalize dI/dV if required
    if norm:
        dIdV_array = dIdV_array / dIdV_array[0]  # Normalize to the first value
    return V_array, dIdV_array


def tunnelingBroadenedStepFunction(x_V):
    """
    Compute the tunneling broadened step function.

    This function calculates a broadened step function commonly used in tunneling spectroscopy.

    Parameters
    ----------
    x_V : array_like or float
        Input value(s) at which to evaluate the broadened step function.

    Returns
    -------
    y : ndarray or float
        The value(s) of the tunneling broadened step function at `x_V`. If the computation results
        in NaN values, they are replaced with 0.0.

    Notes
    -----
    The function uses an exponential transformation and handles potential NaN values gracefully.
    """
    z = np.exp(x_V)
    y = (1 + (x_V - 1) * z) / ((z - 1) ** 2)
    y = np.nan_to_num(y, nan=0.0)
    return y

# endregion 2nd Order

# region 2nd Order + Rate

def calcIETS_2ndR(
    spinsys: SpinSys,
    Vrange: list = [-30, 30],
    N: int = 200,
    AllowPumping: bool = True,
    norm: bool = True,
    Approach: str = "Loth"
):
    """
    Calculate the IETS (Inelastic Electron Tunneling Spectroscopy) spectrum using the rate equation approach.
    
    Parameters
    ----------
    spinsys : SpinSys
        The spin system object containing all relevant parameters and matrices.
    Vrange : list of float, optional
        The voltage range [V_min, V_max] (in mV) over which to calculate the spectrum. Default is [-30, 30].
    N : int, optional
        Number of voltage points in the spectrum. Default is 200.
    AllowPumping : bool, optional
        If True, allows for non-thermal population of spin states (pumping). If False, uses thermal (Boltzmann) population. Default is True.
    norm : bool, optional
        If True, normalizes the resulting dI/dV spectrum to the range [0, 1]. Default is True.
    Approach : {'Loth', 'Ternes'}, optional
        The approach used for rate factor calculation. "Loth" uses the Loth 2010 method, "Ternes" uses the Ternes 2010 method. Default is "Loth".
    
    Returns
    -------
    V_array : ndarray
        Array of voltage values (in mV) at which the spectrum is calculated.
    dIdV : ndarray
        The normalized differential conductance (dI/dV) spectrum.
    P : ndarray
        Matrix of state populations for each voltage point.
    
    Notes
    -----
    - The calculation is based on the rate equation approach and supports both thermal and non-thermal
      population distributions.
    """
    # Initialize the arrays
    V_array = np.linspace(Vrange[0], Vrange[1], N)
    I = np.zeros((N), dtype=float)
    P = np.zeros((spinsys.dimensionOfMatrix, N), dtype=float)
    Rates = np.zeros((4, spinsys.dimensionOfMatrix, spinsys.dimensionOfMatrix))

     # Calculate the Eignevalues and Eignevectors if not already done
    if not hasattr(spinsys, 'eigVectors') or spinsys.eigVectors is None:
        hamil.calcEigEnergies(spinsys)

    # Getting the relevant matrix elements
    spinsys.Matrix = bath.calcTunnelingMatrixElements(spinsys, "st")[0]
    spinsys.MatrixSS = bath.calcTunnelingMatrixElements(spinsys, "ss")
    spinsys.MatrixTT = bath.calcTunnelingMatrixElements(spinsys, "tt")[0]

    MatrixSS_summed = np.zeros((spinsys.dimensionOfMatrix, spinsys.dimensionOfMatrix))
    for i in range(spinsys.NSpins):
        MatrixSS_summed += spinsys.MatrixSS[i, :, :] * spinsys.G_ss[i]
   
    Rate_0 = spinsys.Matrix[0][0]  # Transition matrix element of the ground state

    # Rate-Factors using Loth 2010
    if Approach == "Loth":
        spinsys.G_st = (1 - spinsys.b0) * spinsys.G
        RateFactor_ts = spinsys.G_st / (const.e * Rate_0) / 1e3
        RateFactor_st = spinsys.G_st / (const.e * Rate_0) / 1e3
        RateFactor_ss = 1 / (const.e * Rate_0) / 1e3
        RateFactor_tt = spinsys.G_st**2/spinsys.G_ss[spinsys.ReadoutSpin]/(const.e*Rate_0)/1e3*spinsys.G_tt; #Gtt is actually calculated from Gss and Gst, we use the Gtt value 0,1 here just o allow it or not

    # Rate-Factors using Ternes 2010 (not default)
    if Approach == "Ternes":
        RateFactor_ts = 2 * np.pi / spinsys.hbar_meV * spinsys.T02
        RateFactor_st = 2 * np.pi / spinsys.hbar_meV * spinsys.T02
        RateFactor_ss = 2 * np.pi / spinsys.hbar_meV * spinsys.Jrs**2
        RateFactor_tt = 2 * np.pi / spinsys.hbar_meV * spinsys.T02**2 / (spinsys.Jrs**2)

    # Calculate the rates that stem from Tip tip and sample sample scattering, since they are not affected by the voltage
    RateIntegrals_ss = bath.calcRateIntegrals(spinsys, 'ss')
    RateIntegrals_tt = RateIntegrals_ss
    Rates[2, :, :] = RateFactor_ss * MatrixSS_summed * RateIntegrals_ss
    Rates[3, :, :] = RateFactor_tt * spinsys.MatrixTT * RateIntegrals_tt*spinsys.G_tt

    # This part happens now as often as we need it
    for I_index in range(N):
        V_DC = V_array[I_index]
        spinsys.V_DC = V_DC

        # Calculate rate integrals for the current voltage (These are the only changing parts)
        RateIntegrals_ts = bath.calcRateIntegrals(spinsys, 'ts')
        RateIntegrals_st = bath.calcRateIntegrals(spinsys, 'st')
        
        # Calculate rates based on rate factors and integrals
        Rates[0, :, :] = RateFactor_ts * spinsys.Matrix.T * RateIntegrals_ts
        Rates[1, :, :] = RateFactor_st * spinsys.Matrix * RateIntegrals_st
        
        # Sum the rates
        Rates_Summed = (
            Rates[0, :, :] + Rates[1, :, :] +
            Rates[2, :, :] + Rates[3, :, :] 
        )
        #print(f"Calculated Rates at V={V_DC} mV")
        #print(Rates_Summed[0,1])
        # Build the Rate-Equation matrix
        K = np.zeros((spinsys.dimensionOfMatrix, spinsys.dimensionOfMatrix))
        for i in range(spinsys.dimensionOfMatrix):
            for j in range(spinsys.dimensionOfMatrix):
                if i != j:
                    K[i, i] -= Rates_Summed[i, j]  # Rates leaving state i
                    K[i, j] = Rates_Summed[j, i]   # Rates leading to state i
        #print(K)
        # Calculate the thermal population (Boltzmann distribution)
        p0 = np.exp(-spinsys.E_All / (spinsys.kB * spinsys.T))
        pTot = np.sum(p0)
        spinsys.pop_0 = p0 / pTot  # Normalized thermal population

        # Solve the Rate-Equation if non-thermal population is desired
        if AllowPumping:
            try:
                p = null_space(K)
            except Exception as e:
                print("Warning: null_space failed, using thermal population. Error:", e)
                p = np.empty((0, 0))
            if p.size == 0:  # If no valid solution, use thermal distribution
                Populations = spinsys.pop_0
                print("Warning: No valid solution for rate equations found, using thermal population instead.")
            else:
                Populations = p[:, 0] / np.sum(p[:, 0])
        else:
            Populations = spinsys.pop_0
            print("Using thermal population (Boltzmann distribution).")

        # Deriving the tunneling current using Equation 41 from Ternes paper
        I_value = 0
        for i in range(spinsys.dimensionOfMatrix):
            for j in range(spinsys.dimensionOfMatrix):
                I_temp = const.e * Populations[i] * (Rates[0, i, j] - Rates[1, i, j])
                I_value += I_temp

        # Adjust current calculation based on approach
        if Approach == "Ternes":
            I_value = I_value / (const.e**2 * spinsys.T02 / spinsys.h_meV)
        if Approach == "Loth":
            I_value += spinsys.b0 * spinsys.G * spinsys.V_DC

        I[I_index] = I_value
        P[:, I_index] = Populations    

    # Calculate the derivative (dI/dV)
    dIdV = calculate_derivative(I, (V_array * 1e-3))  # Convert V_array to volts

    # Normalize dI/dV if required
    if norm:
        dIdV = (dIdV / dIdV[0]) # Normalize to the first value

    return V_array, dIdV, P


def calculate_derivative(I: list, V: list):
    """
    Calculate the numerical derivative dI/dV of current (I) with respect to voltage (V).

    Parameters
    ----------
    I : array_like
        Array of current values.
    V : array_like
        Array of voltage values. Must be the same length as `I`.
        
    Returns
    -------
    dIdV : ndarray
        Array of numerical derivatives dI/dV, same length as input arrays.

    Notes
    -----
    Assumes that `I` and `V` are one-dimensional arrays of equal length.
    """
    n = len(I)  # Length of the array
    dIdV = np.zeros(n)  # Initialize the output array for derivative

    # Forward difference for the first point
    dIdV[0] = (I[1] - I[0]) / (V[1] - V[0])

    # Central difference for the middle points
    for i in range(1, n - 1):
        dIdV[i] = (I[i + 1] - I[i - 1]) / (V[i + 1] - V[i - 1])

    # Backward difference for the last point
    dIdV[n - 1] = (I[n - 1] - I[n - 2]) / (V[n - 1] - V[n - 2])

    return dIdV

# endregion 2nd Order + Rate

# region Visualization
def plotIETS(spinsys: SpinSys, Vrange: list=[-30, 30], N: int=100, norm: bool=False, order: str='2nd', plotAllSpins: bool=False):
    """
    Plot the Inelastic Electron Tunneling Spectroscopy (IETS) spectrum for a given spin system.

    Parameters
    ----------
    spinsys : SpinSys
        The spin system object for which the IETS spectrum is to be calculated.
    Vrange : list, optional
        The voltage range [min, max] in millivolts over which to calculate the spectrum (default is [-30, 30]).
    N : int, optional
        Number of points in the voltage range for the spectrum calculation (default is 400).
    norm : bool, optional
        If True, normalize the calculated spectrum (default is False).
    order : str, optional
        The order of the IETS calculation. Supported orders are '2nd' and '2ndR'.

    Raises
    ------
    ValueError
        If an unsupported order is specified.

    Returns
    -------
    return_dict : dict
        A dictionary containing:
        - 'fig': plotly.graph_objects.Figure
            The plotly figure object containing the ESR spectrum plot.
        - 'V_array': numpy.ndarray
            The voltage values used in the spectrum.
        - 'dIdVsignal': numpy.ndarray
            The calculated dI/dV signal values.
    Notes
    -----
    - the dimension of the dIdVsignal changes when plotAllSpins=True
    """
    # Define y-axis label according to normalization 
    if norm:
        y_label = 'Normalized dI/dV (a.u.)'
    else:
        y_label = 'dI/dV (S)'

    if plotAllSpins==False:

        # Initialize x and y arrays
        x = np.zeros(N) # Voltage 
        y = np.zeros(N) # dI/dV Signal
        # Calculate the IETS spectrum for the specified order
        if order == '2nd':
            x, y = calcIETS_2nd(spinsys, Vrange=Vrange, N=N, norm=norm)
        elif order == '2ndR':
            x, y, _ = calcIETS_2ndR(spinsys, Vrange=Vrange, N=N, norm=norm)
        else:
            raise ValueError("Unsupported order. Currently only '2nd' order is implemented.")
    
        # Plot the IETS spectrum using Plotly with white background and black border
        y_max = np.max(y)
        if y_max == 0:
            y_max = 1.0
        x0, x1 = float(np.min(x)), float(np.max(x))
        y0, y1 = 0.0, float(y_max * 1.1)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode='lines',
            name='IETS Spectrum',
            line=dict(color='black', width=2)
        ))

        fig.update_layout(
            title=f'IETS Spectrum ({order} Order)',
            xaxis_title='Bias Voltage (mV)',
            yaxis_title=y_label,
            plot_bgcolor='white',
            paper_bgcolor='white',
            width=800,
            height=500,
            margin=dict(l=60, r=20, t=60, b=60)
        )

        # set explicit axis ranges and add a black rectangle border around the plotting area
        fig.update_xaxes(range=[x0, x1], showgrid=True, gridcolor='lightgray', zeroline=False)
        fig.update_yaxes(range=[y0, y1], showgrid=True, gridcolor='lightgray', zeroline=False)

        fig.update_layout(
            shapes=[dict(
            type="rect",
            xref="x",
            yref="y",
            x0=x0,
            x1=x1,
            y0=y0,
            y1=y1,
            line=dict(color="black", width=2),
            fillcolor="rgba(0,0,0,0)"
            )]
        )

        fig.show()

    elif plotAllSpins==True:

        # Initialize x and y arrays
        x = np.zeros(N)
        y = np.zeros((len(spinsys.Spins),N))

        for i in range(len(spinsys.Spins)):
            spinsys.ReadoutSpin = i
            #print(f"Calculating IETS for Spin {i} with ReadoutSpin={spinsys.ReadoutSpin}")
            if order == '2nd':
                x, y[i,:] = calcIETS_2nd(spinsys, Vrange=Vrange, N=N, norm=norm)
            elif order == '2ndR':
                x, y[i,:], _ = calcIETS_2ndR(spinsys, Vrange=Vrange, N=N, norm=norm)
            else:
                raise ValueError("Unsupported order. Currently only '2nd' '2ndR' and '3rd' order are implemented.")
            
            # Plot all IETS spectra in the same figure on top of each other
            # Only create the figure once, before the loop
            #if i == 0:
        # After computing x and y for all spins, build a Plotly figure similar to the single-spin case
        color_list = plotly.colors.qualitative.Plotly
        fig = go.Figure()

        # Add one trace per spin (overlayed)
        nspins = y.shape[0]
        for i in range(nspins):
            fig.add_trace(go.Scatter(
            x=x,
            y=y[i, :],
            mode='lines',
            line=dict(color=color_list[i % len(color_list)], width=2),
            name=f'Spin {i}'
            ))

        # Determine axis ranges similar to single-spin branch
        y_max = np.max(y) if np.max(y) != 0 else 1.0
        x0, x1 = float(np.min(x)), float(np.max(x))
        y0, y1 = 0.0, float(y_max * 1.1)

        fig.update_layout(
            title=f'IETS Spectrum ({order} Order) for All Spins',
            xaxis_title='Bias Voltage (mV)',
            yaxis_title=y_label,
            plot_bgcolor='white',
            paper_bgcolor='white',
            width=800,
            height=500,
            margin=dict(l=60, r=20, t=60, b=60)
        )

        fig.update_xaxes(range=[x0, x1], showgrid=True, gridcolor='lightgray', zeroline=False)
        fig.update_yaxes(range=[y0, y1], showgrid=True, gridcolor='lightgray', zeroline=False)

        # add a black rectangle border around the plotting area
        fig.update_layout(
            shapes=[dict(
            type="rect",
            xref="x",
            yref="y",
            x0=x0,
            x1=x1,
            y0=y0,
            y1=y1,
            line=dict(color="black", width=2),
            fillcolor="rgba(0,0,0,0)"
            )]
        )

        fig.show()

    return_dict = {
        'fig': fig,
        'V_array': x,
        'dIdVsignal': y}      
        
    return return_dict

# endregion Visualization

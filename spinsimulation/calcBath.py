import numpy as np
import scipy as sp
import scipy.constants as const
from scipy.linalg import null_space
import plotly.graph_objects as go
import plotly.colors
from plotly.subplots import make_subplots

from spinsimulation.SpinSys import SpinSys
import spinsimulation.SpinHamiltonian as hamil 

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
        calcPopulations(spinsys, AllowPumping=True)
    
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

def calcLifetimes(spinsys: SpinSys, AllowPumping: bool = True):
    """
    Calculate the lifetimes (T1) for each state in a spin system.

    This function computes the total and natural lifetimes for each state in the given spin system
    by summing the transition rates out of each state. If the rate matrix is not already calculated,
    it will be computed.

    Parameters
    ----------
    spinsys : SpinSys
        The spin system object containing state information and rate matrices.
    AllowPumping : bool, optional
        Whether to allow pumping effects in the calculation (default is True).

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
        calcRates(spinsys)

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

def calcTunnelCurrent(spinsys: SpinSys, AllowPumping: bool = True):
    """
    Calculate the tunneling current for the given spin system.

    Parameters
    ----------
    spinsys : SpinSys
        The spin system object containing energy levels, populations, rates, and other relevant parameters.
    AllowPumping : bool, optional
        If True, allows for non-thermal population effects (default is True).

    Returns
    -------
    I : float
        The calculated tunneling current.

    Notes
    -----
    - The function updates the `I` attribute of the `spinsys` object in-place.
    - The current is calculated using the populations and transition rates between spin states.
    - If populations are not available, they are calculated using `calcPopulations`.
    """
    if not hasattr(spinsys, 'Populations') or spinsys.Populations is None:
            calcPopulations(spinsys, AllowPumping=AllowPumping)

    # Deriving the tunneling current using Equation 41 from Ternes paper
    spinsys.I = 0
    for i in range(spinsys.dimensionOfMatrix):
        for j in range(spinsys.dimensionOfMatrix):
            I_temp = const.e * spinsys.Populations[i] * (spinsys.Rates[0, i, j] - spinsys.Rates[1, i, j])
            spinsys.I += I_temp

    # Adjust current calculation
    spinsys.I += spinsys.b0 * spinsys.G * spinsys.V_DC

    return spinsys.I

def calcPopulations(spinsys: SpinSys, AllowPumping:bool = True):
    """
    Calculate the population distribution of spin states in a given spin system.

    Computes the thermal population distribution based on Boltzmann statistics.
    If pumping is allowed, calculates the non-thermal population by solving the steady-state
    of the rate equations.

    Parameters
    ----------
    spinsys : SpinSys
        The spin system object containing energy levels, temperature, and other relevant parameters.
    AllowPumping : bool, optional
        If True, calculates the non-thermal population considering pumping effects (default is True).

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
            calcRates(spinsys)
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

def calcRates(spinsys: SpinSys):
    """
    Calculate transition rates for a given spin system.

    Parameters
    ----------
    spinsys : SpinSys
        The spin system object containing all relevant parameters and matrices.

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
    - The rate factors are computed.
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
    spinsys.G_st = (1 - spinsys.b0) * spinsys.G

    RateFactor_ts = spinsys.G_st / (const.e * Rate_0) / 1e3
    RateFactor_st = spinsys.G_st / (const.e * Rate_0) / 1e3
    # The G_ss value might be different for every spin thats why we previously incorpareted it when we summed the Matrix
    RateFactor_ss = 1 / (const.e * Rate_0) / 1e3
    RateFactor_tt = spinsys.G_st**2/spinsys.G_ss[spinsys.ReadoutSpin]/(const.e*Rate_0)/1e3*spinsys.G_tt; #Gtt is actually calculated from Gss and Gst, we use the Gtt value 0,1 here just o allow it or not

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
def plotPopulations(spinsys: SpinSys, AllowPumping: bool = True):
    """
    Plot the thermal populations of the spin system's states.

    Parameters
    ----------
    spinsys : SpinSys
        The spin system object containing the necessary parameters and methods.
    AllowPumping : bool, optional
        Whether to allow pumping effects in the calculation (default is True).

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
            calcPopulations(spinsys,AllowPumping=AllowPumping)
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

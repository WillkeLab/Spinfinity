import numpy as np
import scipy as sp
import scipy.constants as const
from scipy.linalg import null_space

from spinsimulation.SpinSys import SpinSys
import spinsimulation.SpinHamiltonian as hamil 


# region Physical Quantities

def calcMagnetization(spinsys: SpinSys, type: str='z', AllowPumping: bool = True, Approach: str = "Loth"):
    """
    Calculates the magnetization of a spin system along a specified axis.

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
    Calculates the lifetimes (T1) for each state in a spin system.
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
    Calculates the tunneling current for the given spin system.

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
    Calculates the population distribution of spin states in a given spin system.
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
        p = null_space(spinsys.K)
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
    Calculates transition rates for a given spin system.

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

# region SpinPumping 

def split_Populations(spinsys: SpinSys, subspin: SpinSys):
    """
    Splits the populations of a spin system into two subpopulations based on overlaps with subspin eigenvectors.
    This function first calculates the rates and populations for the full spin system. It then projects the populations
    onto the eigenvectors of a subspin system using overlap calculations, resulting in two sets of subpopulations
    (A and B) according to the specified overlap order.

    Parameters
    ----------
    spinsys : SpinSys
        The full spin system object, containing eigenvectors and populations.
    subspin : SpinSys
        The subspin system object, whose eigenvectors are used for projection.

    Returns
    -------
    Populations_A : np.ndarray
        Array of projected populations using overlap order 'BA'.
    Populations_B : np.ndarray
        Array of projected populations using overlap order 'AB'.

    Notes
    -----
    This function assumes that `calcRates`, `calcPopulations`, and `hamil.calcOverlap` are defined elsewhere.
    The function modifies the populations of `spinsys` in-place before projection.
    """
    
    # First we calculate the Rates and Populations of the whole system
    #hamil.calcEigEnergies(subspin)
    #hamil.calcEigEnergies(spinsys)
    #hamil.calcEigStates(spinsys)
    #hamil.calcEigStates(subspin)
    calcRates(spinsys)
    calcPopulations(spinsys, AllowPumping=True)

    # Initialize 
    Populations_A=np.zeros(subspin.dimensionOfMatrix)
    Populations_B=np.zeros(subspin.dimensionOfMatrix)

    for idx in range(subspin.dimensionOfMatrix):
        overlap_A = hamil.calcOverlap(spinsys, subspin.dimensionOfMatrix, subspin.dimensionOfMatrix, subspin.eigVectors[:, idx], order='BA')
        overlap_B = hamil.calcOverlap(spinsys, subspin.dimensionOfMatrix, subspin.dimensionOfMatrix, subspin.eigVectors[:, idx], order='AB')
        Populations_A[idx] = sum(overlap_A * spinsys.Populations)
        Populations_B[idx] = sum(overlap_B * spinsys.Populations)

    return Populations_A, Populations_B

def split_Magnetization(spinsys: SpinSys, subspin: SpinSys, type: str='z'):
    
    # Calculate first the sub populations
    pA, pB = split_Populations(spinsys=spinsys, subspin=subspin)

    # A
    subspin.Populations = pA
    calcMagnetization(subspin, type=type, AllowPumping=True)
    Mag_A = subspin.Mag_tot

    # B
    subspin.Populations = pB
    calcMagnetization(subspin, type=type, AllowPumping=True)
    Mag_B = subspin.Mag_tot

    return Mag_A, Mag_B

# endregion SpinPumping



import numpy as np
import scipy as sp
import scipy.constants as const
from scipy.linalg import null_space

from SpinSys import SpinSys
import SpinHamiltonian as hamil 

# region 2nd Order

def calcIETS_2nd(spinsys: SpinSys, Vrange: list = [-10,10], N: int = 100, norm: bool = False):
    """
    Calculates the IETS spectrum for a thermally occupied spin system in 2nd order perturbation.
    """
    # Initialize the arrays
    V_array = np.linspace(Vrange[0], Vrange[1], N)
    dIdV_array = np.zeros(N, dtype=float)

    # Calculate the Eignevalues and Eignevectors if not already done
    if not hasattr(spinsys, 'eigVectors') or spinsys.eigVectors is None:
        hamil.calcEigEnergies(spinsys)

    # Calculate the thermal population (Boltzmann distribution)
    pop_0 = np.exp(-spinsys.E_All / (spinsys.kB * spinsys.T))
    pTot = np.sum(pop_0)
    spinsys.pop_0 = pop_0 / pTot  # Normalized thermal population

    # Calculate the Rate Matrix     
    RateMatrix = calcTunnelingMatrixElements(spinsys, "st")[0]

    for i in range(spinsys.dimensionOfMatrix):
        if spinsys.pop_0[i] > 10**(-6):
            for j in range(spinsys.dimensionOfMatrix):
                ep = tunnelingBroadenedStepFunction(np.array((spinsys.E_All[j] - spinsys.E_All[i] - V_array) / (spinsys.T * spinsys.kB)))
                en = tunnelingBroadenedStepFunction(np.array((spinsys.E_All[j] - spinsys.E_All[i] + V_array) / (spinsys.T * spinsys.kB)))
                ytemp = spinsys.pop_0[i] * (RateMatrix[j, i] * ep + RateMatrix[i, j] * en)
                dIdV_array = dIdV_array + ytemp

    # Normalize dI/dV if required
    if norm:
        dIdV_array = (dIdV_array - np.min(dIdV_array)) / (np.max(dIdV_array) - np.min(dIdV_array))

    return V_array, dIdV_array

def calcTunnelingMatrixElements(spinsys: SpinSys, scattertype: str):
    """
    Calculate the tunneling matrix elements for a given spin system and scattering type.
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

def tunnelingBroadenedStepFunction(x_V):
    """
    Compute the tunneling-broadened step function.
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
    Rates = np.zeros((5, spinsys.dimensionOfMatrix, spinsys.dimensionOfMatrix))

     # Calculate the Eignevalues and Eignevectors if not already done
    if not hasattr(spinsys, 'eigVectors') or spinsys.eigVectors is None:
        hamil.calcEigEnergies(spinsys)

    # Getting the relevant matrix elements
    spinsys.Matrix = calcTunnelingMatrixElements(spinsys, "st")[0]
    spinsys.MatrixSS = calcTunnelingMatrixElements(spinsys, "ss")
    spinsys.MatrixTT = calcTunnelingMatrixElements(spinsys, "tt")[0]

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
        RateFactor_tt = 0

    # Rate-Factors using Ternes 2010 (not default)
    if Approach == "Ternes":
        RateFactor_ts = 2 * np.pi / spinsys.hbar_meV * spinsys.T02
        RateFactor_st = 2 * np.pi / spinsys.hbar_meV * spinsys.T02
        RateFactor_ss = 2 * np.pi / spinsys.hbar_meV * spinsys.Jrs**2
        RateFactor_tt = 2 * np.pi / spinsys.hbar_meV * spinsys.T02**2 / (spinsys.Jrs**2) * spinsys.G_tt

    # Calculate the rates that stem from Tip tip and sample sample scattering, since they are not affected by the voltage
    RateIntegrals_ss = calcRateIntegrals(spinsys, spinsys.V_DC, 'ss')
    RateIntegrals_tt = RateIntegrals_ss
    Rates[2, :, :] = RateFactor_ss * MatrixSS_summed * RateIntegrals_ss
    Rates[3, :, :] = RateFactor_tt * spinsys.MatrixTT * RateIntegrals_tt

    # This part happens now as often as we need it
    for I_index in range(N):
        V_DC = V_array[I_index]

        RateIntegrals_ts = calcRateIntegrals(spinsys, V_DC, 'ts')
        RateIntegrals_st = calcRateIntegrals(spinsys, V_DC, 'st')
        
        # Calculate rates based on rate factors and integrals
        Rates[0, :, :] = RateFactor_ts * spinsys.Matrix.T * RateIntegrals_ts
        Rates[1, :, :] = RateFactor_st * spinsys.Matrix * RateIntegrals_st

        # Sum the rates
        Rates_Summed = (
            Rates[0, :, :] + Rates[1, :, :] +
            Rates[2, :, :] + Rates[3, :, :] +
            Rates[4, :, :]
        )

        # Build the Rate-Equation matrix
        K = np.zeros((spinsys.dimensionOfMatrix, spinsys.dimensionOfMatrix))
        for i in range(spinsys.dimensionOfMatrix):
            for j in range(spinsys.dimensionOfMatrix):
                if i != j:
                    K[i, i] -= Rates_Summed[i, j]  # Rates leaving state i
                    K[i, j] = Rates_Summed[j, i]   # Rates leading to state i

        # Calculate the thermal population (Boltzmann distribution)
        p0 = np.exp(-spinsys.E_All / (spinsys.kB * spinsys.T))
        pTot = np.sum(p0)
        spinsys.pop_0 = p0 / pTot  # Normalized thermal population

        # Solve the Rate-Equation if non-thermal population is desired
        if AllowPumping:
            p = null_space(K)
            if p.size == 0:  # If no valid solution, use thermal distribution
                Populations = spinsys.p0
            else:
                Populations = p[:, 0] / np.sum(p[:, 0])
        else:
            Populations = spinsys.pop_0

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
        dIdV = (dIdV - np.min(dIdV)) / (np.max(dIdV) - np.min(dIdV))

    return V_array, dIdV, P


def calcRateIntegrals(spinsys: SpinSys, V_DC: float, type: str):
    """
    Calculate rate integrals for a given spin system and bias voltage.
    
    Parameters
    ----------
    spinsys : SpinSys
        An instance of the SpinSys class containing system parameters such as energy levels,
        Boltzmann constant, temperature, and matrix dimension.
    V_DC : float
        The applied DC bias voltage. If zero, a small value is used to avoid division by zero.
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

def calcRates(spinsys: SpinSys, AllowPumping: bool = True, Approach: str = "Loth"):
    """
    Calculate transition rates, populations, and tunneling current for a given spin system and voltage.

    Parameters
    ----------
    spinsys : SpinSys
        The spin system object containing all relevant parameters, matrices, and state information.
    AllowPumping : bool, optional
        If True, allows for non-thermal population pumping by solving the rate equations.
        If False, uses the thermal (Boltzmann) population. Default is True.
    Approach : str, optional
        The theoretical approach to use for rate calculations. Options are:
        - "Loth": Uses the Loth 2010 approach (default).
        - "Ternes": Uses the Ternes 2010 approach.

    Returns
    -------
    None
        The function updates the following attributes of the `spinsys` object in-place:
        - Rates : ndarray
            Transition rates between states for different tunneling processes.
        - Rates_Summed : ndarray
            Summed transition rates for all processes.
        - K : ndarray
            Rate-equation matrix.
        - Tau_total : ndarray
            Total lifetimes of each state.
        - pop_0 : ndarray
            Normalized thermal (Boltzmann) population.
        - Populations : ndarray
            Steady-state populations (thermal or pumped).
        - I : float
            Calculated tunneling current.

    Notes
    -----
    - The function assumes that the spin system object (`spinsys`) contains all necessary
      physical constants, tunneling matrices, and energy levels.
    - The function modifies the `spinsys` object in-place and does not return any value.
    - The calculation of rates and current depends on the chosen theoretical approach.
    """

    # Initialize the Rates matrix
    spinsys.Rates = np.zeros((5,spinsys.dimensionOfMatrix, spinsys.dimensionOfMatrix))

     # Calculate the Eignevalues and Eignevectors if not already done
    if not hasattr(spinsys, 'eigVectors') or spinsys.eigVectors is None:
        hamil.calcEigEnergies(spinsys)

    # Getting the relevant matrix elements
    spinsys.Matrix = calcTunnelingMatrixElements(spinsys,"st")[0]
    spinsys.MatrixSS = calcTunnelingMatrixElements(spinsys,"ss")
    spinsys.MatrixTT = calcTunnelingMatrixElements(spinsys,"tt")[0]
    
    MatrixSS_summed = np.zeros((spinsys.dimensionOfMatrix, spinsys.dimensionOfMatrix))

    for i in range(spinsys.NSpins):
        MatrixSS_summed += spinsys.MatrixSS[i, :, :] * spinsys.G_ss[i]

    # Calculating the 3 different integrals
    RateIntegrals_ts = calcRateIntegrals(spinsys, spinsys.V_DC, 'ts')
    RateIntegrals_st = calcRateIntegrals(spinsys, spinsys.V_DC, 'st')
    RateIntegrals_ss = calcRateIntegrals(spinsys, spinsys.V_DC, 'ss')
    #RateIntegrals_tt = spinsys.calcRateIntegrals(spinsys.V_DC, 'tt')

    Rate_0 = spinsys.Matrix[0][0]  # Transition matrix element of the ground state

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
    spinsys.Rates[0, :, :] = RateFactor_ts * spinsys.Matrix.T *  RateIntegrals_ts
    spinsys.Rates[1, :, :] = RateFactor_st * spinsys.Matrix *  RateIntegrals_st
    spinsys.Rates[2, :, :] = RateFactor_ss * MatrixSS_summed *  RateIntegrals_ss
    spinsys.Rates[3, :, :] = RateFactor_tt * spinsys.MatrixTT *  RateIntegrals_ss

    # Sum the rates
    spinsys.Rates_Summed = (spinsys.Rates[0,:, :] + spinsys.Rates[1, :, :] + 
                        spinsys.Rates[2, :, :] + spinsys.Rates[3, :, :] + 
                        spinsys.Rates[4, :, :])

    # Build the Rate-Equation matrix
    spinsys.K = np.zeros((spinsys.dimensionOfMatrix, spinsys.dimensionOfMatrix))
    R1 = np.zeros(spinsys.dimensionOfMatrix)
    RSS = np.zeros(spinsys.dimensionOfMatrix)

    for i in range(spinsys.dimensionOfMatrix):
        for j in range(spinsys.dimensionOfMatrix):
            if i != j:
                spinsys.K[i, i] -= spinsys.Rates_Summed[i, j]  # Rates leaving state i
                spinsys.K[i, j] = spinsys.Rates_Summed[j, i]  # Rates leading to state i
                R1[i] += spinsys.Rates[2, i, j]  # Summing up all the rates

    # Calculate the total lifetime (Tau_total)
    spinsys.Tau_total = 1 / R1

    # Calculate the thermal population (Boltzmann distribution)
    p0 = np.exp(-spinsys.E_All / (spinsys.kB * spinsys.T))
    pTot = np.sum(p0)
    spinsys.pop_0 = p0 / pTot  # Normalized thermal population

    # Solve the Rate-Equation if non-thermal population is desired
    if AllowPumping:
        p = null_space(spinsys.K)
        if p.size == 0:  # If no valid solution, use thermal distribution
            spinsys.Populations = spinsys.pop_0
        else:
            spinsys.Populations = p[:, 0] / np.sum(p[:, 0])
    else:
        spinsys.Populations = spinsys.pop_0

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

    return

# endregion 2nd Order + Rate

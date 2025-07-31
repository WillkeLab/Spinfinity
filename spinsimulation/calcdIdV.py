import numpy as np
import scipy as sp
import scipy.constants as const
from scipy.linalg import null_space

from spinsimulation.SpinSys import SpinSys
import spinsimulation.SpinHamiltonian as hamil 
import spinsimulation.calcBath as bath

# region 2nd Order

def calcIETS_2nd(spinsys: SpinSys, Vrange: list = [-30,30], N: int = 100, norm: bool = False):
    """
    Calculate the second derivative of the tunneling current (dI/dV) for a given spin system  in 2nd Order.

    Parameters
    ----------
    spinsys : SpinSys
        The spin system object containing eigenvalues, eigenvectors, temperature, and other relevant parameters.
    Vrange : list of float, optional
        The voltage range [V_min, V_max] over which to calculate dI/dV. Default is [-10, 10].
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

# end

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
        RateFactor_tt = 0

    # Rate-Factors using Ternes 2010 (not default)
    if Approach == "Ternes":
        RateFactor_ts = 2 * np.pi / spinsys.hbar_meV * spinsys.T02
        RateFactor_st = 2 * np.pi / spinsys.hbar_meV * spinsys.T02
        RateFactor_ss = 2 * np.pi / spinsys.hbar_meV * spinsys.Jrs**2
        RateFactor_tt = 2 * np.pi / spinsys.hbar_meV * spinsys.T02**2 / (spinsys.Jrs**2) * spinsys.G_tt

    # Calculate the rates that stem from Tip tip and sample sample scattering, since they are not affected by the voltage
    RateIntegrals_ss = bath.calcRateIntegrals(spinsys, 'ss')
    RateIntegrals_tt = RateIntegrals_ss
    Rates[2, :, :] = RateFactor_ss * MatrixSS_summed * RateIntegrals_ss
    Rates[3, :, :] = RateFactor_tt * spinsys.MatrixTT * RateIntegrals_tt

    # This part happens now as often as we need it
    for I_index in range(N):
        V_DC = V_array[I_index]
        spinsys.V_DC = V_DC

        RateIntegrals_ts = bath.calcRateIntegrals(spinsys, 'ts')
        RateIntegrals_st = bath.calcRateIntegrals(spinsys, 'st')
        
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

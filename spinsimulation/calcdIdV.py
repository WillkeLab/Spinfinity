import numpy as np
import scipy as sp
import scipy.constants as const
from SpinSys import SpinSys
import SpinHamiltonian as hamil 

# region 2nd Order

def calcIETS_2nd(spinsys: SpinSys, Vrange: list = [-10,10], N: int = 100, norm: bool = False):
    """
    Calculates the second derivative of the tunneling current (IETS spectrum) for a thermally occupied spin system.
    This function computes the inelastic electron tunneling spectroscopy (IETS) spectrum by evaluating the second derivative of the tunneling current with respect to voltage, considering thermal population of spin states according to the Boltzmann distribution.

    Parameters
    ----------
    spinsys : SpinSys
        The spin system object containing all relevant parameters, eigenvalues, and eigenvectors.
    Vrange : list of float, optional
        The voltage range over which to calculate the spectrum, given as [V_min, V_max]. Default is [-10, 10].
    N : int, optional
        Number of voltage points in the output spectrum. Default is 100.
    norm : bool, optional
        If True, normalizes the resulting dI/dV spectrum to the range [0, 1]. Default is False.

    Returns
    -------
    V_array : ndarray
        Array of voltage values over which the spectrum is calculated.
    dIdV_array : ndarray
        Calculated IETS spectrum (dI/dV) at each voltage point.

    Notes
    -----
    - The function assumes that the spin system object (`spinsys`) contains all necessary attributes, including eigenvalues (`E_All`), temperature (`T`), Boltzmann constant (`kB`), and matrix dimension (`dimensionOfMatrix`).
    - If eigenvalues and eigenvectors are not already computed, they will be calculated within the function.
    - The tunneling rate matrix is computed using `calcTunnelingMatrixElements`.
    - The function uses a tunneling broadened step function to account for thermal and tunneling broadening effects.
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
    This function computes the rate matrix elements associated with electron tunneling processes
    in a spin system, based on the specified scattering type. The calculation involves transforming
    spin operators into the eigenbasis of the system and combining them with tunneling electron
    matrix elements.

    Parameters
    ----------
    spinsys : SpinSys
        The spin system object containing eigenvectors, spin operators, and other relevant properties.
    scattertype : str
        The type of scattering process. Must be one of:
        - "ss": sample-sample scattering
        - "st": sample-tip scattering
        - "tt": tip-tip scattering

    Returns
    -------
    RateMatrix : np.ndarray
        The calculated rate matrix elements. The shape of the returned array depends on the
        scattering type:
        - For "ss": shape is (NSpins, dimensionOfMatrix, dimensionOfMatrix)
        - For "st" or "tt": shape is (1, dimensionOfMatrix, dimensionOfMatrix)

    Notes
    -----
    The function assumes that the tunneling electron matrix elements (x_el, y_el, z_el, u_el)
    are 2x2 matrices. The spin operators (Sx, Sy, Sz, U) are transformed into the eigenbasis
    before being combined with the tunneling matrix elements.
    """
    # Get the electron tunneling matrix elements
    # scattertype = "st", "tt", "ss"
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
    # Loop over 4 elements, combining matrix operations and calculating the rate matrix
    for k in range(N):
        # Calculate matrix elements for Sx, Sy, Sz, and U using the eigenvectors and spin operators
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
    This function computes the matrix elements for tunneling electrons between a tip and a sample,
    considering their respective spin polarizations. The calculation is performed for different
    scattering types: tip-to-sample ("st"), tip-to-tip ("tt"), and sample-to-sample ("ss").
    The function normalizes the polarizations if necessary, constructs the corresponding density
    matrices, computes their eigenvalues and eigenvectors, and then evaluates the matrix elements
    for the spin operators (S_x, S_y, S_z) and the identity operator.

    Parameters
    ----------
    spinsys : SpinSys
        An object representing the spin system, containing the tip and sample polarizations.
    scattertype : str
        The type of scattering matrix elements to compute. Must be one of:
        - "st": tip-to-sample
        - "tt": tip-to-tip
        - "ss": sample-to-sample

    Returns
    -------
    x : np.ndarray
        Matrix elements for the S_x operator.
    y : np.ndarray
        Matrix elements for the S_y operator.
    z : np.ndarray
        Matrix elements for the S_z operator.
    u : np.ndarray
        Matrix elements for the identity operator.

    Notes
    -----
    - The function assumes that the tip and sample polarizations are 3-element arrays.
    - If the norm of a polarization vector exceeds 1, it is normalized.
    - The function swaps the order of eigenvalues and eigenvectors to match MATLAB conventions.
    - If an invalid `scattertype` is provided, the function defaults to "st".
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
    # Convert eigenvalue arrays to diagonal matrices (like MATLAB output)
    eigValTip = np.diag(eigValTip_temp)
    eigValSample = np.diag(eigValSample_temp)

    # Swap the eigen value order
    eigValTip[[0, 1], [0, 1]] = eigValTip[[1, 0], [1, 0]]
    eigVecTip[:, [0, 1]] = eigVecTip[:, [1, 0]]

    # Calculate the weights matrix as sqrt(diag(eigValSample) * diag(eigValTip))
    Weights_st = np.sqrt(np.outer(np.diag(eigValSample), np.diag(eigValTip)))
    Weights_tt = np.sqrt(np.outer(np.diag(eigValTip), np.diag(eigValTip)))
    Weights_ss = np.sqrt(np.outer(np.diag(eigValSample), np.diag(eigValSample)))

    # Correct alignment of eigenvectors if needed (manual step)
    # Make sure that x_el, y_el, z_el are calculated correctly
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
    The function is defined in terms of the input variable `x_V`, and handles numerical instabilities 
    for large values by setting resulting NaNs to zero.

    Parameters
    ----------
    x_V : array_like
        Input value(s) at which to evaluate the broadened step function. Can be a scalar or a NumPy array.

    Returns
    -------
    y : ndarray or float
        The value(s) of the tunneling-broadened step function evaluated at `x_V`. Returns a NumPy array if
        input is array-like, or a float if input is scalar.

    Notes
    -----
    - For very small values of `x_V`, the function may approach 0.5, but this is not explicitly enforced.
    - NaN results (typically for large `x_V`) are set to 0 to avoid propagation of invalid values.
    """


    # Set small values of x to zero
    z = np.exp(x_V)
    y = (1 + (x_V - 1) * z) / ((z - 1) ** 2)

    # Set y for small x values to 0.5
    # y = np.where(np.abs(x_V) < 1e-5, 0.5, y)

    # We get some NaN results because of large x, we set them to 0
    y = np.nan_to_num(y, nan=0.0)

    return y

# endregion 2nd Order

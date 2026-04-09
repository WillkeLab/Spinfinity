"""
Utilities for calculating and visualizing IETS signals. 

This module provides routines to compute differential conductance spectra using:
- second-order perturbation theory,
- second-order theory with rate-equation populations,
- Plotly-based visualization of the resulting spectra.
The main workflow is based on a :class:`~spinsimulation.SpinSys.SpinSys` instance,
which is expected to contain the spin Hamiltonian, eigenstates, populations,
coupling parameters, and other system-specific quantities required for the
calculation.

Functions
---------
calc_IETS_2nd
    Compute dI/dV using a second-order tunneling model.
calc_IETS_2ndR
    Compute dI/dV using a second-order model with voltage-dependent rate equations.
calculate_derivative
    Numerically differentiate current with respect to voltage.
plot_IETS
    Plot IETS spectra for one or all spins using Plotly.
- Energies and voltages are treated in the conventions used by the surrounding
  spin-simulation package.
- Some functions modify attributes of the provided spin system in place, such as
  eigenstates, populations, or readout spin selection.
"""
import numpy as np
import plotly.colors
import plotly.graph_objects as go
import scipy.constants as const
from scipy.linalg import null_space

import spinsimulation.calcBath as bath
import spinsimulation.SpinHamiltonian as hamil
from spinsimulation.SpinSys import SpinSys

# region 2nd Order


def calc_IETS_2nd(
    spinsys: SpinSys,
    V_array: np.ndarray = None,
    norm: bool = False,
):
    """
    Calculate the derivative of the tunneling current (dI/dV) in 2nd Order.

    Parameters
    ----------
    spinsys : SpinSys
        The spin system object containing eigenvalues, eigenvectors, temperature, 
        and other relevant parameters.
    V_array : ndarray, optional
        An array of voltage values (in mV) at which to calculate dI/dV. 
        If None, a default range from -30 mV to 30 mV with 200 points will be used.
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
    - The function assumes that the spin system contains all necessary attributes.
    - Eigenvalues and eigenvectors are calculated if not already present.
    - The calculation uses a Boltzmann distribution for thermal population.
    - The tunneling rate matrix is computed using `calcTunnelingMatrixElements`.
    - The output can be normalized for easier comparison.
    """
    # Initialize the arrays
    # Default range from -30 mV to 30 mV with 200 points
    if V_array is None:
        V_array = np.linspace(-30, 30, 200)  
    N = len(V_array)
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
                ep = tunnelingBroadenedStepFunction(
                    np.array(
                        (spinsys.E_All[j] - spinsys.E_All[i] - V_array)
                        / (spinsys.T * spinsys.kB)
                    )
                )
                en = tunnelingBroadenedStepFunction(
                    np.array(
                        (spinsys.E_All[j] - spinsys.E_All[i] + V_array)
                        / (spinsys.T * spinsys.kB)
                    )
                )
                ytemp = (spinsys.pop_0[i] * (RateMatrix[j, i] 
                                             * ep + RateMatrix[i, j] * en))
                dIdV_array = dIdV_array + ytemp

    # Normalize with Rate_00 to get correct units
    Rate_0 = RateMatrix[0][0]  # Transition matrix element of the ground state
    dIdV_array = dIdV_array * spinsys.G_st / Rate_0
    
    # Normalize dI/dV if required
    if norm:
        dIdV_array = dIdV_array / dIdV_array[0]  # Normalize to the first value
    return V_array, dIdV_array


def tunnelingBroadenedStepFunction(x_V):
    """
    Compute the tunneling broadened step function.

    This function calculates a broadened step function used in tunneling spectroscopy.

    Parameters
    ----------
    x_V : array_like or float
        Input value(s) at which to evaluate the broadened step function.

    Returns
    -------
    y : ndarray or float
        The value(s) of the tunneling broadened step function at `x_V`. 
        If the computation results in NaN values, they are replaced with 0.0.
    """
    z = np.exp(x_V)
    y = (1 + (x_V - 1) * z) / ((z - 1) ** 2)
    y = np.nan_to_num(y, nan=0.0)
    return y

# endregion 2nd Order

# region 2nd Order + Rate


def calc_IETS_2ndR(
    spinsys: SpinSys,
    V_array: np.ndarray = None,
    AllowPumping: bool = True,
    norm: bool = True
):
    """
    Calculate the IETS spectrum using the rate equation approach.
    
    Parameters
    ----------
    spinsys : SpinSys
        The spin system object containing all relevant parameters and matrices.
    V_array : ndarray, optional
        An array of voltage values (in mV) at which to calculate dI/dV.
        If None, a default range from -30 mV to 30 mV with 200 points will be used.
    AllowPumping : bool, optional
        If True, allows for non-thermal population of spin states (pumping). 
        If False, uses thermal (Boltzmann) population. Default is True.
    norm : bool, optional
        If True, normalizes the resulting dI/dV spectrum to the range [0, 1]. 
        Default is True.
    
    Returns
    -------
    V_array : ndarray
        Array of voltage values (in mV) at which the spectrum is calculated.
    dIdV : ndarray
        The normalized differential conductance (dI/dV) spectrum.
    P : ndarray of shape (dimensionOfMatrix, N)
        Matrix of state populations for each voltage point.
    
    Notes
    -----
    - The calculation is based on the rate equation approach and supports both thermal
      and non-thermal population distributions.
    """
    # Initialize the arrays
    # Default range from -30 mV to 30 mV with 200 points
    if V_array is None:
        V_array = np.linspace(-30, 30, 200)  
    N = len(V_array)
    Current = np.zeros((N), dtype=float)
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

    # Rate-Factors using Loth et al. (2010), https://doi.org/10.1038/nphys1616
    spinsys.G_st = (1 - spinsys.b0) * spinsys.G_st
    RateFactor_ts = spinsys.G_st / (const.e * Rate_0) / 1e3
    RateFactor_st = spinsys.G_st / (const.e * Rate_0) / 1e3
    RateFactor_ss = 1 / (const.e * Rate_0) / 1e3
    # Gtt is calculated from Gss and Gst, the Gtt value 0,1 here allows it or not
    RateFactor_tt = (
    spinsys.G_st**2
    / spinsys.G_ss[spinsys.ReadoutSpin]
    / (const.e * Rate_0)
    / 1e3 * spinsys.G_tt
    )
    # Calculate the rates that stem from Tip tip and sample sample scattering,
    #  since they are not affected by the voltage
    RateIntegrals_ss = bath.calcRateIntegrals(spinsys, 'ss')
    RateIntegrals_tt = RateIntegrals_ss
    Rates[2, :, :] = RateFactor_ss * MatrixSS_summed * RateIntegrals_ss
    Rates[3, :, :] = RateFactor_tt * spinsys.MatrixTT * RateIntegrals_tt * spinsys.G_tt

    # This part happens now as often as we need it
    for I_index in range(N):
        V_DC = V_array[I_index]
        spinsys.V_DC = V_DC

        # Calculate changing rate integrals for the current voltage
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
            try:
                p = null_space(K)
            except Exception as e:
                print("Warning: null_space failed, using thermal population. Error:", e)
                p = np.empty((0, 0))
            if p.size == 0:  # If no valid solution, use thermal distribution
                Populations = spinsys.pop_0
                print("Warning: No valid solution for rate equations found, "
                "using thermal population instead.")
            else:
                Populations = p[:, 0] / np.sum(p[:, 0])
        else:
            Populations = spinsys.pop_0
            print("Using thermal population (Boltzmann distribution).")

        # Deriving the tunneling current using Equation 41 from Ternes (2015),
        # https://dx.doi.org/10.1088/1367-2630/17/6/063016
        I_value = 0
        for i in range(spinsys.dimensionOfMatrix):
            for j in range(spinsys.dimensionOfMatrix):
                I_temp = const.e * Populations[i] * (Rates[0, i, j] - Rates[1, i, j])
                I_value += I_temp

        # Adjust current calculation
        I_value += spinsys.b0 * spinsys.G_st * spinsys.V_DC

        Current[I_index] = I_value
        P[:, I_index] = Populations    

    # Calculate the derivative (dI/dV)
    dIdV = calculate_derivative(Current, (V_array * 1e-3))  # Convert V_array to volts

    # Normalize dI/dV if required
    if norm:
        dIdV = (dIdV / dIdV[0])  # Normalize to the first value

    return V_array, dIdV, P


def calculate_derivative(Current: list, Voltage: list):
    """
    Calculate the numerical derivative dI/dV of current (I) with respect to voltage (V).

    Parameters
    ----------
    Current : array_like
        Array of current values.
    Voltage : array_like
        Array of voltage values. Must be the same length as `Current`.
        
    Returns
    -------
    dIdV : ndarray
        Array of numerical derivatives dI/dV, same length as input arrays.

    Notes
    -----
    Assumes that `Current` and `Voltages` are one-dimensional arrays of equal length.
    """
    n = len(Current)  # Length of the array
    dIdV = np.zeros(n)  # Initialize the output array for derivative

    # Forward difference for the first point
    dIdV[0] = (Current[1] - Current[0]) / (Voltage[1] - Voltage[0])

    # Central difference for the middle points
    for i in range(1, n - 1):
        dIdV[i] = (Current[i + 1] - Current[i - 1]) / (Voltage[i + 1] - Voltage[i - 1])

    # Backward difference for the last point
    dIdV[n - 1] = (Current[n - 1] - Current[n - 2]) / (Voltage[n - 1] - Voltage[n - 2])

    return dIdV


# endregion 2nd Order + Rate

# region Visualization


def plot_IETS(
    spinsys: SpinSys,
    V_array: np.ndarray = None,
    norm: bool = False,
    order: str = '2nd',
    plotAllSpins: bool = False,
):
    """
    Plot the Inelastic Electron Tunneling Spectroscopy (IETS) spectrum.
    
    The function calculates the IETS spectrum for the given spin system 
    and visualizes it using Plotly. It supports both 2nd order and 2nd order
    with rate equations calculations.

    Parameters
    ----------
    spinsys : SpinSys
        The spin system object for which the IETS spectrum is to be calculated.
    V_array : numpy.ndarray, optional
        An array of voltage values (in mV) at which to calculate the spectrum.
        If None, a default range from -30 mV to 30 mV with 200 points will be used.
    norm : bool, optional
        If True, normalize the calculated spectrum (default is False).
    order : str, optional
        The order of the IETS calculation. Supported orders are '2nd' and '2ndR'.

    Returns
    -------
    return_dict : dict
        A dictionary containing:
        - 'fig': plotly.graph_objects.Figure
            The plotly figure object containing the plot.
        - 'V_array': numpy.ndarray
            The voltage values used in the spectrum.
        - 'dIdVsignal': numpy.ndarray
            The calculated dI/dV signal values.

    Raises
    ------
    ValueError
        If an unsupported order is specified.

    Notes
    -----
    - the dimension of the dIdVsignal changes when plotAllSpins=True
    """
    # Define y-axis label according to normalization
    y_label = 'Normalized dI/dV (a.u.)' if norm else 'dI/dV (S)'

    # Default to 200 points if V_array is None
    N = len(V_array) if V_array is not None else 200  

    if not plotAllSpins:

        # Initialize x and y arrays
        x = np.zeros(N)  # Voltage 
        y = np.zeros(N)  # dI/dV Signal
        # Calculate the IETS spectrum for the specified order
        if order == '2nd':
            x, y = calc_IETS_2nd(spinsys, V_array=V_array, norm=norm)
        elif order == '2ndR':
            x, y, _ = calc_IETS_2ndR(spinsys, V_array=V_array, norm=norm)
        else:
            raise ValueError("Unsupported order. Currently only '2nd'"
            "and '2ndR' order is implemented.")
    
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

        fig.update_xaxes(range=[x0, x1], showgrid=True,
                          gridcolor='lightgray', zeroline=False)
        fig.update_yaxes(range=[y0, y1], showgrid=True, 
                         gridcolor='lightgray', zeroline=False)

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

    elif plotAllSpins:

        # Initialize x and y arrays
        x = np.zeros(N)
        y = np.zeros((len(spinsys.Spins), N))

        for i in range(len(spinsys.Spins)):
            spinsys.ReadoutSpin = i
            
            if order == '2nd':
                x, y[i, :] = calc_IETS_2nd(spinsys, V_array=V_array, norm=norm)
            elif order == '2ndR':
                x, y[i, :], _ = calc_IETS_2ndR(spinsys, V_array=V_array, norm=norm)
            else:
                raise ValueError("Unsupported order. Currently only '2nd' "
                "and '2ndR' order are implemented.")
            
            # Plot all IETS spectra in the same figure on top of each other
            # Only create the figure once, before the loop
            # if i == 0:
        # After computing x and y for all spins, build a Plotly figure
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

        fig.update_xaxes(range=[x0, x1], showgrid=True, 
                         gridcolor='lightgray', zeroline=False)
        fig.update_yaxes(range=[y0, y1], showgrid=True, 
                         gridcolor='lightgray', zeroline=False)

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

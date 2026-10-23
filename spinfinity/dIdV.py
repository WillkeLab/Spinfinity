"""
Calculating and visualizing differential conductence (dI/dV) spectra.

This module provides routines to compute differential conductance spectra using
second-order perturbation theory.

Functions
---------
calc_IETS_2nd
    Compute dI/dV using a second-order tunneling model and thermally populated states.
tunnelingBroadenedStepFunction
    Compute the tunneling broadened step function used in the IETS calculations.
calc_IETS_2ndR
    Compute dI/dV using a second-order tunneling model with voltage-dependent
    rate equations.
calc_derivative
    Numerically differentiate current with respect to voltage.
plot_IETS
    Plot IETS spectra for one or all spins using Plotly.

Dependencies
------------
- Uses `spinfinity.SpinSys.SpinSys` class to represent the spin system
  and its properties.
- Uses `spinfinity.SpinHamiltonian` for eigenvalue and eigenvector calculations.
- Uses functions from `spinfinity.Bath` for population and rate calculations.
- Uses `spinfinity.dIdV3` for third-order calculations if specified.

References
----------
- Ternes, M., *Spin excitations and correlations in scanning tunneling spectroscopy*.
New J. Phys. 17, 063016 (2015).
[https://dx.doi.org/10.1088/1367-2630/17/6/063016](https://dx.doi.org/10.1088/1367-2630/17/6/063016)
- Loth, S., von Bergmann, K., Ternes, M. et al. *Controlling the state of quantum
spins with electric currents*. Nature Phys 6, 340–344 (2010).
[https://doi.org/10.1038/nphys1616](https://doi.org/10.1038/nphys1616)
"""

import numpy as np
import plotly.colors
import plotly.graph_objects as go
import scipy.constants as const
from scipy.linalg import null_space

import spinfinity.Bath as bath
import spinfinity.dIdV3 as dIdV3
import spinfinity.SpinHamiltonian as hamil
from spinfinity import plotting
from spinfinity.SpinSys import SpinSys

# region 2nd Order


def calc_IETS_2nd(
    spinsys: SpinSys,
    V_array: np.ndarray = None,
    norm: bool = False,
):
    """
    Calculate the derivative of the tunneling current (dI/dV) in 2nd Order.

    This functions calculates the dI/dV signal as a function of voltage using
    a second-order tunneling model with thermally populated states.

    Parameters
    ----------
    spinsys : SpinSys
        The spin system object containing eigenvalues, eigenvectors, temperature,
        and other relevant parameters.
    V_array : ndarray, optional
        An array of voltage values (in mV) at which to calculate dI/dV.
        If None, a default range from -30 mV to 30 mV with 200 points will be used.
    norm : bool, optional
        If True, normalizes the resulting dI/dV spectrum by dividing by the first value.
        Default is False.

    Returns
    -------
    V_array : ndarray
        Array of voltage values over the specified range.
    dIdV_array : ndarray
        Calculated dI/dV values corresponding to each voltage in V_array.
    """
    # Initialize the arrays
    # Default range from -30 mV to 30 mV with 200 points
    if V_array is None:
        V_array = np.linspace(-30, 30, 200)
    V_array = np.asarray(V_array, dtype=float)
    N = len(V_array)
    dIdV_array = np.zeros(N, dtype=float)

    # Calculate the Eignevalues and Eignevectors if not already done
    if not hasattr(spinsys, "eigVectors") or spinsys.eigVectors is None:
        hamil.build_Hamiltonian(spinsys)
        hamil.calc_EigEnergies(spinsys)

    # Calculate the thermal population (Boltzmann distribution)
    bath.calc_Populations(spinsys, AllowPumping=False)

    # Calculate the Rate Matrix
    RateMatrix = bath.calc_TunnelingMatrixElements(spinsys, "st")[0]

    for i in range(spinsys.dimensionOfMatrix):
        if spinsys.pop_0[i] > 10 ** (-6):
            for j in range(spinsys.dimensionOfMatrix):
                delta_E = spinsys.E_All[j] - spinsys.E_All[i]

                if spinsys.T == 0:
                    # Exact zero-temperature limit.
                    ep = np.heaviside(V_array - delta_E, 0.5)
                    en = np.heaviside(-V_array - delta_E, 0.5)
                else:
                    thermal_energy = spinsys.kB * spinsys.T

                    ep = tunnelingBroadenedStepFunction(
                        (delta_E - V_array) / thermal_energy
                    )
                    en = tunnelingBroadenedStepFunction(
                        (delta_E + V_array) / thermal_energy
                    )

                ytemp = spinsys.pop_0[i] * (
                    RateMatrix[j, i] * ep + RateMatrix[i, j] * en
                )
                dIdV_array += ytemp

    # Normalize with Rate_00 to get correct units
    Rate_0 = RateMatrix[0][0]  # Transition matrix element of the ground state
    # Adjust dIdV_array to account for the background contribution
    # G_st_prime is the effective tunneling conductance
    G_st_prime = (1 - spinsys.b0) * spinsys.G_st
    # spin dependent tunneling + background contribution
    dIdV_array = dIdV_array * G_st_prime / Rate_0 + spinsys.b0 * spinsys.G_st

    # Normalize dI/dV if required
    if norm:
        dIdV_array = dIdV_array / dIdV_array[0]  # Normalize to the first value
    return V_array, dIdV_array


def tunnelingBroadenedStepFunction(x):
    """
    Compute the tunneling broadened step function.

    This function calculates a broadened step function used in tunneling spectroscopy.

    Parameters
    ----------
    x : array_like or float
        Input value(s) at which to evaluate the broadened step function.

    Returns
    -------
    y : ndarray or float
        The value(s) of the tunneling broadened step function at `x_V`.
    """
    x = np.asarray(x, dtype=float)
    y = np.empty_like(x)

    finite = np.isfinite(x)

    # Well-defined infinite limits
    y[np.isneginf(x)] = 1.0
    y[np.isposinf(x)] = 0.0
    y[np.isnan(x)] = np.nan

    small = finite & (np.abs(x) < 1e-3)
    xs = x[small]

    y[small] = 0.5 - xs / 6.0 + xs**3 / 180.0 - xs**5 / 5040.0

    # Avoid overflow for large positive arguments
    large_positive = finite & (x >= 50.0)
    xp = x[large_positive]
    exp_minus_x = np.exp(-xp)

    y[large_positive] = (exp_minus_x**2 + (xp - 1.0) * exp_minus_x) / (
        1.0 - exp_minus_x
    ) ** 2

    # Avoid cancellation for large negative arguments
    large_negative = finite & (x <= -50.0)
    xn = x[large_negative]
    exp_x = np.exp(xn)

    y[large_negative] = (1.0 + (xn - 1.0) * exp_x) / (1.0 - exp_x) ** 2

    # Stable evaluation throughout the remaining range
    regular = finite & ~small & ~large_positive & ~large_negative
    xr = x[regular]
    expm1_x = np.expm1(xr)

    y[regular] = (xr + (xr - 1.0) * expm1_x) / expm1_x**2

    return y.item() if y.ndim == 0 else y


# endregion 2nd Order

# region 2nd Order + Rate


def calc_IETS_2ndR(
    spinsys: SpinSys,
    V_array: np.ndarray = None,
    AllowPumping: bool = True,
    norm: bool = True,
):
    """
    Calculate the IETS spectrum using the rate equation approach.

    This function calculates the dI/dV spectrum as a function of voltage using
    a second-order tunneling model combined with voltage-dependent rate equations that
    redistribute the spin state populations.

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
        If True, normalizes the resulting dI/dV spectrum by the first conduction value.
        Default is True.

    Returns
    -------
    V_array : ndarray
        Array of voltage values (in mV) at which the spectrum is calculated.
    dIdV : ndarray
        differential conductance (dI/dV) spectrum.
    P : ndarray of shape (dimensionOfMatrix, N)
        Matrix of state populations for each voltage point.

    Raises
    ------
    ValueError
        If tip-tip scattering is enabled and G_ss[ReadoutSpin] is not positive.

    Notes
    -----
    The calculation is based on the rate equation approach and supports both thermal
    and non-thermal population distributions. `For AllowPumping=False` the function
    should output the same result as `calc_IETS_2nd`.
    The function updates the following attributes of `spinsys`:

    - `spinsys.pop_0`: The thermal population distribution (Boltzmann distribution)
      for the spin states.
    - `spinsys.MatrixST`: The tunneling matrix elements for tip-sample scattering.
    - `spinsys.MatrixTS`: The tunneling matrix elements for sample-tip scattering.
    - `spinsys.MatrixSS`: The tunneling matrix elements for sample-sample scattering.
    - `spinsys.MatrixTT`: The tunneling matrix elements for tip-tip scattering.
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
    if not hasattr(spinsys, "eigVectors") or spinsys.eigVectors is None:
        hamil.build_Hamiltonian(spinsys)
        hamil.calc_EigEnergies(spinsys)

    # Getting the relevant matrix elements
    spinsys.MatrixST = bath.calc_TunnelingMatrixElements(spinsys, "st")[0]
    spinsys.MatrixTS = spinsys.MatrixST.T
    spinsys.MatrixSS = bath.calc_TunnelingMatrixElements(spinsys, "ss")
    spinsys.MatrixTT = bath.calc_TunnelingMatrixElements(spinsys, "tt")[0]

    MatrixSS_summed = np.zeros((spinsys.dimensionOfMatrix, spinsys.dimensionOfMatrix))
    for i in range(spinsys.NSpins):
        MatrixSS_summed += spinsys.MatrixSS[i, :, :] * spinsys.G_ss[i]

    Rate_0 = spinsys.MatrixST[0][0]  # Transition matrix element of the ground state

    # Rate-Factors using Loth et al. (2010), https://doi.org/10.1038/nphys1616
    # G_st_prime is the effective tunneling conductance
    G_st_prime = (1 - spinsys.b0) * spinsys.G_st
    RateFactor_ts = G_st_prime / (const.e * Rate_0) / 1e3
    RateFactor_st = G_st_prime / (const.e * Rate_0) / 1e3
    RateFactor_ss = 1 / (const.e * Rate_0) / 1e3
    # Gtt is calculated from Gss and Gst, the Gtt value 0,1 here allows it or not
    RateFactor_tt = 0.0
    if spinsys.G_tt:
        if spinsys.G_ss[spinsys.ReadoutSpin] <= 0:
            raise ValueError("G_tt requires positive G_ss for the readout spin.")
        RateFactor_tt = (
            G_st_prime**2
            / spinsys.G_ss[spinsys.ReadoutSpin]
            / (const.e * Rate_0)
            / 1e3
            * spinsys.G_tt
        )
    # Calculate the rates that stem from Tip tip and sample sample scattering,
    #  since they are not affected by the voltage
    RateIntegrals_ss = bath.calc_RateIntegrals(spinsys, "ss")
    RateIntegrals_tt = RateIntegrals_ss
    Rates[2, :, :] = RateFactor_ss * MatrixSS_summed * RateIntegrals_ss
    Rates[3, :, :] = RateFactor_tt * spinsys.MatrixTT * RateIntegrals_tt

    # Save the original voltage to restore it later
    original_V_DC = spinsys.V_DC

    # This part happens now as often as we need it
    for I_index in range(N):
        V_DC = V_array[I_index]
        spinsys.V_DC = V_DC

        # Calculate changing rate integrals for the current voltage
        RateIntegrals_ts = bath.calc_RateIntegrals(spinsys, "ts")
        RateIntegrals_st = bath.calc_RateIntegrals(spinsys, "st")

        # Calculate rates based on rate factors and integrals
        Rates[0, :, :] = RateFactor_ts * spinsys.MatrixTS * RateIntegrals_ts
        Rates[1, :, :] = RateFactor_st * spinsys.MatrixST * RateIntegrals_st

        # Sum the rates
        Rates_Summed = Rates[0, :, :] + Rates[1, :, :] + Rates[2, :, :] + Rates[3, :, :]

        # Build the Rate-Equation matrix
        K = np.zeros((spinsys.dimensionOfMatrix, spinsys.dimensionOfMatrix))
        for i in range(spinsys.dimensionOfMatrix):
            for j in range(spinsys.dimensionOfMatrix):
                if i != j:
                    K[i, i] -= Rates_Summed[i, j]  # Rates leaving state i
                    K[i, j] = Rates_Summed[j, i]  # Rates leading to state i

        # Calculate the thermal population (Boltzmann distribution)
        spinsys.pop_0 = bath.calc_ThermalPopulations(spinsys)

        # Solve the Rate-Equation if non-thermal population is desired
        if AllowPumping:
            try:
                p = null_space(K)
            except Exception as e:
                print("Warning: null_space failed, using thermal population. Error:", e)
                p = np.empty((0, 0))
            if p.size == 0:  # If no valid solution, use thermal distribution
                Populations = spinsys.pop_0
                print(
                    "Warning: No valid solution for rate equations found, "
                    "using thermal population instead."
                )
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

        # Adjust current calculation by adding the elastic contribution
        I_value += spinsys.b0 * spinsys.G_st * spinsys.V_DC * 1e-3

        Current[I_index] = I_value
        P[:, I_index] = Populations

    # Calculate the derivative (dI/dV)
    dIdV = calc_derivative(Current, (V_array * 1e-3))  # Convert V_array to volts

    # Normalize dI/dV if required
    if norm:
        dIdV = dIdV / dIdV[0]  # Normalize to the first value

    # Restore the original voltage
    spinsys.V_DC = original_V_DC

    return V_array, dIdV, P


def calc_derivative(Current: list, Voltage: list):
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
    order: str = "2nd",
    plotAllSpins: bool = False,
    show: str | None = None,
):
    """
    Plot the Inelastic Electron Tunneling Spectroscopy (IETS) spectrum.

    Calculate and visualize an IETS spectrum using Plotly. Supports
    second- and third-order calculations with thermal populations or
    stationary rate equations.

    Parameters
    ----------
    spinsys : SpinSys
        The spin system object for which the IETS spectrum is to be calculated.
    V_array : numpy.ndarray, optional
        An array of voltage values (in mV) at which to calculate the spectrum.
        If None, a default range from -30 mV to 30 mV with 200 points will be used.
    norm : bool, optional
        Divide each spectrum by its first conductance value if True.
        Default is False.
    order : str, optional
        Calculation mode: '2nd', '2ndR', '3rd' or '3rdR'.
        Modes ending in 'R' use stationary rate equations.
        Default is '2nd'.
    plotAllSpins : bool, optional
        Calculate a spectrum for each readout spin if True.
        Default is False.

    show : {"fixed", "responsive", "no"} or None, optional
        Display mode. ``None`` uses the global setting, initially ``"fixed"``.
        ``"fixed"`` uses the preferred width; ``"responsive"`` fills the
        container width. ``"no"`` suppresses display without skipping calculations.

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
    y_label = "Normalized dI/dV (a.u.)" if norm else "dI/dV (S)"

    # Default to 200 points if V_array is None
    N = len(V_array) if V_array is not None else 200

    if not plotAllSpins:
        # Initialize x and y arrays
        x = np.zeros(N)  # Voltage
        y = np.zeros(N)  # dI/dV Signal
        # Calculate the IETS spectrum for the specified order
        if order == "2nd":
            x, y = calc_IETS_2nd(spinsys, V_array=V_array, norm=norm)
        elif order == "2ndR":
            x, y, _ = calc_IETS_2ndR(spinsys, V_array=V_array, norm=norm)
        elif order == "3rd":
            x, y = dIdV3.calc_IETS_3rd(spinsys, V_array=V_array, norm=norm)
        elif order == "3rdR":
            x, y, _ = dIdV3.calc_IETS_3rdR(spinsys, V_array=V_array, norm=norm)
        else:
            raise ValueError(
                "Unsupported order. Currently only '2nd', '2ndR', '3rd',"
                "and '3rdR' order is implemented."
            )

        # Plot the IETS spectrum using Plotly with white background and black border
        y_max = np.max(y)
        if y_max == 0:
            y_max = 1.0
        x0, x1 = float(np.min(x)), float(np.max(x))
        y0, y1 = 0.0, float(y_max * 1.1)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                name="IETS Spectrum",
                line=dict(color="black", width=2),
            )
        )

        fig.update_layout(
            title=f"IETS Spectrum ({order} Order)",
            xaxis_title="Bias Voltage (mV)",
            yaxis_title=y_label,
            plot_bgcolor="white",
            paper_bgcolor="white",
            margin=dict(l=60, r=20, t=60, b=60),
        )

        fig.update_xaxes(
            range=[x0, x1], showgrid=True, gridcolor="lightgray", zeroline=False
        )
        fig.update_yaxes(
            range=[y0, y1], showgrid=True, gridcolor="lightgray", zeroline=False
        )

        fig.update_layout(
            shapes=[
                dict(
                    type="rect",
                    xref="x",
                    yref="y",
                    x0=x0,
                    x1=x1,
                    y0=y0,
                    y1=y1,
                    line=dict(color="black", width=2),
                    fillcolor="rgba(0,0,0,0)",
                )
            ]
        )

        plotting.show_figure(fig, preferred_width=800, show=show)

    elif plotAllSpins:
        # Initialize x and y arrays
        x = np.zeros(N)
        y = np.zeros((len(spinsys.Spins), N))

        for i in range(len(spinsys.Spins)):
            spinsys.ReadoutSpin = i

            if order == "2nd":
                x, y[i, :] = calc_IETS_2nd(spinsys, V_array=V_array, norm=norm)
            elif order == "2ndR":
                x, y[i, :], _ = calc_IETS_2ndR(spinsys, V_array=V_array, norm=norm)
            elif order == "3rd":
                x, y[i, :] = dIdV3.calc_IETS_3rd(spinsys, V_array=V_array, norm=norm)
            elif order == "3rdR":
                x, y[i, :], _ = dIdV3.calc_IETS_3rdR(
                    spinsys, V_array=V_array, norm=norm
                )
            else:
                raise ValueError(
                    "Unsupported order. Use '2nd', '2ndR', '3rd' or '3rdR'."
                )

            # Plot all IETS spectra in the same figure on top of each other
            # Only create the figure once, before the loop
            # if i == 0:
        # After computing x and y for all spins, build a Plotly figure
        color_list = plotly.colors.qualitative.Plotly
        fig = go.Figure()

        # Add one trace per spin (overlayed)
        nspins = y.shape[0]
        for i in range(nspins):
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y[i, :],
                    mode="lines",
                    line=dict(color=color_list[i % len(color_list)], width=2),
                    name=f"Spin {i}",
                )
            )

        # Determine axis ranges similar to single-spin branch
        y_max = np.max(y) if np.max(y) != 0 else 1.0
        x0, x1 = float(np.min(x)), float(np.max(x))
        y0, y1 = 0.0, float(y_max * 1.1)

        fig.update_layout(
            title=f"IETS Spectrum ({order} Order) for All Spins",
            xaxis_title="Bias Voltage (mV)",
            yaxis_title=y_label,
            plot_bgcolor="white",
            paper_bgcolor="white",
            margin=dict(l=60, r=20, t=60, b=60),
        )

        fig.update_xaxes(
            range=[x0, x1], showgrid=True, gridcolor="lightgray", zeroline=False
        )
        fig.update_yaxes(
            range=[y0, y1], showgrid=True, gridcolor="lightgray", zeroline=False
        )

        # add a black rectangle border around the plotting area
        fig.update_layout(
            shapes=[
                dict(
                    type="rect",
                    xref="x",
                    yref="y",
                    x0=x0,
                    x1=x1,
                    y0=y0,
                    y1=y1,
                    line=dict(color="black", width=2),
                    fillcolor="rgba(0,0,0,0)",
                )
            ]
        )

        plotting.show_figure(fig, preferred_width=800, show=show)

    return_dict = {"fig": fig, "V_array": x, "dIdVsignal": y}

    return return_dict


# endregion Visualization

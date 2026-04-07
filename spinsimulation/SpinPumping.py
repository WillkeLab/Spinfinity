import numpy as np
import plotly.colors
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import spinsimulation.calcBath as bath
import spinsimulation.SpinHamiltonian as hamil
from spinsimulation.SpinSys import SpinSys


# region Calculations
def split_Populations(spinsys: SpinSys, subspin: SpinSys):
    """
    Split the populations of a spin system into two subpopulations based on overlaps with subspin eigenvectors.

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
    This function assumes that `calcRates`, `calcPopulations`, and `calcOverlap` are defined elsewhere.
    The function modifies the populations of `spinsys` in-place before projection.
    """
    # First we calculate the Rates and Populations of the whole system
    # hamil.calcEigEnergies(subspin)
    # hamil.calcEigEnergies(spinsys)
    # hamil.calcEigStates(spinsys)
    # hamil.calcEigStates(subspin)
    bath.calcRates(spinsys)
    bath.calcPopulations(spinsys, AllowPumping=True)

    # Initialize 
    Populations_A = np.zeros(subspin.dimensionOfMatrix)
    Populations_B = np.zeros(subspin.dimensionOfMatrix)

    for idx in range(subspin.dimensionOfMatrix):
        overlap_A = calcOverlap(spinsys, subspin.dimensionOfMatrix, subspin.dimensionOfMatrix, subspin.eigVectors[:, idx], order='BA')
        overlap_B = calcOverlap(spinsys, subspin.dimensionOfMatrix, subspin.dimensionOfMatrix, subspin.eigVectors[:, idx], order='AB')
        Populations_A[idx] = sum(overlap_A * spinsys.Populations)
        Populations_B[idx] = sum(overlap_B * spinsys.Populations)

    return Populations_A, Populations_B


def split_Magnetization(spinsys: SpinSys, subspin: SpinSys, type: str = 'z'):
    """
    Compute magnetization contributions from two population partitions of a subsystem.

    This function splits the populations of ``spinsys`` into two parts (A and B)
    with respect to ``subspin``, then computes the total magnetization of
    ``subspin`` for each partition independently.

    Parameters
    ----------
    spinsys : SpinSys
        Full spin system used as the source for population splitting.
    subspin : SpinSys
        Subsystem for which magnetization is evaluated. Its ``Populations``
        attribute is overwritten during execution.
    type : str, default='z'
        Magnetization component/type passed to ``calcMagnetization``.

    Returns
    -------
    Mag_A : float or complex
        Total magnetization of ``subspin`` after assigning partition A populations.
    Mag_B : float or complex
        Total magnetization of ``subspin`` after assigning partition B populations.

    Notes
    -----
    This function mutates ``subspin.Populations`` as part of the calculation.
    After return, ``subspin.Populations`` remains set to the B partition.
    """    
    # Calculate first the sub populations
    pA, pB = split_Populations(spinsys=spinsys, subspin=subspin)

    # A
    subspin.Populations = pA
    bath.calcMagnetization(subspin, type=type, AllowPumping=True)
    Mag_A = subspin.Mag_tot

    # B
    subspin.Populations = pB
    bath.calcMagnetization(subspin, type=type, AllowPumping=True)
    Mag_B = subspin.Mag_tot

    return Mag_A, Mag_B


def calcOverlap(spinsys: SpinSys, nA: int, nB: int, stateB: list, order='AB'):
    """
    Calculate the overlap between a given state and the eigenvectors of a spin system.

    Parameters
    ----------
    spinsys : SpinSys
        The spin system object containing eigenvectors as an attribute `eigVectors`.
    nA : int
        Dimension of subsystem A.
    nB : int
        Dimension of subsystem B.
    stateB : list or array-like
        State vector for subsystem B.
    order : {'AB', 'BA'}, optional
        Specifies the ordering of the subsystems in the eigenvector reshaping.
        If 'AB', eigenvectors are reshaped as (nA, nB, -1).
        If 'BA', eigenvectors are reshaped as (nB, nA, -1).
        Default is 'AB'.

    Returns
    -------
    O : numpy.ndarray
        Array of overlaps between the projected state and the eigenvectors, shape (n,).

    Notes
    -----
    This function projects the eigenvectors of the composite system onto a given state
    of subsystem B and computes the overlap for each eigenvector.
    """
    EV = np.asarray(spinsys.eigVectors)
    stateB = np.asarray(stateB).reshape(-1)
    if order.upper() == 'AB':
        V = EV.reshape(nA, nB, -1)               # (dA, dB, n)
        PhiA = np.tensordot(V, np.conj(stateB), axes=([1], [0]))  # (dA, n)
    else:
        V = EV.reshape(nB, nA, -1)               # (dB, dA, n)
        PhiA = np.tensordot(V.transpose(1, 0, 2), np.conj(stateB), axes=([1], [0]))  # (dA, n)
    O = np.einsum('in,in->n', np.conj(PhiA), PhiA).real
    
    return O


# endregion Calculations

# region Visualization
def plotMagnetization_Voltage(spinsys: SpinSys, Vrange: list = [-200, 200], N: int = 100, type: str = 'z', spins: list = [0], norm: bool = True, total: bool = True):
    """
    Plot the magnetization of specified spins and the total magnetization as a function of voltage.

    Parameters
    ----------
    spinsys : SpinSys
        The spin system object containing the spin configuration and state.
    Vrange : list of float, optional
        The voltage range [V_min, V_max] (in mV) over which to sweep. Default is [-200, 200].
    N : int, optional
        Number of voltage points in the sweep. Default is 100.
    type : str, optional
        The type of magnetization component to calculate ('x', 'y', or 'z'). Default is 'z'.
    spins : list of int, optional
        List of spin indices to plot. Default is [0].
    norm : bool, optional
        If True, normalize the magnetization values. Default is True.
    total : bool, optional
        If True, plot the total magnetization. Default is True.

    Returns
    -------
    return_dict : dict
        A dictionary containing:
        - 'fig': plotly.graph_objects.Figure
            The plotly figure object containing the ESR spectrum plot.
        - 'V_array': numpy.ndarray
            The array of voltage points in mV.
        - 'M_array': numpy.ndarray
            The calculated Magnetization for each spin, the last column is the total magnetization.

    Notes
    -----
    The function restores its original value after plotting. 
    """
    # Save the old voltage value
    V_initial = spinsys.V_DC

    if len(spins) > spinsys.NSpins:
        print("Error: Too many spins specified in the 'spin' argument.")
        return
    
    # Initialize arrays to store voltage and magnetization data
    V_array = np.linspace(Vrange[0], Vrange[1], N)
    M_array = np.zeros((N, spinsys.NSpins + 1)) 

    for idx, V in enumerate(V_array):
        spinsys.V_DC = V
        bath.calcRates(spinsys)
        bath.calcPopulations(spinsys, AllowPumping=True)
        bath.calcMagnetization(spinsys, type=type, AllowPumping=True)
        for i in range(spinsys.NSpins):
            if norm:
                M_array[idx, i] = spinsys.Mag[i] / spinsys.Mag_max[i]
            else:
                M_array[idx, i] = spinsys.Mag[i]
        
        if norm:
            M_array[idx, -1] = spinsys.Mag_tot / spinsys.Mag_tot_max
        else:
            M_array[idx, -1] = spinsys.Mag_tot

    # Plotting
    fig = go.Figure()
    color_list = plotly.colors.qualitative.Plotly

    for idx, spin in enumerate(spins):
        fig.add_trace(go.Scatter(
            x=V_array,
            y=M_array[:, spin],
            mode='lines',
            name=f'Spin {spin} Magnetization',
            line=dict(color=color_list[idx % len(color_list)], width=2)
        ))

    if total:
        fig.add_trace(go.Scatter(
            x=V_array,
            y=M_array[:, -1],
            mode='lines',
            name='Total Magnetization',
            line=dict(color='red', width=2)
        ))

    if norm:
        tp = float(spinsys.TipPolarization[2])
        # Tip polarization lines (positive and negative). keep second line out of legend to match original behavior.
        fig.add_trace(go.Scatter(
            x=[V_array[0], V_array[-1]],
            y=[tp, tp],
            mode='lines',
            name='Tip Polarization',
            line=dict(color='black', dash='dash'),
            opacity=0.5
        ))
        fig.add_trace(go.Scatter(
            x=[V_array[0], V_array[-1]],
            y=[-tp, -tp],
            mode='lines',
            name='',
            line=dict(color='black', dash='dash'),
            opacity=0.5,
            showlegend=False
        ))

    fig.update_layout(
        title='Magnetization vs Voltage',
        xaxis_title='Voltage (mV)',
        yaxis_title='Magnetization',
        plot_bgcolor='white',
        paper_bgcolor='white',
        width=800,
        height=500,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        margin=dict(l=60, r=20, t=60, b=60)
    )

    # axis styling
    fig.update_xaxes(range=[float(V_array[0]), float(V_array[-1])], showgrid=True, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridcolor='lightgray')
    if norm:
        fig.update_yaxes(range=[-1, 1])

    fig.show()
    
    # Restore the initial voltage value
    spinsys.V_DC = V_initial
    bath.calcRates(spinsys)
    bath.calcPopulations(spinsys, AllowPumping=True)
    bath.calcMagnetization(spinsys, type=type, AllowPumping=True)
    
    return_dict = {
        'fig': fig,
        'V_array': V_array,
        'M_array': M_array}

    return return_dict


def plotPopulations_Voltage(
        spinsys: SpinSys,
        Vrange: list = [-200, 200],
        N: int = 200):
    """
    Plot the populations of spin states as a function of applied voltage.
    
    Parameters
    ----------
    spinsys : SpinSys
        The spin system object containing the current state and parameters.
    Vrange : list, optional
        The voltage range [V_min, V_max] (in mV) over which to sweep. Default is [-200, 200].
    N : int, optional
        The number of voltage points to evaluate. Default is 200.

    Returns
    -------
    return_dict : dict
        A dictionary containing:
        - 'fig': plotly.graph_objects.Figure
            The plotly figure object containing the ESR spectrum plot.
        - 'V_array': numpy.ndarray
            The array of voltage points in mV.
        - 'P_array': numpy.ndarray
            The calculated Populations for each spin state at each voltage. 

    Notes
    -----
    - The function restores the original values at the end.
    """
    # Save the old voltage value
    V_initial = spinsys.V_DC

    # Initialize arrays to store voltage and population data
    V_array = np.linspace(Vrange[0], Vrange[1], N)
    P_array = np.zeros((N, spinsys.dimensionOfMatrix))
    for idx, V in enumerate(V_array):
        spinsys.V_DC = V
        bath.calcRates(spinsys)
        bath.calcPopulations(spinsys, AllowPumping=True)
        P_array[idx, :] = spinsys.Populations

    # Plotting
    fig = go.Figure()
    color_list = plotly.colors.qualitative.Plotly
    for idx in range(spinsys.dimensionOfMatrix):
        fig.add_trace(go.Scatter(
            x=V_array,
            y=P_array[:, idx],
            mode='lines',
            name=f'State {idx}',
            line=dict(color=color_list[idx % len(color_list)], width=2)
        ))

    fig.update_layout(
        title='Populations vs Voltage',
        xaxis_title='Voltage (mV)',
        yaxis_title='Populations',
        plot_bgcolor='white',
        paper_bgcolor='white',
        width=800,
        height=500,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        margin=dict(l=60, r=20, t=60, b=60)
    )

    fig.update_xaxes(showgrid=True, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridcolor='lightgray', type='log')

    fig.show()
    
    # Restore the initial voltage value
    spinsys.V_DC = V_initial
    bath.calcRates(spinsys)
    bath.calcPopulations(spinsys, AllowPumping=True)

    return_dict = {
        'fig': fig,
        'V_array': V_array,
        'P_array': P_array}

    return return_dict


def plotRates_Voltage(spinsys: SpinSys, Vrange: list = [-200, 200], N: int = 100):
    """
    Plot the transition rates between spin states as a function of applied voltage.
    
    Parameters
    ----------
    spinsys : SpinSys
        The spin system object containing the current state and parameters, including the
        voltage (`V_DC`) and the rates matrix (`Rates_Summed`).
    Vrange : list of float, optional
        The voltage range [V_min, V_max] (in mV) over which to sweep. Default is [-200, 200].
    N : int, optional
        The number of voltage points to evaluate. Default is 100.

    Returns
    -------
    return_dict : dict
        A dictionary containing:
        - 'fig': plotly.graph_objects.Figure
            The plotly figure object containing the ESR spectrum plot.
        - 'V_array': numpy.ndarray
            The array of voltage points in mV.
        - 'R_array': numpy.ndarray
            The calculated Rates for each transition (i → f) at each voltage.
        
    Notes
    -----
    - The function temporarily modifies `spinsys.V_DC` during the sweep and restores its original
      value at the end.
    - Only off-diagonal transition rates (i ≠ f) are plotted.
    """
    # Save the old voltage value
    V_initial = spinsys.V_DC

    # Initialize arrays to store voltage and rate data
    V_array = np.linspace(Vrange[0], Vrange[1], N)
    R_array = np.zeros((N, spinsys.dimensionOfMatrix, spinsys.dimensionOfMatrix))
    
    for idx, V in enumerate(V_array):
        spinsys.V_DC = V
        bath.calcRates(spinsys)
        for i in range(spinsys.dimensionOfMatrix):
            for f in range(spinsys.dimensionOfMatrix):
                if i != f:
                    R_array[idx, i, f] = spinsys.Rates_Summed[i, f]

    # Plotting
    # Plot with Plotly (white background)
    fig = go.Figure()
    color_list = plotly.colors.qualitative.Plotly
    trace_idx = 0
    for i in range(spinsys.dimensionOfMatrix):
        for f in range(spinsys.dimensionOfMatrix):
            if i != f:
                y = R_array[:, i, f].copy()
                # avoid plotting non-positive values on a log scale
                y = np.where(y > 0, y, np.nan)
                fig.add_trace(go.Scatter(
                    x=V_array,
                    y=y,
                    mode='lines',
                    name=f'Rate {i}→{f}',
                    line=dict(color=color_list[trace_idx % len(color_list)], width=2),
                    hovertemplate='Voltage: %{x}<br>Rate: %{y}<extra></extra>'
                ))
                trace_idx += 1

    fig.update_layout(
        title='Transition Rates vs Voltage',
        xaxis_title='Voltage (mV)',
        yaxis_title='Transition Rates',
        plot_bgcolor='white',
        paper_bgcolor='white',
        width=900,
        height=500,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    fig.update_xaxes(showgrid=True, gridcolor='lightgray')
    fig.update_yaxes(type='log', showgrid=True, gridcolor='lightgray')
    fig.show()

    # Restore the initial voltage value
    spinsys.V_DC = V_initial
    bath.calcRates(spinsys)

    return_dict = {
        'fig': fig,
        'V_array': V_array,
        'R_array': R_array}

    return return_dict


def plotSplitMagnetization_Voltage(spinsys: SpinSys, subspin: SpinSys, Vrange: list = [-200, 200], N: int = 100, type: str = 'z', normVal: float = None):
    """
    Plot the split magnetization of two sub-systems as a function of DC bias voltage.

    This function sweeps the DC bias voltage of the provided SpinSys object across a
    specified range, computes the split magnetization for the two sub-systems using
    split_Magnetization

    Parameters
    ----------
    spinsys : SpinSys
        The main spin system whose V_DC attribute will be varied during the sweep.
    subspin : SpinSys
        The sub-system passed to split_Magnetization to compute the split
        magnetizations.
    Vrange : list, optional
        Two-element sequence [Vmin, Vmax] specifying the voltage sweep range in
        millivolts (mV). Default is [-200, 200].
    N : int, optional
        Number of voltage points to evaluate in the sweep (linearly spaced).
        Default is 100.
    type : str, optional
        Component/type of magnetization to compute (passed through to
        split_Magnetization). Default is 'z'.
    normVal : float or None, optional
        If provided, both computed magnetization arrays (Mag_A and Mag_B) are
        normalized by dividing by this value. If None, no normalization is applied.

    Returns
    -------
    dict
        A dictionary with the following keys:
        - 'fig': plotly.graph_objects.Figure
            The Plotly figure containing the magnetization vs voltage traces.
            The figure is shown (fig.show()) before the function returns.
        - 'V_array': numpy.ndarray
            The 1D array of voltage values (length N) used for the sweep (units mV).
        - 'Mag_A': numpy.ndarray
            Computed magnetization values for subsystem A corresponding to V_array.
        - 'Mag_B': numpy.ndarray
            Computed magnetization values for subsystem B corresponding to V_array.

    Notes
    -----
    - The initial values are restored after the sweep.
    """
    # Save the old voltage value
    V_initial = spinsys.V_DC

    # Initialize arrays to store voltage and rate data
    V_array = np.linspace(Vrange[0], Vrange[1], N)
    Mag_A = np.zeros(N)
    Mag_B = np.zeros(N)

    for idx, V in enumerate(V_array):
        spinsys.V_DC = V
        mA, mB = split_Magnetization(spinsys, subspin, type=type)
        Mag_A[idx] = mA
        Mag_B[idx] = mB
    
    if normVal is not None:
        Mag_A = Mag_A / normVal
        Mag_B = Mag_B / normVal

    # Plotting with Plotly (white background)
    color_list = plotly.colors.qualitative.Plotly
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=V_array,
        y=Mag_A,
        mode='lines',
        name='Magnetization A',
        line=dict(color=color_list[0 % len(color_list)], width=2)
    ))
    fig.add_trace(go.Scatter(
        x=V_array,
        y=Mag_B,
        mode='lines',
        name='Magnetization B',
        line=dict(color=color_list[1 % len(color_list)], width=2)
    ))

    fig.update_layout(
        title='Split Magnetization vs Voltage',
        xaxis_title='Voltage (mV)',
        yaxis_title='Magnetization',
        plot_bgcolor='white',
        paper_bgcolor='white',
        width=800,
        height=500,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        margin=dict(l=60, r=20, t=60, b=60)
    )
    fig.update_xaxes(showgrid=True, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridcolor='lightgray')

    fig.show()

    # Restore the initial voltage value
    spinsys.V_DC = V_initial
    bath.calcRates(spinsys)
    bath.calcPopulations(spinsys, AllowPumping=True)

    return_dict = {
        'fig': fig,
        'V_array': V_array,
        'Mag_A': Mag_A,
        'Mag_B': Mag_B}

    return return_dict


def plotSplitPopulations_Voltage(spinsys: SpinSys, subspin: SpinSys, Vrange: list = [-200, 200], N: int = 100):
    """
    Plot split populations as a function of voltage for a given spin subsystem.

    This function sweeps the system voltage over a specified range, computes the split
    populations for the provided subspin at each voltage (using split_Populations).

    Parameters
    ----------
    spinsys : SpinSys
        Main spin system object whose V_DC is swept. The function temporarily sets
        spinsys.V_DC and restores it on exit.
    subspin : SpinSys
        Subsystem for which eigenenergies/eigenstates are computed and populations
        are evaluated.
    Vrange : list, optional
        Two-element list [V_min, V_max] specifying the voltage sweep range in mV.
        Default is [-200, 200].
    N : int, optional
        Number of voltage points in the sweep. Default is 100.

    Returns
    -------
    dict
        A dictionary with the following keys:
        - 'fig': plotly.graph_objs.Figure
            Plotly figure containing two subplots (Populations A and B vs Voltage).
        - 'V_array': ndarray
            1D array of voltages used in the sweep (length N).
        - 'Populations_A': ndarray
            Array of shape (N, subspin.dimensionOfMatrix) containing populations for A.
        - 'Populations_B': ndarray
            Array of shape (N, subspin.dimensionOfMatrix) containing populations for B.

    Notes
    -----
    - After restoring the initial voltage it calls calcRates(spinsys) and
      calcPopulations(spinsys, AllowPumping=True).
    """
    # Save the old voltage value
    V_initial = spinsys.V_DC

    hamil.calcEigEnergies(subspin)
    hamil.calcEigStates(subspin)

    # Initialize arrays to store voltage and rate data
    V_array = np.linspace(Vrange[0], Vrange[1], N)
    Populations_A = np.zeros((N, subspin.dimensionOfMatrix))
    Populations_B = np.zeros((N, subspin.dimensionOfMatrix))

    for idx, V in enumerate(V_array):
        spinsys.V_DC = V
        pA, pB = split_Populations(spinsys, subspin)
        Populations_A[idx, :] = pA
        Populations_B[idx, :] = pB

    # Prepare plotly subplot figure with white background
    color_list = plotly.colors.qualitative.Plotly
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True,
                        subplot_titles=('Populations A vs Voltage', 'Populations B vs Voltage'))

    for idx in range(subspin.dimensionOfMatrix):
        color = color_list[idx % len(color_list)]
        label = f'{subspin.statesWithoutE[idx]}'
        fig.add_trace(
            go.Scatter(x=V_array, y=Populations_A[:, idx], mode='lines', name=label,
                       line=dict(color=color)),
            row=1, col=1
        )
        # hide duplicate legend entries for the right subplot
        fig.add_trace(
            go.Scatter(x=V_array, y=Populations_B[:, idx], mode='lines', name=label,
                       line=dict(color=color), showlegend=False),
            row=1, col=2
        )

    # Layout styling: white background, axis labels, limits, legend
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        width=1000,
        height=480,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    fig.update_xaxes(title_text='Voltage (mV)', row=1, col=1)
    fig.update_xaxes(title_text='Voltage (mV)', row=1, col=2)
    fig.update_yaxes(title_text='Population', range=[0, 1], row=1, col=1)

    fig.show()

    # Restore the initial voltage value
    spinsys.V_DC = V_initial
    bath.calcRates(spinsys)
    bath.calcPopulations(spinsys, AllowPumping=True)

    return_dict = {
        'fig': fig,
        'V_array': V_array,
        'Populations_A': Populations_A,
        'Populations_B': Populations_B}

    return return_dict


def plotSplitPopulations(spinsys: SpinSys, subspin: SpinSys):
    """
    Plot horizontal bar charts of split populations for two sub-spins using Plotly.

    This function computes the population distributions for two parts (A and B) of
    a given spin subsystem by calling split_Populations(spinsys, subspin).

    Parameters
    ----------
    spinsys : SpinSys
        The full spin system object from which split populations are computed.
        Passed to split_Populations along with `subspin`.
    subspin : SpinSys
        A SpinSys object describing the subsystem to visualize. Must provide:
        - dimensionOfMatrix (int): length of returned population arrays.
        - statesWithoutE (sequence of str): labels for each state used as y-axis tick
          labels.

    Returns
    -------
    dict
        A dictionary with the following keys:
        - 'fig' : plotly.graph_objs._figure.Figure
            The Plotly Figure object containing two horizontal bar subplots.
        - 'Populations_A' : numpy.ndarray
            1D array of length `subspin.dimensionOfMatrix` with populations for spin A.
        - 'Populations_B' : numpy.ndarray
            1D array of length `subspin.dimensionOfMatrix` with populations for spin B.

    Raises
    ------
    ValueError
        If the arrays returned by split_Populations do not match
        `subspin.dimensionOfMatrix` in length.
    RuntimeError
        If required plotting dependencies (Plotly) or the split_Populations call
        fail.
    """
    Populations_A = np.zeros(subspin.dimensionOfMatrix)
    Populations_B = np.zeros(subspin.dimensionOfMatrix)
    
    Populations_A, Populations_B = split_Populations(spinsys, subspin)
    
    x = np.arange(subspin.dimensionOfMatrix)

    # Plot with Plotly, white background
    y_labels = subspin.statesWithoutE
    colors = plotly.colors.qualitative.Plotly
    col_a = colors[0] if len(colors) > 0 else 'blue'
    col_b = colors[1] if len(colors) > 1 else 'green'

    fig = make_subplots(rows=1, cols=2, shared_yaxes=True,
                        subplot_titles=('Population of Spin A', 'Population of Spin B'),
                        horizontal_spacing=0.08)

    fig.add_trace(go.Bar(
        x=Populations_A,
        y=y_labels,
        orientation='h',
        name='Populations A',
        marker=dict(color=col_a),
        opacity=0.6,
        showlegend=True
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=Populations_B,
        y=y_labels,
        orientation='h',
        name='Populations B',
        marker=dict(color=col_b),
        opacity=0.6,
        showlegend=True
    ), row=1, col=2)

    # layout & styling with white background
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        width=900,
        height=500,
        margin=dict(l=120, r=40, t=60, b=40),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )

    # x-axis ranges [0,1] and gridlines
    fig.update_xaxes(range=[0, 1], showgrid=True, gridcolor='lightgray', row=1, col=1, title_text='Population')
    fig.update_xaxes(range=[0, 1], showgrid=True, gridcolor='lightgray', row=1, col=2, title_text='Population')

    # show y-labels only on the left subplot
    fig.update_yaxes(title_text='States', row=1, col=1)
    fig.update_yaxes(showticklabels=False, row=1, col=2)

    fig.show()

    return_dict = {
        'fig': fig,
        'Populations_A': Populations_A,
        'Populations_B': Populations_B}

    return return_dict

# endregion Visualization
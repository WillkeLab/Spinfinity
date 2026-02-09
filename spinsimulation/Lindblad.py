# import python libraries
import numpy as np
import scipy
import scipy.constants as const
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.colors
from plotly.subplots import make_subplots

# import other SpinFinity packages
from spinsimulation.SpinSys import SpinSys
import spinsimulation.SpinHamiltonian as hamil 
import spinsimulation.calcBath as bath

# import qutip code
from qutip import Qobj, mesolve
from qutip.solver.result import Result as QuTiPResult  # to be able to make sure that entities are of the qutip result class


# Content

# - calculate collapse operators with off diagonal elements
# - Filter rates and kick out values below a threshhold
# - Translator into QuTiP objects 
# -   
# - Lindblad master equation solver for spin systems
# (- Current from density matrix)
# -  

# TODO matplotlib to Plotly
# TODO range and number of points instead of linspace as arg


def test():
    """
    To test if Lindblad package was successfully imported
    """
    print("hello!")

# region Lindblad solving
def collapseOperators(spinsys: SpinSys, threshold_percent: float = 0.01):
    """
    Builds Lindblad relaxation operators from the system's rates. 
    Adds attribute collapse_ops to spinsys.

    Parameters
    ----------
    spinsys : SpinSys
        The spin system object containing eigenvectors, spin operators, and populations.

    threshold_percent : float, Optional
        Threshold in [0, 1] for filtering rates. If a rate is less than threshold_percent of the highest rate, it will not be considered in the operators.
        Default is 0.01.

    Returns
    -------
    list[Qobj]
        Off-diagonal relaxation operators in the energy eigenbasis.

    Raises
    ------
    ValueError
        If threshold_percent is not between 0 and 1 (exclusive).
    """ 
    # TODO what needs to be recalculated every time?
    hamil.calcEigStates(spinsys)  # QUESTION should this be recalculated every time?
    bath.calcRates(spinsys)  
    bath.calcPopulations(spinsys)
    Rates_Summed = spinsys.Rates_Summed

    # calculate threshold
    if threshold_percent > 1 or threshold_percent < 0:
        raise ValueError('threshold_percent should be between 0 and 1')
    max_rate = np.nanmax(Rates_Summed)   # find maximum value in Rates_Summed while ignoring Nan entries
    threshold_value = threshold_percent * max_rate

    N_levels = len(spinsys.E_All)    # number of energy levels (NxN: dimension of Lindblad operators)

    # calculate relaxation operators
    collapse_ops = []   # empty list to store operators

    for i_initial in range(N_levels):
        for j_final in range(N_levels):

            if i_initial == j_final:
                continue  # skip diagonal elements
            
            if abs(Rates_Summed[i_initial][j_final]) < threshold_value:
                continue   # skip too small rates

            ##E_op = np.zeros((N_levels, N_levels))    # matrix filled with zeros of dimension NxN as base for Lindblad hopping operators
            ##E_op[j_final][i_initial] = 1    # hopping operator from energy i (column) to j (row)
            E_op = Qobj(spinsys.eigVectors[j_final]) * Qobj(spinsys.eigVectors[i_initial]).dag()  # define hopping operator from eigenstate i to j

            L_ij = np.sqrt(Rates_Summed[i_initial][j_final]) * E_op    # multiply E_op with rates/thermal prefactor            

            collapse_ops.append(Qobj(L_ij))  # append new L_ij to the list of collapse operators

    spinsys.collapse_ops = collapse_ops

    return collapse_ops 


def calcTimeEvolution(spinsys: SpinSys, psi_0: list, times: list =np.linspace(0, 1e-7, 500), spin_idx: list =[0], Saxis: str ="z"):
    """
    Solve the Lindblad master equation with QuTiP and return mesolve result.

    Parameters
    ----------
    spinsys : SpinSys
        Spin system with Hamiltonian, eigenstates, and spin operators.
    psi_0 : Qobj, array or np.array
        Initial state vector for the time evolution. 
        Can also be a density matrix of the same dimension as the Hamiltonian.
    times : array_like, optional
        Time points (in seconds) at which to evaluate the evolution.
        Default is ``np.linspace(0, 1e-7, 500)``.
    spin_idx : list[int], optional
        Indices of spins for which the expectation values over time are computed.
        Default is ``[0]``.
    Saxis : {"x", "y", "z"}, optional
        Axis along which the expectation value should be computed. The expectation value of Sx, Sy or Sz are computed depending on axis.
        Default is ``"z"``.

    Returns
    -------
    qutip.solver.Result
        QuTiP result object with ``expect`` and ``states``.

    Raises
    ------
    ValueError
        If ``Saxis`` is not one of ``"x"``, ``"y"``, or ``"z"``.
    """
    # TODO add spinsys.timeEvolution_result to docstring

    # convert psi_0 into Qobj if it is not already one
    if not isinstance(psi_0, Qobj):
        psi_0 = Qobj(psi_0)

    # Ensure the Lindblad collapse operators are calculated before using mesolve
    if not hasattr(spinsys, 'collapse_ops') or spinsys.collapse_ops is None:
        collapseOperators(spinsys)

    # convert spinsys Hamiltonian into Qobj with unit 1/s
    meV_to_J = 1e-3 * const.e
    H_q = Qobj(spinsys.H * (meV_to_J / const.hbar))

    # calculate list of operators e_ops for which the Lindblad equation will be solved
    S = getattr(spinsys, f"S{Saxis}", None)   # gets Sx, Sy or Sz depending on Saxis
    if S is None:
        raise ValueError('Saxis must be "x", "y" or "z"')
        
    e_ops = []  # initialize list of operators for which the expectation value will be computed from the total density matrix

    # calculate S matrix for each spin in spin_idx
    for spin in spin_idx:   
        E_z = 2*S[spin]   
        e_ops.append(Qobj(E_z))

    # solve Lindblad equation
    result = mesolve(H_q, psi_0, times, spinsys.collapse_ops, e_ops, options={'store_states':True})  
    result_dict = {
        'QuTip_result': result,
        'Spin_idx': spin_idx,
        'S_axis': Saxis
    }

    # add results as attributes to spinsys object
    spinsys.timeEvolution_result = result
    spinsys.timeEvolution_spinidx = spin_idx
    spinsys.timeEvolution_Saxis = Saxis

    return result


def plotTimeEvolution(spinsys: SpinSys, psi_0: list, times: list =np.linspace(0, 1e-7, 500), spin_idx: list =[0], Saxis: str ="z", title: str ="Time evolution"):
    """
    Plot spin expectation values along Saxis of selected spins over time.

    Parameters
    ----------
    spinsys : SpinSys
        Spin system with Hamiltonian, eigenstates, and spin operators.
    psi_0 : Qobj
        Initial state (ket) for the time evolution.
        Can also be a density matrix.
    times : array_like, optional
        Time points (in seconds) at which to evaluate the evolution.
        Default is ``np.linspace(0, 1e-7, 500)``.
    spin_idx : list[int], optional
        Indices of spins for which expectation values are plotted.
        Default is ``[0]``.
    Saxis : {"x", "y", "z"}, optional
        Spin component to plot (Sx, Sy or Sz correspondingly).
        Default is ``"z"``.
    title : string, optional
        Title of the plot
        Default is "Time evolution"

    Returns
    -------
    dict
        Dictionary containing:
        ``fig`` : matplotlib.figure.Figure
            The created figure.
        ``time_values`` : numpy.ndarray
            Time values used for the plot.
        ``expectation_values`` : list[numpy.ndarray]
            Expectation values for each requested spin index.
        ``state_values`` : list[Qobj]
            Time-evolved states from the solver.
        ``spin_idx`` : list[int]
            Indices of the spins for which the expectation values were calculated
    """
    # TODO: plot without recalculating result

    # calculate time evolution with QuTiP
    result = calcTimeEvolution(spinsys, psi_0, times=times, spin_idx=spin_idx, Saxis=Saxis)  

    fig, ax = plt.subplots()

    for j, spin in enumerate(spin_idx):    # plot expectation value for each spin
        ax.plot(times, result.expect[j], label=f'n={spin}')

    ax.legend()
    ax.set_ylim([-1, 1])   # Sz from -1 to 1
    ax.set_xlabel('t/s')
    ax.set_ylabel(f'<S{Saxis}>')
    plt.title(title)
    plt.show()

    return_dict = {
        'fig': fig,
        'time_values': times,
        'expectation_values': result.expect,    # expectation values for each spin in spin_idx
        'state_values': result.states,
        'QuTiP_result': result,
        'spin_idx': spin_idx
        }
    
    return return_dict

# endregion Lindblad solver

# region tip field sweep
def calcDecayTipfieldsweep(spinsys: SpinSys, psi_0: list, times: list =np.linspace(0, 1e-7, 100), BTip_values: list = np.linspace(-0.1, 0.1, 100), Baxis: str ="z", spinidx: int=0):
    """
    Calculates <Sz>(t) of a selected spin at different tip fields. 
    Sweeps through one tip field component over a specified range and computes the expectation value of Sz of a specified spin
    at each time step.
        
    Parameters
    ---------
    spinsys : SpinSys
        The spin system object 
    psi_0 : Qobj, array or np.array
        Initial state of spinsys. Can also be a density matrix.
    times: np.array, optional
        Time values at which Lindblad equation should be solved for each tip field strength.
        Default is np.linspace(0, 1e-7, 100)
    BTip_values: np.array, optional
        Tip field strengths to sweep through.
        Default is np.linspace(-0.1, 0.1, 100)
    Baxis: string, optional
        which B axis ("x", "y", or "z") should be swept through.
        Default is "z".
    spinidx: int, optional
        Which single spin of the spinsys is plotted.
        Default is 0 (which is typically the readout spin)

    Returns
    -------
    resulti: numpy.ndarray
        2D array of shape (len(BTip_values), len(times)) containing ⟨Sz⟩(t) for each time and tip field value.

    """
    # TODO check which spinsys attributes have to be recalculated and where

    # save initial value for later
    BTip_initial = spinsys.BTip.copy()   

    # convert psi_0 into Qobj if it is not already one
    if not isinstance(psi_0, Qobj):
        psi_0 = Qobj(psi_0)

    # convert axis to sweep through to numerical value for indexing
    if Baxis=="z":
        Bidx = 2
    elif Baxis=="y":
        Bidx = 1
    elif Baxis=="x":
        Bidx = 0

    hamil.calcAngMomMatrices(spinsys.ReadoutSpin)  # calculate S matrices

    # define Sz of the spin of which the expectation value will be calculated
    E_z = Qobj(2*spinsys.Sz[spinidx])   # *2 to have values from -1 to 1

    resulti = []  # initialize array to store <Sz> results

    # get decay values for spin spinidx. Store result of each tip field step in resulti.
    for BTip_value in BTip_values:
        spinsys.BTip[Bidx] = BTip_value 
        hamil.calcEigEnergies(spinsys)  # calculate new Hamiltonian and eigenenergies
    
        L_operators = collapseOperators(spinsys)  # calculate new collapse operators with new Hamiltonian

        meV_to_J = 1e-3 * const.e
        H_q = Qobj(spinsys.H * (meV_to_J / const.hbar))  # convert from meV to rad/s

        result = mesolve(H_q, psi_0, times, L_operators, [E_z])  # calculate S expectation values for given tip field

        resulti.append(result.expect[0])  # add expectation values with BTip_value to resulti
        
    resulti = np.stack(resulti)
    
    # Restore original BTip
    spinsys.BTip = BTip_initial 
    collapseOperators(spinsys);

    return resulti


def plotDecayTipfieldsweep(spinsys: SpinSys, psi_0: list, times: list =np.linspace(0, 1e-7, 100), BTip_values: list = np.linspace(-0.1, 0.1, 100), Baxis: str ="z", spinidx: int=0):
        """
        Plots a colormap the absolute values of <Sz>(t) at different tip fields. 
        Sweeps through one tip field component over a specified range, computes the expectation value of Sz of a specified spin
        at each time step. 
        
        Parameters
        ---------
        spinsys : SpinSys
            The spin system object 
        psi_0 : Qobj, array or np.array
            Initial state of spinsys. Can also be a density matrix.
        times: np.array, optional
            Time values at which Lindblad equation should be solved for each tip field strength.
            Default is np.linspace(0, 1e-7, 100)
        BTip_values: np.array, optional
            Tip field strengths to sweep through.
            Default is np.linspace(-0.1, 0.1, 100)
        Baxis: string, optional
            which B axis ("x", "y", or "z") should be swept through.
            Default is "z".
        spinidx: int, optional
            Which single spin of the spinsys is plotted.
            Default is 0 (which is typically the readout spin)

        Returns
        -------
        returndict: dict
        Dictionary containing:
        ``fig`` : matplotlib.figure.Figure
            The created figure.
        ``time_values`` : numpy.ndarray
            Time values used for the plot.
        ``BTip_values`` : numpy.ndarray
            Expectation values for each requested spin index.
        ``Sz`` : numpy.ndarray
            2D array of shape (len(BTip_values), len(times)) containing ⟨Sz⟩(t) for each time and tip field value.

        Notes
        -------
        The <Sz> values are computed with calcDecayTipfieldsweep.

        """
        # calculate <Sz> at each time and BTip field step with calcDecayTipfieldsweep
        resulti = calcDecayTipfieldsweep(spinsys, psi_0, times, BTip_values, Baxis, spinidx)

        abs_resulti = abs(resulti)

        # Plot as a 2D colormap: x=time, y=tip field values
        fig, ax = plt.subplots()
        im = ax.imshow(
            abs_resulti,
            aspect="auto",
            origin="lower",
            extent=[times[0], times[-1], BTip_values[0], BTip_values[-1]],
        )
        ax.set_xlabel("time/s")
        ax.set_ylabel(f"BTip_{Baxis}/T")
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(r'$\langle S_z \rangle$')
        plt.show()

        returndict = {
            'fig': fig,
            'time_values': times,
            'BTip_values': BTip_values,
            'Sz': resulti
        }

        return returndict

def plotTipZeemanDiagramm(spinsys: SpinSys, BTip_values: list = np.linspace(-0.1, 0.1, 100), Baxis: str ="z", energyscale: str='meV'):
    """
    Plots the Zeeman diagram for a given spin system over a specified magnetic tip field range.

    Parameters
    ----------
    spinsys : SpinSys
        Spin system containing the tip field vector, Hamiltonian, and
        energy levels (``E_sort``).
    BTip_values : array_like, optional
        Tip field values (in Tesla) to sweep for the selected component.
        Default is ``np.linspace(-0.1, 0.1, 100)``.
    Baxis : {"x", "y", "z"}, optional
        Which component of ``spinsys.BTip`` is swept. Default is ``"z"``.
        Only this component is swept through, the other two stay the same.
    energyscale : {"meV", "GHz"}, optional
        Unit for the plotted energies. Default is ``"meV"``. 

    Returns
    -------
    return_dict : dict
        A dictionary containing:
        - ``fig`` : plotly.graph_objects.Figure
            Plotly figure containing the Zeeman diagram.
        - ``B_array`` : numpy.ndarray
            Tip field values used in the sweep (same as ``BTip_values``).
        - ``E_array`` : numpy.ndarray
            Energy eigenvalues for each tip field value.
    
    Notes
    -----
    - The function restores the original ``spinsys.BTip`` after the sweep.
    - Energies are recalculated for each tip field value via
      ``hamil.calcEigEnergies``.
    """

    # Save the initial BTip-Vector
    BTipInitial = spinsys.BTip.copy()  

    N = len(BTip_values)
    E = np.zeros((N, spinsys.dimensionOfMatrix))

    # convert axis to sweep through to numerical value for indexing
    if Baxis=="z":
        Bidx = 2
    elif Baxis=="y":
        Bidx = 1
    elif Baxis=="x":
        Bidx = 0
    
    for i in range(N):
        spinsys.BTip[Bidx] = BTip_values[i]
        hamil.calcEigEnergies(spinsys)
        E[i,:] = spinsys.E_sort

    # Restore the initial solution
    spinsys.BTip = BTipInitial
    hamil.calcEigEnergies(spinsys)
        
    # Build a Plotly figure with white background
    fig = go.Figure()
    for i in range(spinsys.dimensionOfMatrix):
        if energyscale == 'GHz':
            y = E[:, i] * spinsys.meVtoGHzConversion
        else:
            y = E[:, i]
        fig.add_trace(go.Scatter(x=BTip_values, y=y, mode='lines', name=f"E_{i}"))

    fig.update_layout(
        title='Tip field Zeeman Diagramm',
        xaxis_title=f'BTip_{Baxis} (T)',
        yaxis_title=f'Energy ({energyscale})',
        plot_bgcolor='white',
        paper_bgcolor='white',
        width=600,
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_xaxes(showgrid=True, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridcolor='lightgray')
    fig.show()
    
    return_dict = {
        'fig': fig,
        'B_array': BTip_values,
        'E_array': E}

    return return_dict




# endregion tip field sweep

# region density matrix
def calcDensityMatrix(spinsys: SpinSys, time_type: str, time_index: int =None, time_value: float =None):
    """
    Return the density matrix at a selected time (value or index) from a QuTiP result. 

    Parameters
    ----------
    spinsys : SpinSys
        Spin system containing time evolution attributes.
    time_type : {"index", "value"}
        Selects whether the time is given by an explicit index or by a
        floating-point time value to be matched to the closest entry in
        ``result.times``.
    time_index : int, optional
        Index to ``result.times`` when ``time_type == "index"``.
    time_value : float, optional
        Time value (in seconds) used to find the closest entry in
        ``result.times`` when ``time_type == "value"``.

    Returns
    -------
    time : float
        Time value selected from ``result.times``.
    densmat : np.ndarray
        Density matrix corresponding to the selected time.


    Raises
    ------
    ValueError
        If ``time_type`` is "index" and ``time_index`` is not provided, or if
        ``time_type`` is "value" and ``time_value`` is not provided.
    ValueError
        If time_type is not "value" or "index".

    """
    # get time evolution result stored in spinsys
    if not hasattr(spinsys, 'timeEvolution_result'):
        raise AttributeError("The SpinSys object is missing the attribute timeEvolution_result. Run lind.calcTimeEvolution first.")
    result = spinsys.timeEvolution_result
    times = result.times

    # choose state and time of result corresponding to time_index
    if time_type == "index":  
        if time_index is None:  # make sure a time_index is given
            raise ValueError('time_index must be provided when time_type == "index"')
        state = result.states[time_index]
        time = times[time_index]

    # choose state and time at the time closest to time_value
    elif time_type == "value":
        if time_value is None:   # make sure a time_value is given
            raise ValueError('time_value must be provided when time_type == "value"')
        
        time_index = np.searchsorted(times, time_value, side='left')   # time_index is index at which times[time_index-1] < time_value <= times[time_index]
        
        if abs(times[time_index]-time_value) > abs(time_value-times[time_index-1]) and time_index != 0:
            time_index-=1     # take time_index-1 in case times[time_index-1] is closer to time_value than times[time_index]
        
        state = result.states[time_index]
        time = times[time_index]

    else: 
        raise ValueError('time_type must be "value" or "index"')


    # Get the full density matrix as a numpy array (not Qobj)
    densmat = state.full()  

    return time, densmat



def plotDensityMatrix(spinsys: SpinSys, time_type: str, time_index: int =None, time_value: float =None):
    """
    Plots real parts of the density matrix at a selected time (value or index) as a 3D bar chart.
    The density matrix is extracted of spinsys.timeEvolution_result in calc_densmat.

    Parameters
    ----------
    spinsys : SpinSys
        Spin system providing the time evolution result with density matrices and state labels used for axis tick text.
    time_type : {"index", "value"}
        Selects whether the time is given by an explicit index or by a
        floating-point time value to be matched to the closest entry in
        ``result.times``.
    time_index : int, optional
        Index to ``result.times`` when ``time_type == "index"`` at which 
        ``result.states`` density matrix is plotted.
    time_value : float, optional
        Time value (in seconds) used to find the closest entry in
        ``result.times`` when ``time_type == "value"``.

    Returns
    -------
    return_dict : dict
    A dictionary containing:
    - 'fig': matplotlib figure
    - 'time': np.ndarray
        The time at which the density matrix is plotted.
    - 'density_matrix': np.ndarray
        The density matrix that is plotted.

    Notes
    -------
    The ``result`` given to the function can be calculated via calc_time_evolution or plot_time_evolution. There it i
    """
    # TODO center state labels, make prettier
    # TODO do we want abs or real part plotted?

    if not hasattr(spinsys, 'timeEvolution_result'):
        raise AttributeError("The SpinSys object is missing the attribute timeEvolution_result. Run lind.calcTimeEvolution first.")
    time, densmat = calcDensityMatrix(spinsys, time_type, time_index, time_value)
    print(np.real(densmat.flatten()))
    ##print(densmat)

    fig = plt.figure()  # wider so that z axis label also gets displayed
    ax = fig.add_subplot(111, projection='3d')

    # Create grid for plotting in 3d
    x = np.arange(densmat.shape[0])
    y = np.arange(densmat.shape[1])
    X, Y = np.meshgrid(x, y) 

    dx = dy = 0.5 # width of the bars

    # when values are 0 in the matrix they get plotted in grey. Below zero in red, above in blue.
    is_zero = (abs(densmat.flatten()) == 0.0)
    is_neg = np.real(densmat.flatten()) < 0
    colors = np.where(is_zero, "lightgrey",
          np.where(is_neg, "tab:red", "tab:blue"))

    # get the x and y axis labels (eigenstates of H) in the format |0>, |1> etc.
    xy_axtext = spinsys.statesWithoutE

    # Plot the real part of the density matrix
    ax.bar3d(X.flatten()-dx/2, Y.flatten()-dy/2, np.zeros_like(densmat.flatten()), 0.5, 0.5, np.real(densmat.flatten()), shade=True, color=colors)
    ax.set_xlabel('State')
    ax.set_ylabel('State')
    ax.set_zlabel('Real Part of Density Matrix')
    ax.set_box_aspect(None, zoom=0.85)
    ax.set_xticks(x)
    ax.set_yticks(y)
    ax.set_xlim(-0.5, len(x) - 0.5)
    ax.set_ylim(-0.5, len(y) - 0.5)
    ax.set_zlim(0, 1)
    ax.set_xticklabels(xy_axtext)
    ax.set_yticklabels(xy_axtext)
    plt.title(f'Density Matrix at t = {time:.2} s')
    plt.show()

    return_dict = {
        'fig': fig,
        'time': time,
        'density_matrix': densmat,
        }
    
    return return_dict
# endregion density matrix

# region current
def calcTunnelCurrent_time(spinsys: SpinSys, Approach: str = "Loth"):
    """
    Calculate the tunneling current over time from a QuTiP time-evolution result.

    Parameters
    ----------
    spinsys : SpinSys
        Spin system containing Rates and timeEvolution_result.
    Approach : {"Loth", "Ternes"}, Optional
        Determines the current normalization and offset. 
        Default is ``"Loth"``.

    Returns
    -------
    I : list[float]
        Current values at each time point in ``result.times``.

    Raises
    -------
    TODO

    Notes
    -----
    - The function assumes that required attribute `Rates` are available in the provided `spinsys` object.
    - A possibility to get the QuTiP result is using the function calc_time_evolution

    """
    if not hasattr(spinsys, 'timeEvolution_result'):
        raise AttributeError("The SpinSys object is missing the attribute timeEvolution_result. Run lind.calcTimeEvolution first.")
    
    result = spinsys.timeEvolution_result
    densmats = result.states  # density matrices at each time

    # Derive the tunneling current at each time step using Equation 41 from Ternes paper
    I = []
    for timeidx in range(len(densmats)):   # iterate through time steps
        I_timeidx = 0   # initialize current at time step times[timeidx]
        densmat_timestep = densmats[timeidx].full()   # get density matrix at times[timeidx] as numpy.ndarray (not Qobj)

        for i in range(spinsys.dimensionOfMatrix):
            for j in range(spinsys.dimensionOfMatrix):
                population_i = densmat_timestep[i][i]    # population of state is diagonal entry of density matrix
                I_temp = const.e * population_i * (spinsys.Rates[0, i, j] - spinsys.Rates[1, i, j])
                I_timeidx += np.real(I_temp)    # I_temp is real but has 0j part, discard it to avoid warnings when plotting
        I.append(I_timeidx)

    I = np.array(I)    # convert to numpy array

    # Adjust current calculation based on approach
    if Approach == "Ternes":
        I = I / (const.e**2 * spinsys.T02 / spinsys.h_meV)
    if Approach == "Loth":
        I = np.add(I, spinsys.b0 * spinsys.G * spinsys.V_DC)  # add factor to every element in array I

    return I

def plotTunnelCurrent_time(spinsys: SpinSys,  Approach: str = "Loth", title: str ="Tunneling current"):
    """
    Plot the tunneling current as a function of time.

    Parameters
    ----------
    spinsys : SpinSys
        Spin system containing rate matrices, tunneling parameters and timeEvolution_result containing
        a QuTiP result object returned by ``mesolve`` with ``states`` and ``times``..
    Approach : {"Loth", "Ternes"}, optional
        Current calculation approach passed through to `calcTunnelCurrent_time`. 
        Default is ``"Loth"``.
    title : str, optional
        Title for the plot. Default is ``Tunneling current``.

    Returns
    -------
    return_dict : dict
        Dictionary containing:
        ``fig`` : matplotlib.figure.Figure
            The created figure.
        ``time_values`` : numpy.ndarray
            Time values used on the x-axis.
        ``I_values`` : list
            Current values at each time point.
        ``Approach`` : str
            The approach used for the current calculation.

    Notes
    -----
    The unit of the y-axis label depends on the selected approach. 

    """

    # TODO check units Ternes
    if not hasattr(spinsys, 'timeEvolution_result'):
        raise AttributeError("The SpinSys object is missing the attribute timeEvolution_result. Run lind.calcTimeEvolution first.")
    
    result = spinsys.timeEvolution_result
    times = result.times
    I = calcTunnelCurrent_time(spinsys, Approach=Approach)

    # set y label (different units depending on Approach)
    if Approach == "Ternes":
        ylabel = r"Current $(\frac{meV}{C^2 s})$"
    elif Approach == "Loth":
        ylabel = "Current (A)"

    fig, ax = plt.subplots()
    ax.plot(times, I)
    ax.set_xlabel("Time (s)")
    
    ax.set_ylabel(ylabel)
    plt.title(title)
    plt.show()

    return_dict = {
        'fig': fig,
        'time_values': times,
        'I_values': I,
        'Approach': Approach
    }

    return return_dict


# endregion current 

# region Fourier
def calcFourierFreqs(signal, times, reals = False):
    """
     Compute the discrete Fourier transform and frequency axis for a time signal using scipy.fft.

    Parameters
    ----------
    signal : array_like or qutip.Qobj
        Time-domain signal to transform. 
    times : array_like or qutip.Qobj
        Time values corresponding to ``signal``. Must be uniformly sampled.
    reals : bool, optional
        If ``True``, compute a real FFT (``rfft``) and return only the
        non-negative frequencies. If ``False``, compute the full complex FFT.
        Default is ``False``.

    Returns
    -------
    freqs : numpy.ndarray
        Frequency values corresponding to the FFT bins.
    amps : numpy.ndarray
        Fourier amplitudes of ``signal`` at the returned frequencies.

    Notes
    -----
    - This function assumes uniform sampling to calculate the difference between the time steps.
    """
    # convert to arrays in case the input is Qobj
    if isinstance(signal, Qobj):
        signal = signal.full()

    if isinstance(times, Qobj):
        times = times.full()

    N = len(times)  # number of data points

    dt = times[1] - times[0]    # time difference between data points (assuming uniform distribution)

    if reals:
        amps = scipy.fft.rfft(signal)
        freqs = scipy.fft.rfftfreq(N, dt)

    else:
        amps = scipy.fft.fft(signal)
        freqs = scipy.fft.fftfreq(N, dt)

    return freqs, amps


def plotFourierFreqs(spinsys: SpinSys, title: str ="Frequencies", spin_idxToPlot: list[int] =None, recalculateTimeEvolution: bool = False, psi_0: list=None, times: list =np.linspace(0, 1e-7, 500), spin_idx: list[int] =[0], axis: str ="z"):
    """
    Plot the frequency spectrum from a time evolution result. Also calculates the frequencies at which peaks are found.

    Parameters
    ----------
    spinsys : SpinSys
        Spin system containing (or used to compute) the time evolution result.
    title : str, optional
        Title of the plot. Default is ``"Frequencies"``.
    spin_idxToPlot : list[int], optional
        Spin indices to plot. If ``None``, all spins in
        ``spinsys.timeEvolution_spinidx`` are plotted.
    recalculateTimeEvolution : bool, optional
        If ``True``, recompute the time evolution even if a cached result
        exists. Default is ``False``. Then psi_0 must also be defined.
    psi_0 : array_like or Qobj, optional
        Initial state (ket or density matrix) used only when the time evolution must be recomputed.
    times : array_like, optional
        Time points (in seconds) used only when the time evolution must be recomputed.
        Default is ``np.linspace(0, 1e-7, 500)``.
    spin_idx : list[int], optional
        Spin indices used only when the time evolution must be recomputed.
        Default is ``[0]``.
    axis : {"x", "y", "z"}, optional
        Spin component used to calculate expectation values only when the time evolution must be recomputed.
        Default is ``"z"``.

    Returns
    -------
    return_dict: dict
    A dictionary containing:
    - ``fig`` : matplotlib.figure.Figure
        The created figure.
    - ``frequencies`` : list[numpy.ndarray]
        Frequency arrays for each plotted spin (same order as ``spin_indices``).
    - ``amplitudes`` : list[numpy.ndarray]
        Fourier amplitudes corresponding to ``frequencies``.
    - ``frequency_peaks`` : list[numpy.ndarray]
        Peak frequencies for each plotted spin. 
        scipy.signal.find_peaks usually doesn't find peaks at zero frequency.
    - ``spin_indices`` : list[int]
        Spin indices corresponding to each spectrum in the plot.

    Notes
    -----
    - Expectation values are real, so a real FFT (``rfft``) is used.
    - ``psi_0``, ``times``, ``spin_idx`` and ``axis`` are only used if
      ``spinsys.timeEvolution_result`` is not defined or if
      ``recalculateTimeEvolution`` is ``True``.
      Otherwise the used time evolution is ``spinsys.timeEvolution_result``,
      NOT necessarily calculated with the values given to ``plotFourierFreqs``.
    - ``psi_0``, ``times``, ``spin_idx`` and ``axis`` are only needed when
      the time evolution has to be recalculated.
    """
    # TODO add offset to docstring

    # calculate the time evolution result to be Fourier transformed if necessary or wanted
    if not hasattr(spinsys, 'timeEvolution_result') or recalculateTimeEvolution==True:
        if psi_0 == None:
            raise ValueError('psi_0 has to be defined to calculate the time evolution.')
        calcTimeEvolution(spinsys, psi_0, times=times, spin_idx=spin_idx, axis=axis)
        # calcTimeEvolution(spinsys: SpinSys, psi_0: list, times: list =np.linspace(0, 1e-7, 500), spin_idx: list =[0], axis: str ="z"):

    # get data to Fourier transform from time evolution result
    result = spinsys.timeEvolution_result  # result with expectation values
    S_expvals = result.expect
    times_used = result.times
    spin_idx = spinsys.timeEvolution_spinidx  # spin indices corresponding to time evolution results. For labeling the plot only.

    # select indices of spins to plot if not specified
    if spin_idxToPlot == None:
        spin_idxToPlot = spin_idx    # If not specified, the frequencies of all spins for which the time evolution was calculated is plotted.

    # check that requested spin indices exist in result and map to positions in spin_idx
    missing = [s for s in spin_idxToPlot if s not in spin_idx]   # list that contains missing spin indices
    if missing:     # non-empty list is considered True
        raise ValueError(f"spin_idxToPlot contains indices not present in timeEvolution_spinidx: {missing}")

    plot_idx = [i for i, s in enumerate(spin_idx) if s in spin_idxToPlot]   # indices of spin_idxToPlot values in spin_idx

    # get frequencies from calcFourierFreqs
    freqs = []
    amps = []
    labels = []

    for idxToPlot in plot_idx:
        S_values = np.real(S_expvals[idxToPlot])
        S_values = S_values - np.mean(S_values)    # remove offset (otherwise big peak at zero frequency)
        freqs_i, amps_i = calcFourierFreqs(S_values, times_used, reals=True)
        freqs.append(freqs_i)
        amps.append(amps_i)
        labels.append(spin_idx[idxToPlot])

    # calculate frequency peaks
    freq_peaks = []
    for freqs_i, amps_i in zip(freqs, amps):
        peak_idx, _ = scipy.signal.find_peaks(np.abs(amps_i))
        freq_peaks.append(freqs_i[peak_idx])

    # plot frequency spectrum
    fig, ax = plt.subplots()
    for spectrum_idx in range(len(freqs)):
        ax.plot(freqs[spectrum_idx], abs(amps[spectrum_idx]), label=f'n={labels[spectrum_idx]}')
    ax.set_xlabel("Frequency (Hz)")  
    ax.set_ylabel("Amplitude")      # TODO units
    # ax.vlines(freq_peaks, 0, 50, color='r')
    plt.title(title)
    plt.legend()
    plt.show()

    return_dict = {
        'fig': fig,
        'frequencies': freqs,
        'amplitudes': amps,
        'frequency_peaks': freq_peaks,
        'spin_indices': labels
    }

    return return_dict

# endregion Fourier


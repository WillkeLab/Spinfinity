# import python libraries
import numpy as np
import scipy.constants as const
import matplotlib.pyplot as plt

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


def test():
    """
    To test if Lindblad package was successfully imported
    """
    print("hello!")


def collapse_operators(spinsys: SpinSys, threshold_percent: float = 0.01):
    """
    Build Lindblad relaxation operators from the system's rates.

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


def calc_time_evolution(spinsys: SpinSys, psi_0: list, times: list =np.linspace(0, 1e-7, 500), spin_idx: list =[0], axis: str ="z"):
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
    axis : {"x", "y", "z"}, optional
        Axis along which the expectation value should be computed. The expectation value of Sx, Sy or Sz are computed depending on axis.
        Default is ``"z"``.

    Returns
    -------
    qutip.solver.Result
        QuTiP result object with ``expect`` and ``states``.

    Raises
    ------
    ValueError
        If ``axis`` is not one of ``"x"``, ``"y"``, or ``"z"``.
    """
    # convert psi_0 into Qobj if it is not already one
    if not isinstance(psi_0, Qobj):
        psi_0 = Qobj(psi_0)

    # Ensure the Lindblad collapse operators are calculated before using mesolve
    if not hasattr(spinsys, 'collapse_ops') or spinsys.collapse_ops is None:
        collapse_operators(spinsys)

    # convert spinsys Hamiltonian in Qobj with unit 1/s
    meV_to_J = 1e-3 * const.e
    H_q = Qobj(spinsys.H * (meV_to_J / const.hbar))

    # calculate list of operators e_ops for which the Lindblad equation will be solved
    S = getattr(spinsys, f"S{axis}", None)   # gets Sx, Sy or Sz depending on axis
    if S is None:
        raise ValueError('axis must be "x", "y" or "z"')
        
    e_ops = []  # initialize list of operators for which the expectation value will be computed from the total density matrix

    for spin in spin_idx:   # calculate S matrix for each spin in spin_idx
        E_z = 2*S[spin]   
        e_ops.append(Qobj(E_z))

    result = mesolve(H_q, psi_0, times, spinsys.collapse_ops, e_ops, options={'store_states':True})  # solve Lindblad equation

    return result


def plot_time_evolution(spinsys: SpinSys, psi_0: list, times: list =np.linspace(0, 1e-7, 500), spin_idx: list =[0], axis: str ="z", title: str ="Time evolution"):
    """
    Plot spin expectation values along axis of selected spins over time.

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
    axis : {"x", "y", "z"}, optional
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
    """
    # calculate time evolution with QuTiP
    result = calc_time_evolution(spinsys, psi_0, times=times, spin_idx=spin_idx, axis=axis)  

    fig, ax = plt.subplots()

    for j, spin in enumerate(spin_idx):    # plot expectation value for each spin
        ax.plot(times, result.expect[j], label=f'n={spin}')

    ax.legend()
    ax.set_ylim([-1, 1])
    ax.set_xlabel('t/s')
    ax.set_ylabel(f'<S{axis}>')
    plt.title(title)
    plt.show()

    return_dict = {
        'fig': fig,
        'time_values': times,
        'expectation_values': result.expect,    # expectation values for each spin in spin_idx
        'state_values': result.states,
        'QuTiP_result': result
        }
    
    return return_dict


# region tip field sweep
def calc_decay_tipfieldsweep(spinsys: SpinSys, psi_0: list, times: list =np.linspace(0, 1e-7, 500), BTip_values: list = np.linspace(-0.1, 0.1, 100), Baxis: str ="z", spinidx: int=0):
    """
    Docstring for calc_decay_tipfieldsweep
    
    :param spinsys: Description
    :type spinsys: SpinSys
    :param psi_0: Description
    :type psi_0: list
    :param times: Description
    :type times: list
    :param BTip_values: Description
    :type BTip_values: list
    :param Baxis: Description
    :type Baxis: str
    :param spinidx: Description
    :type spinidx: int
    """
    # save initial value for later
    BTip_initial = spinsys.BTip.copy()   

    # convert psi_0 into Qobj if it is not already one
    if not isinstance(psi_0, Qobj):
        si_0 = Qobj(psi_0)

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

    # get decay values for spin spinidx. Store result of each tip field step in resulti
    for BTip_value in BTip_values:
        spinsys.BTip[Bidx] = BTip_value 
        hamil.calcEigEnergies(spinsys)  # calculate new Hamiltonian and eigenenergies
    
        L_operators = collapse_operators(spinsys)  # calculate new collapse operators with new Hamiltonian

        meV_to_J = 1e-3 * const.e
        H_q = Qobj(spinsys.H * (meV_to_J / const.hbar))  # convert from meV to rad/s

        result = mesolve(H_q, psi_0, times, L_operators, [E_z])  # calculate S expectation values for given tip field

        resulti.append(result.expect[0])  # add expectation values with BTip_value to resulti
        
    resulti = np.stack(resulti)
    
    # Restore original BTip
    spinsys.BTip = BTip_initial

    return resulti


def plot_decay_tipfieldsweep(spinsys: SpinSys, psi_0: list, times: list =np.linspace(0, 1e-7, 500), BTip_values: list = np.linspace(-0.1, 0.1, 100), Baxis: str ="z", spinidx: int=0):
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
            Default is np.linspace(0, 1e-7, 500)
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
        ---------
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

        """
        # TODO check which spinsys attributes have to be recalculated and where

        resulti = calc_decay_tipfieldsweep(spinsys, psi_0, times, BTip_values, Baxis, spinidx)

        # Plot as a 2D colormap: x=time, y=tip field values
        fig, ax = plt.subplots()
        im = ax.imshow(
            abs(resulti),
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
# endregion tip field sweep

# region density matrix
def calc_densmat(spinsys: SpinSys, result: QuTiPResult, time_type: str, time_index: int =None, time_value: float =None):
    """
    Return the density matrix at a selected time (value or index). 

    Parameters
    ----------
    spinsys : SpinSys
        Spin system (currently unused, kept for symmetry with plotting).
    result : qutip.solver.result.Result
        QuTiP solver result containing ``states`` and ``times``.
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



def plot_densmat(spinsys: SpinSys, result: QuTiPResult, time_type: str, time_index: int =None, time_value: float =None):
    """
    Plots the density matrix at a selected time (value or index) as a 3D bar chart.
    Density matrix is extracted of result in calc_densmat.

    Parameters
    ----------
    spinsys : SpinSys
        Spin system providing the state labels used for axis tick text.
    result : qutip.solver.result.Result
        QuTiP solver result containing ``states`` and ``times``. 
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

    time, densmat = calc_densmat(spinsys, result, time_type, time_index, time_value)

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



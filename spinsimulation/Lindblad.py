# import python libraries
import numpy as np
import scipy
import scipy.constants as const
import matplotlib.pyplot as plt
import matplotlib.colors as mcol
import plotly.graph_objects as go
import plotly.colors
from plotly.subplots import make_subplots
from copy import deepcopy
from joblib import Parallel, delayed


# import other SpinFinity packages
from spinsimulation.SpinSys import SpinSys
import spinsimulation.SpinHamiltonian as hamil 
import spinsimulation.calcBath as bath

# import qutip code
from qutip import Qobj, mesolve
from qutip.solver.result import Result as QuTiPResult  # to be able to make sure that entities are of the qutip result class


# Content TODO update

# - calculate collapse operators with off diagonal elements
# - Filter rates and kick out values below a threshold
# - Translator into QuTiP objects 
# -   
# - Lindblad master equation solver for spin systems
# (- Current from density matrix)
# -  

# TODO matplotlib to Plotly


# region Lindblad solving
def collapseOperators(spinsys: SpinSys, threshold_percent: float = 0.01, recalcRates: bool =True):
    """
    Builds Lindblad relaxation operators from the system's rates. 
    Adds attribute collapse_ops to spinsys.

    Parameters
    ----------
    spinsys : SpinSys
        The spin system object containing eigenvectors, spin operators, rates and populations.
    threshold_percent : float, Optional
        Threshold in [0, 1] for filtering rates. If a rate is less than threshold_percent * the highest rate, it will not be considered in the operators.
        Default is 0.01.
    recalcRates : bool, Optional
        Whether to recalculate the rates with bath.calcRates as well as the eigenstates with hamil.calcEigStates before calculating 
        the collapse operators.
        Default is True. 

    Returns
    -------
    list[Qobj]
        Off-diagonal relaxation operators in the energy eigenbasis.

    Raises
    ------
    ValueError
        If threshold_percent is not between 0 and 1 (exclusive).
    """ 
    # define rates for the operator's prefactors if specified
    if recalcRates:
        hamil.calcEigEnergies(spinsys)
        hamil.calcEigStates(spinsys)
        bath.calcRates(spinsys)
    elif not hasattr(spinsys, 'Rates_Summed') or spinsys.Rates_Summed is None:
        bath.calcRates(spinsys)

    Rates_Summed = spinsys.Rates_Summed

    # calculate threshold
    if threshold_percent > 1 or threshold_percent < 0:
        raise ValueError('threshold_percent should be between 0 and 1')
    max_rate = np.nanmax(Rates_Summed)   # find maximum value in Rates_Summed while ignoring Nan entries
    threshold_value = threshold_percent * max_rate

    N_levels = spinsys.dimensionOfMatrix    # number of energy levels (NxN: dimension of Lindblad operators)

    # calculate relaxation operators
    collapse_ops = []   # empty list to store operators

    for i_initial in range(N_levels):
        for j_final in range(N_levels):

            if i_initial == j_final:
                continue  # skip diagonal elements
            
            if abs(Rates_Summed[i_initial][j_final]) < threshold_value:
                continue   # skip too small rates

            E_op = Qobj(spinsys.eigVectors[:, j_final]) * Qobj(spinsys.eigVectors[:, i_initial]).dag()  # define hopping operator from eigenstate i to j

            L_ij = np.sqrt(Rates_Summed[i_initial][j_final]) * E_op    # multiply E_op with rates/thermal prefactor            

            collapse_ops.append(Qobj(L_ij))  # append new L_ij to the list of collapse operators

    spinsys.collapse_ops = collapse_ops

    return collapse_ops 


def calcTimeEvolution(spinsys: SpinSys, psi_0: list, timeRange: list[float] =[0, 1e-7], N_times: int =500,  spin_idx: list[int] ="all", Saxis: str ="z", options: dict =None, sparse: bool =False):
    """
    Solve the Lindblad master equation with QuTiP and return mesolve result.

    Parameters
    ----------
    spinsys : SpinSys
        Spin system with Hamiltonian, eigenstates, and spin operators.
    psi_0 : Qobj, array or np.array
        Initial state vector (ket) for the time evolution. 
        Can also be a density matrix of the same dimension as the Hamiltonian.
    timeRange : list of two floats, optional
        Time range in seconds for which to calculate the evolution.
        Default is [0, 1e-7]
    N_times: int, optional
        For how many time steps to calculate the evolution. The time values will be N_times equidistant values
        in timeRange.
        Default is 500.
    spin_idx : list[int], optional
        Indices of spins for which the expectation values over time are computed.
        Default is ``"all"``, in which case the expectation values of all spins in spinsys are calculated.
    Saxis : {"x", "y", "z"}, optional
        Axis along which the expectation value should be computed. The expectation value of Sx, Sy or Sz are computed depending on axis.
        Default is ``"z"``.
    options: dict, optional
        Options to give to the QuTiP mesolve solver. See possible options in the QuTiP documentation.
        Default is None which is changed to {'store_states':True, 'nsteps':1000} in the code to avoid a mutable default argument. 
        'store_states' needs to be True to plot density matrices, otherwise it can be set to False to use less memory.
    sparse : bool, optional
        If ``True``, the Hamiltonian and collapse operators are converted
        to sparse ``qutip.Qobj`` objects before solving.  This can reduce
        memory usage for large systems.  Default is ``False``.

    Returns
    -------
    qutip.solver.Result
        QuTiP result object with ``expect`` and ``states`` if 'store_states':True in options.

    Raises
    ------
    ValueError
        If ``Saxis`` is not one of ``"x"``, ``"y"``, or ``"z"`` or spinsys.S(x, y or z) are not defined.

    Notes
    ------
    To avoid 'IntegratorException', 'nsteps' gets set to 20000 and the solver method to 'bdf' if such an exception occurs. This can take up 
    some calculation time.
    You can also increase 'nsteps' manually with the options parameter or increase N_times.

    This function adds the following attributes to the given spinsys:
    - spinsys.timeEvolution_result : the QuTiP result of mesolve
    - spinsys.timeEvolution_spinidx : spin_idx
    - spinsys.timeEvolution_Saxis : Saxis
    """
    # TODO add option to recalculate collapse operators? 

    # convert Ntimes and timeRange to list with time values
    times = np.linspace(timeRange[0], timeRange[1], N_times, dtype=float)

    # convert psi_0 into Qobj if it is not already one
    if not isinstance(psi_0, Qobj):
        psi_0 = Qobj(psi_0)

    # Ensure the Lindblad collapse operators are calculated before using mesolve
    if not hasattr(spinsys, 'collapse_ops') or spinsys.collapse_ops is None:
        collapseOperators(spinsys)
    
    # set default options for mesolve if not specified otherwise
    if options is None:
        solver_options = {'store_states': True, 'nsteps': 1000}  # default options for mesolve. Set here to avoid mutable default argument.
    else:
        solver_options = options

    # convert spinsys Hamiltonian into Qobj with unit 1/s
    meV_to_J = 1e-3 * const.e

    if sparse:  # TODO delete or test and edit
        # build sparse Hamiltonian Qobj; Qobj will infer dims from the matrix shape
        H_sparse = scipy.sparse.csr_matrix(spinsys.H * (meV_to_J / const.hbar))  # convert to CSR
        H_q = Qobj(H_sparse)
        # convert each collapse operator to a sparse Qobj as well
        c_ops = []
        for op in spinsys.collapse_ops:
            if isinstance(op, Qobj):
                mat = op.data
                if not scipy.sparse.issparse(mat):
                    mat = scipy.sparse.csr_matrix(op.full())
                c_ops.append(Qobj(mat))
            else:
                # fall back for raw arrays
                mat = op
                if not scipy.sparse.issparse(mat):
                    mat = scipy.sparse.csr_matrix(mat)
                c_ops.append(Qobj(mat))
    else:
        H_q = Qobj(spinsys.H * (meV_to_J / const.hbar))
        c_ops = spinsys.collapse_ops

    # calculate list of operators e_ops for which the Lindblad equation will be solved
    S = getattr(spinsys, f"S{Saxis}", None)   # gets Sx, Sy or Sz depending on Saxis
    if S is None:
        raise ValueError('Saxis must be "x", "y" or "z" and the angular momentum matrices must have been calculated.')
        
    e_ops = []  # initialize list of operators for which the expectation value will be computed from the total density matrix

    # calculate S matrix for each spin in spin_idx
    if spin_idx=="all":
        spin_idx = np.arange(0, spinsys.NSpins, 1)  # use all spins in spinsys
    
    for spin in spin_idx:   
        S_spin = 2*S[spin]   
        e_ops.append(Qobj(S_spin))

    # solve Lindblad equation, and retry with increased nsteps if needed
    try:
        result = mesolve(H_q, psi_0, times, c_ops, e_ops, options=solver_options)
    except Exception as exc:
        if 'Excess work done on this call' not in str(exc): 
            raise   # show original QuTiP error message

        retry_options = dict(solver_options)
        retry_options['method'] = retry_options.get('method', 'bdf')    # default method is 'adams'. 'bdf' is more stable for stiff problems but also slower.
        retry_options['nsteps'] = max(int(retry_options.get('nsteps', 1000)), 20000)    # increase nsteps to 20000 to have higher chance to get a result from the solver.
        if len(times) > 1:
            retry_options.setdefault('max_step', float(times[1] - times[0]))

        # use the same collapse operators as originally passed (may be sparse)
        result = mesolve(H_q, psi_0, times, c_ops, e_ops, options=retry_options)

    # add results as attributes to spinsys object
    spinsys.timeEvolution_result = result
    spinsys.timeEvolution_spinidx = spin_idx.copy()
    spinsys.timeEvolution_Saxis = Saxis

    return result


def plotTimeEvolution(spinsys: SpinSys, timeRange: list[float] = "all", spin_idxToPlot: list[int] ="all"):
    """
    Plot spin expectation values over time from a previously computed time evolution.

    Parameters
    ----------
    spinsys : SpinSys
        Spin system containing ``timeEvolution_result`` produced by `calcTimeEvolution`.
    timeRange : {"all"} or sequence of float, optional
        Time interval ``[t_min, t_max]`` in seconds to display on the x-axis.
        If ``"all"``, the full simulated time interval is used.
    spin_idxToPlot : {"all"} or sequence of int, optional
        Spin indices to include in the plot. If ``"all"``, all spin indices for which the time evolution
        was computed are plotted.

    Returns
    -------
    return_dict : dict
        ``fig``
            Matplotlib figure object.
        ``time_values``
            Time grid used in the underlying simulation.
        ``expectation_values``
            Spin expectation values.
        ``Saxis``
            Axis along which the expectation value <S_Saxis> was computed.
        ``spin_idx_plotted``
            Positions of the plotted spins within ``spinsys.timeEvolution_spinidx``.
    The returned expectation_values, time_values and Saxis are the ones calculated in calcTimeEvolution and already 
    stored in the SpinSys object. In order to replicate the plot yourself, plot expectation_values[spin_idx_plotted] 
    and limit the time axis to your desired time range.

    Raises
    ------
    AttributeError
        If ``spinsys.timeEvolution_result`` is missing.
    ValueError
        If ``timeRange`` lies outside the simulated interval.
    ValueError
        If requested ``spin_idxToPlot`` entries are not present in the spins ``spinsys.timeEvolution_spinidx`` 
        for which the time evolution was computed.
        
    Notes
    -----
    This function does not run a new time evolution. It only visualizes data previously
    computed by `calcTimeEvolution`.
    """

    # check if the time evolution was calculated
    if not hasattr(spinsys, 'timeEvolution_result') or spinsys.timeEvolution_result is None:
        raise AttributeError('timeEvolution_result is not defined. The time evolution must first be calculated via calcTimeEvolution.')
    
    result = spinsys.timeEvolution_result
    times_calc = result.times   # times for which the time evolution was calculated
    spin_idx_calc = spinsys.timeEvolution_spinidx    # indices of the spins for which the time evolution was calculated
    Saxis = spinsys.timeEvolution_Saxis

    fig, ax = plt.subplots()

    # set timeRange to the time range for which the time evolution was calculated if not specified otherwise.
    if timeRange == "all":
        timeRange = [times_calc[0], times_calc[-1]]

    # check that timeRange is not bigger than the time range for which the expectation values were calculated
    if timeRange[-1] > times_calc[-1]:
        raise ValueError('The time range upper bound exceeds the time range for which the time evolution was calculated.')
    if timeRange[0] < times_calc[0]:
        raise ValueError('The time range lower bound is below the time range for which the time evolution was calculated.')

    # select spins to plot and check if their time evolutions were calculated
    if spin_idxToPlot == "all":
        spin_idxToPlot = spin_idx_calc    # If not specified, the expectation values of all spins for which the time evolution was calculated are plotted.

    # check that requested spin indices exist in result and map to positions in spin_idx_calc
    missing = [s for s in spin_idxToPlot if s not in spin_idx_calc]   # list that contains missing spin indices
    if missing:     # non-empty list is considered True
        raise ValueError(f"spin_idxToPlot contains indices not present in timeEvolution_spinidx: {missing}")

    plot_idx = [i for i, s in enumerate(spin_idx_calc) if s in spin_idxToPlot]   # indices of spin_idxToPlot values in spin_idx_calc
    for i in plot_idx:    # plot expectation value for each spin
        ax.plot(times_calc, result.expect[i], label=f'n={spin_idx_calc[i]}')

    ax.legend()
    ax.set_ylim([-1, 1])   # <S> from -1 to 1
    ax.set_xlim(timeRange)
    ax.set_xlabel('t/s')
    ax.set_ylabel(f'<S{Saxis}>')
    plt.title("Time evolution")
    plt.show()

    return_dict = {
        'fig': fig,
        'time_values': times_calc,
        'expectation_values': result.expect,
        'Saxis': Saxis,
        'spin_idx_plotted': plot_idx
        }
    
    return return_dict

# endregion Lindblad solver

# region tip field sweep
def calcDecayTipfieldsweep(spinsys: SpinSys, psi_0: list, timeRange: list[float] =[0, 1e-7], N_times: int =100, BTipRange: list[float] =[-0.1, 0.1], N_BTip:int =100, Baxis: str ="z", Saxis: str ="z", spinidx: int=0, options: dict =None):
    """
    Calculates <S>(t) of a selected spin at different tip fields. 
    Sweeps through one tip field component over a specified range and computes the expectation value of S_{Saxis} of one specified spin.
    at each time step.
    
    Parameters
    ---------
    spinsys : SpinSys
        The spin system object 
    psi_0 : Qobj, array or np.array
        Initial state of spinsys. Can also be a density matrix.
    timeRange : list of float, optional
        Time range [min, max] in seconds for which the Lindblad equation should be solved for each tip field strength.
        Default is [0, 1e-7].
    N_times : int, optional
        Number of time points in the sweep.
        Default is 100.
    BTipRange : list of float, optional
        Tip field strength range [min, max] in Tesla for which the Lindblad equation should be solved at each time step.
        Default is [-0.1, 0.1].
    N_BTip : int, optional
        Number of tip field points in the sweep.
        Default is 100.
    Baxis: string, optional
        which B axis ("x", "y", or "z") should be swept through.
        Default is "z".
    Saxis: string, optional
        For which S axis ("x", "y", or "z") the expectation value should be calculated.
        Default is "z".
    spinidx: int, optional
        Which single spin of the spinsys is plotted.
        Default is 0 (which is typically the readout spin)
    options: dict, optional
        Options to give to the QuTiP mesolve solver. See possible options in the QuTiP documentation.
        Default is None and changed to {'store_states':False, 'nsteps':1000} to avoid mutable default argument. 
        'nsteps' can be increased if you get an 'IntegratorException'. 
    
    Returns
    -------
    resulti: numpy.ndarray
        2D array of shape (len(BTip_values), len(times)) containing <S_{Saxis}>(t) for each time and tip field value.

    Raises
    -------
    ValueError
        If Baxis is not "x", "y" or "z".
    ValueError
        If Saxis is not "x", "y" or "z" or spinsys.S(x, y or z) are not defined.

    Notes
    -------
    The time evolution results for the different tip field strengths are calculated in parallel using _calcDecayParallelHelper.
    """
    # # save initial value for later
    # BTip_initial = spinsys.BTip.copy()   

    # define BTip and time values arrays
    times = np.linspace(timeRange[0], timeRange[1], N_times)
    BTip_values = np.linspace(BTipRange[0], BTipRange[1], N_BTip)

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
    else: 
        raise ValueError('Baxis must be "x", "y" or "z".')

    # define Sx/Sy/Sz of the spin of which the expectation value will be calculated
    if Saxis=="x":
        S = Qobj(2*spinsys.Sx[spinidx])   # *2 to have values from -1 to 1
    elif Saxis=="y":
        S = Qobj(2*spinsys.Sy[spinidx])   # *2 to have values from -1 to 1
    elif Saxis=="z":
        S = Qobj(2*spinsys.Sz[spinidx])   # *2 to have values from -1 to 1
    else: 
        raise ValueError('Saxis must be "x", "y" or "z" and the angular momentum matrices must have been calculated.')
    
    # set default options for mesolve if not specified otherwise
    if options is None:
        solver_options = {'store_states': False, 'nsteps': 1000}  # default options for mesolve.
    else:
        solver_options = options

    # compute <Sz(t)> for each tip field strength
    resulti = Parallel(n_jobs=-1)(delayed(_calcDecayParallelHelper)(spinsys, S, BTip_value, Bidx, psi_0, times, solver_options)
                                  for BTip_value in BTip_values)
    resulti = np.stack(resulti)

    return resulti


def _calcDecayParallelHelper(spinsys: SpinSys, S: Qobj, BTip_value: float, Bidx: int, psi_0: Qobj, times: list[float], solver_options: dict):
    """
    Calculates the expectation value of S over time using QuTip mesolve. 
    It is calculated for a specific spin system and changes one component of the tip field locally.
    
    Parameters
    -------
    spinsys : SpinSys
        The spin system object. 
    S: Qobj
        Spin matrix (Sx, Sy or Sz) of the spin of which the expectation value over time should be calculated.
    BTip_value: float
        Tip field strength that should be inserted into the component along the axis Bidx
    Bidx: int {0, 1, 2}
        Index of the BTip axis along which the value will be changed to BTip_value.
        0 corresponds to x, 1 to y and 2 to z.
    psi_0 : Qobj
        Initial state or density matrix of spinsys.
    times : list of float
        Time values in seconds for which the Lindblad equation should be solved.
    solver_options: dict
        Options to give to the QuTiP mesolve solver.
    
    Returns
    -------
    resulti: numpy.ndarray
        Array containing <S>(t) for each time value.

    Notes
    -------
    This function is designed to be used for parallel computation in calcDecayTipfieldSweep. It creates a local deepcopy of spinsys.
    """
    local_spinsys = deepcopy(spinsys)   # create copy of spinsys to change its BTip only locally
    local_spinsys.BTip[Bidx] = BTip_value   # change BTip value along Bidx axis

    hamil.calcEigEnergies(local_spinsys)  # calculate new Hamiltonian, eigenenergies and eigenstates
    L_operators = collapseOperators(local_spinsys)  # calculate new collapse operators with new Hamiltonian

    meV_to_J = 1e-3 * const.e
    H_q = Qobj(local_spinsys.H * (meV_to_J / const.hbar))  # convert from meV to rad/s

    result = mesolve(H_q, psi_0, times, L_operators, [S], options=solver_options)  # calculate time evolution for given tip field
    result_expval = np.real(result.expect[0])    # calculated S expectation value

    return result_expval



def plotDecayTipfieldsweep(spinsys: SpinSys, psi_0: list, timeRange: list[float] =[0, 1e-7], N_times: int =100, BTipRange: list[float] =[-0.1, 0.1], N_BTip: int =100, Baxis: str ="z", Saxis: str ="z", spinidx: int=0, options:dict =None):
        """
        Plots a colormap of <S>(t) at different tip fields over time. 
        Sweeps through one tip field component over a specified range, computes the expectation value of Sx, Sy or Sz of a specified spin
        at each time step. 
        
        Parameters
        ---------
        spinsys : SpinSys
            The spin system object 
        psi_0 : Qobj, array or np.array
            Initial state of spinsys. Can also be a density matrix.
        timeRange : list of float, optional
            Time range [min, max] in seconds for which the Lindblad equation should be solved for each tip field strength.
            Default is [0, 1e-7].
        N_times : int, optional
            Number of time points in the sweep.
            Default is 100.
        BTipRange : list of float, optional
            Tip field strength range [min, max] in Tesla for which the Lindblad equation should be solved at each time step.
            Default is [-0.1, 0.1].
        N_BTip : int, optional
            Number of tip field points in the sweep.
            Default is 100.
        Baxis: string, optional
            Which B axis ("x", "y", or "z") should be swept through.
            Default is "z".
        Saxis: string, optional
            For which S axis ("x", "y", or "z") the expectation value should be calculated.
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
            Tip field values used for the plot (BTip_values = np.linspace(BTipRange[0], BTipRange[1], N_BTip))
        ``S`` : numpy.ndarray
            2D array of shape (len(BTip_values), len(times)) containing the expectation value ⟨S_{Saxis}⟩ 
            for each time and tip field value.

        Notes
        -------
        The <S> values are computed with calcDecayTipfieldsweep.
        """
        # calculate <S> at each time and BTip field step with calcDecayTipfieldsweep
        resulti = calcDecayTipfieldsweep(spinsys, psi_0,  timeRange=timeRange, N_times=N_times, BTipRange=BTipRange, N_BTip=N_BTip, Baxis=Baxis, Saxis=Saxis, spinidx=spinidx, options=options)

        resulti = np.real(resulti)

        # define BTip and time values arrays
        times = np.linspace(timeRange[0], timeRange[1], N_times)
        BTip_values = np.linspace(BTipRange[0], BTipRange[1], N_BTip)

        # Plot as a 2D colormap: x=time, y=tip field values
        fig, ax = plt.subplots()
        im = ax.imshow(
            resulti,
            aspect="auto",
            norm = mcol.Normalize(vmin=-1, vmax=1, clip=False),     # Sz from -1 to 1
            origin="lower",
            extent=[times[0], times[-1], BTip_values[0], BTip_values[-1]],
            cmap="managua_r",
        )
        ax.set_xlabel("time/s")
        ax.set_ylabel(f"BTip_{Baxis}/T")
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(f'<S{Saxis}>')
        plt.show()

        returndict = {
            'fig': fig,
            'time_values': times,
            'BTip_values': BTip_values,
            'S': resulti
        }

        return returndict


def plotTipZeemanDiagramm(spinsys: SpinSys, BTipRange: list[float] = [-0.1, 0.1], N_BTip: int =100, Baxis: str ="z", energyscale: str='meV'):
    """
    Plots the Zeeman diagram for a given spin system over a specified magnetic tip field range.

    Parameters
    ----------
    spinsys : SpinSys
        Spin system containing the tip field vector, Hamiltonian, and
        energy levels (``E_sort``).
    BTipRange : list of float, optional
        Tip field strength range [min, max] in Tesla to sweep the selected BTip component (along Baxis) through.
        Default is [-0.1, 0.1].
    N_BTip : int, optional
        Number of tip field points in the sweep.
        Default is 100.
    Baxis : {"x", "y", "z"}, optional
        Which component of ``spinsys.BTip`` is swept through. Default is ``"z"``.
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

    # create useful arrays
    E = np.zeros((N_BTip, spinsys.dimensionOfMatrix))   # empty array with one entry for each energy eigenvalue and BTip value
    BTip_values = np.linspace(BTipRange[0], BTipRange[1], N_BTip)   # array of BTip values to sweep through

    # convert axis to sweep through to numerical value for indexing
    if Baxis=="z":
        Bidx = 2
    elif Baxis=="y":
        Bidx = 1
    elif Baxis=="x":
        Bidx = 0
    
    for i in range(N_BTip):
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
def calcDensityMatrix(spinsys: SpinSys, time_type: str, time_index: list[int] =None, time_values: list[float] =None):
    """
    Return the density matrices at selected times (values or indices) from a QuTiP result. 

    Parameters
    ----------
    spinsys : SpinSys
        Spin system containing time evolution attributes.
    time_type : {"index", "value"}
        Selects whether the time is given by an explicit index or by a
        floating-point time value to be matched to the closest entry in
        ``result.times``.
    time_index : list[int], optional
        Indices to ``result.times`` when ``time_type == "index"``.
    time_values : list[float], optional
        Time values (in seconds) used to find the closest entry in
        ``result.times`` when ``time_type == "value"``.

    Returns
    -------
    times_densmats : float
        Time values selected from ``result.times``.
    densmats : np.ndarray
        Density matrices corresponding to the selected times.
    result_indices : list[int]
        The indices in ``result.times`` and ``result.states`` corresponding to the returned density matrices.
        If time_type = "index", they are equivalent to time_index.

    Raises
    ------
    ValueError
        If ``time_type`` is "index" and ``time_index`` is not provided, or if
        ``time_type`` is "value" and ``time_values`` is not provided.
    ValueError
        If time_type is not "value" or "index".

    """
    
    # get time evolution result stored in spinsys
    if not hasattr(spinsys, 'timeEvolution_result'):
        raise AttributeError("The SpinSys object is missing the attribute timeEvolution_result. Run lind.calcTimeEvolution first.")
    result = spinsys.timeEvolution_result
    times_result = result.times

    times_densmats = []   # initialize list to store time values for which density matrices are calculated
    densmats = []    # initialize list to store density matrices for each requested time value

    # choose state and time of result corresponding to time_index
    if time_type == "index":  
        if time_index is None:  # make sure a time_index is given
            raise ValueError('time_index must be provided when time_type == "index"')
        for time_index_i in time_index:
            densmat = result.states[time_index_i].full()
            time = times_result[time_index_i]
            times_densmats.append(time)
            densmats.append(densmat)
        result_indices = time_index

    # choose state and time at the time closest to time_value
    elif time_type == "value":
        result_indices = []  # initialize list to store indices corresponding to time_values

        if time_values is None:   # make sure a time_value is given
            raise ValueError('time_value must be provided when time_type == "value"')
        
        for time_value_i in time_values:    # loop through time_values and find closest times in result.times 

            # find indices corresponding to time_value_i in times_result.
            if time_value_i < times_result[0]:     # take first time step if time_value is smaller than the first time in result.times
                time_index = 0

            elif time_value_i > times_result[-1]:        # take last time step if time_value is higher than the last time in result.times
                time_index = len(times_result) - 1

            else:
                time_index = np.searchsorted(times_result, time_value_i, side='left')   # time_index is index at which times_result[time_index-1] < time_value_i <= times_result[time_index]
        
                if abs(times_result[time_index]-time_value_i) > abs(time_value_i-times_result[time_index-1]) and time_index != 0:
                    time_index-=1     # take time_index-1 in case times_result[time_index-1] is closer to time_value_i than times_result[time_index]
            
            # add density matrix and corresponding time to the lists
            densmat = result.states[time_index].full() 
            time = times_result[time_index]
            times_densmats.append(time)
            densmats.append(densmat)
            result_indices.append(time_index)

    else: 
        raise ValueError('time_type must be "value" or "index"')

    return times_densmats, densmats, result_indices



def plotDensityMatrix(spinsys: SpinSys, time_type: str, time_index: list[int] =None, time_value: list[float] =None):
    """
    Plots real parts of the density matrix at a selected time (value or index) as a 3D bar chart.
    The density matrix is extracted of spinsys.timeEvolution_result in calcDensityMatrix.

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
    - 'times': np.ndarray
        The times at which the density matrices are plotted.
    - 'density_matrices': list of np.ndarray
        The density matrices that are plotted.
    - 'result_indices': list[int]
        The indices in ``result.times`` and ``result.states`` corresponding to the plotted density matrices.
    """
    # TODO center state labels, make prettier

    if not hasattr(spinsys, 'timeEvolution_result'):
        raise AttributeError("The SpinSys object is missing the attribute timeEvolution_result. Run lind.calcTimeEvolution first.")
    
    times, densmats, result_indices = calcDensityMatrix(spinsys, time_type, time_index, time_value)

    # plot as grid of 3d bar plots
    n = len(densmats)
    ncols = min(3, n)    # max 3 plots per row
    nrows = int(np.ceil(n / ncols))
    fig = plt.figure(figsize=(6*n, 4*n))
    ##fig = plt.figure()

    # for time, densmat in zip(times, densmats):  # plot all density matrices in case multiple time values/indices were given
    #     ax = fig.add_subplot(111, projection='3d')
    for k, (time, densmat) in enumerate(zip(times, densmats), start=1):
        ax = fig.add_subplot(nrows, ncols, k, projection='3d')
        densmat = np.real(densmat)   # take real part of density matrix for plotting

        # Create grid for plotting in 3d
        x = np.arange(densmat.shape[0])
        y = np.arange(densmat.shape[1])
        X, Y = np.meshgrid(x, y) 

        dx = dy = 0.5   # width of the bars

        # when values are 0 in the matrix they get plotted in grey. Below zero in red, above in blue.
        is_zero = (densmat.flatten() == 0.0)
        is_neg = (densmat.flatten() < 0)
        colors = np.where(is_zero, "lightgrey",
            np.where(is_neg, "tab:red", "tab:blue"))

        # get the x and y axis labels (eigenstates of H) in the format |0>, |1> etc.
        xy_axtext = spinsys.statesWithoutE

        # Plot the real part of the density matrix
        ax.bar3d(X.flatten()-dx/2, Y.flatten()-dy/2, np.zeros_like(densmat.flatten()), 0.5, 0.5, densmat.flatten(), shade=True, color=colors)
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
        plt.title(f'Density Matrix at t = {time:.2} s', fontsize=18)
    plt.show()

    return_dict = {
        'fig': fig,
        'times': times,
        'density_matrices': densmats,
        'result_indices': result_indices,
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
    I : np.ndarray
        Current values at each time point in ``result.times``.

    Raises
    -------
    AttributeError
        If ``spinsys`` does not have ``timeEvolution_result``. Run
        ``calcTimeEvolution`` first.

    Notes
    -----
    - The function assumes that required attribute `Rates` are available in the provided `spinsys` object.
    - The current is calculated from the SpinSys attribute timeEvolution_result that needs to be calculated in calcTime_Evolution first.
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

def plotTunnelCurrent_time(spinsys: SpinSys, timeRange: list[float] = "all", Approach: str = "Loth"):
    """
    Plot the tunneling current as a function of time. It is calculated in calcTunnelCurrent_time from the populations of the states 
    (diagonal entries of the density matrix) and the rates.

    Parameters
    ----------
    spinsys : SpinSys
        Spin system containing rate matrices, tunneling parameters and timeEvolution_result containing
        a QuTiP result object returned by ``mesolve`` with ``states`` and ``times``.
    timeRange: list[float] or "all", optional 
        Time interval [t_min, t_max] for which to plot the current. If "all", the full simulated time interval is used.
    Approach : {"Loth", "Ternes"}, optional
        Current calculation approach passed through to `calcTunnelCurrent_time`. 
        Default is ``"Loth"``.

    Returns
    -------
    return_dict : dict
        Dictionary containing:
        ``fig`` : matplotlib.figure.Figure
            The created figure.
        ``time_values`` : numpy.ndarray
            Time values of the whole simulated time interval (not only the selected timeRange if it is not "all").
        ``I_values`` : np.ndarray
            Current values corresponding to the time values in ``time_values``.
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

    # select time range to plot
    if timeRange == "all":
        timeRange = [times[0], times[-1]]

    # set y label (different units depending on Approach)
    if Approach == "Ternes":
        ylabel = r"Current $(\frac{meV}{C^2 s})$"
    elif Approach == "Loth":
        ylabel = "Current (A)"

    fig, ax = plt.subplots()
    ax.plot(times, I)
    ax.set_xlabel("Time (s)")
    
    ax.set_ylabel(ylabel)
    ax.set_xlim(timeRange)
    plt.title("Tunneling current")
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


def plotFourierFreqs(spinsys: SpinSys, spin_idxToPlot: list[int] ="all"):
    """
    Plot the frequency spectrum from an existing time evolution result and
    return the FFT data and peak frequencies.

    Parameters
    ----------
    spinsys : SpinSys
        Spin system containing ``timeEvolution_result`` produced by
        ``calcTimeEvolution``.
    spin_idxToPlot : {"all"} or sequence of int, optional
        Spin indices to plot. If ``"all"``, all spins in
        ``spinsys.timeEvolution_spinidx`` are plotted.

    Returns
    -------
    return_dict : dict
        A dictionary containing:
        - ``fig`` : matplotlib.figure.Figure
            The created figure.
        - ``frequencies`` : list[numpy.ndarray]
            Frequency arrays for each plotted spin (same order as
            ``spin_indices``).
        - ``amplitudes`` : list[numpy.ndarray]
            Fourier amplitudes corresponding to ``frequencies``.
        - ``frequency_peaks`` : list[numpy.ndarray]
            Peak frequencies for each plotted spin. Only peaks with an amplitude of at least 1% of the maximum amplitude are considered.
        - ``spin_indices`` : list[int]
            Spin indices corresponding to each spectrum in the plot.

    Notes
    -----
    - Expectation values are real, so a real FFT (``rfft``) is used.
    - The mean value of each spin's <S> is subtracted before the FFT to
      remove the DC offset (otherwise a large peak appears at zero frequency).
    - This function does not recompute time evolution; it uses the cached
      ``spinsys.timeEvolution_result``.
    """
    # TODO units amplitude/normalize it?

    # calculate the time evolution result to be Fourier transformed if necessary or wanted
    if not hasattr(spinsys, 'timeEvolution_result') or spinsys.timeEvolution_result == None:
        raise AttributeError('The SpinSys object has no attribute `timeEvolution_result`. Run lind.calcTimeEvolution first.')

    # get data to Fourier transform from time evolution result
    result = spinsys.timeEvolution_result  # result with expectation values
    S_expvals = result.expect
    times = result.times
    spin_idx = spinsys.timeEvolution_spinidx  # spin indices corresponding to time evolution results. For labeling the plot only.

    # select indices of spins to plot if not specified
    if spin_idxToPlot == "all":
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
        freqs_i, amps_i = calcFourierFreqs(S_values, times, reals=True)
        freqs.append(freqs_i)
        amps.append(amps_i)
        labels.append(spin_idx[idxToPlot])

    # calculate frequency peaks
    freq_peaks = []
    for freqs_i, amps_i in zip(freqs, amps):
        peak_idx, _ = scipy.signal.find_peaks(
            np.abs(amps_i),
            height=0.01*np.max(np.abs(amps_i))  # only consider peaks that are at least 1% of the maximum amplitude
            )
        freq_peaks.append(freqs_i[peak_idx])
                
    # calculate frequency axis limits for plotting
    max_peak_freq = [max(peaks) for peaks in freq_peaks if len(peaks) > 0]  # get maximum peak frequencies across all spectra
    if max_peak_freq:   # if peaks were found, set x_limit to 3 times the maximum peak frequency, but not exceeding the maximum frequency in the spectra
        x_limit = 3 * max(max_peak_freq)  # 3 times the maximum peak frequency
        if x_limit > max(freq[-1] for freq in freqs):  # if calculated x_limit exceeds the maximum frequency in the spectra
            x_limit = max(freq[-1] for freq in freqs)  # set x_limit to maximum frequency in the spectra (plot whole spectrum)
    else:   # if no peaks were found
        x_limit = max(freq[-1] for freq in freqs)  # set x_limit to maximum frequency in the spectra

    # plot frequency spectrum
    fig, ax = plt.subplots()
    for spectrum_idx in range(len(freqs)):
        ax.plot(freqs[spectrum_idx], abs(amps[spectrum_idx]), label=f'n={labels[spectrum_idx]}')
    ax.set_xlabel("Frequency (Hz)")  
    ax.set_ylabel("Amplitude")
    ax.set_xlim(0, x_limit)
    plt.title("Fourier transformation")
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


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


# Content

# - calculate collapse operators with off diagonal elements
# - Filter rates and kick out values below a threshhold
# - Translator into QuTip objects 
# -   
# - Lindblad master equation solver for spin systems
# (- Current from density matrix)
# -  


def test():
    """
    To test if Lindblad package was successfully imported
    """
    print("hello!")


def collapse_operators(spinsys, threshold_percent = 0.01):
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

            ##print(f'initial state: \n {spinsys.eigVectors[i_initial]}')
            ##print(f'final state: \n {spinsys.eigVectors[j_final]}\n ')
            ##print(f'hop: \n {E_op*Qobj(spinsys.eigVectors[i_initial])}\n ')

            L_ij = np.sqrt(Rates_Summed[i_initial][j_final]) * E_op    # multiply E_op with rates/thermal prefactor            
 
            ## for testing: print(f'L[{i_initial}][{j_final}] = {L_ij}')

            collapse_ops.append(Qobj(L_ij))  # append new L_ij to the list of collapse operators

    spinsys.collapse_ops = collapse_ops

    return collapse_ops 


def calc_time_evolution(spinsys, psi_0, times=np.linspace(0, 1e-7, 500), spin_idx=[0], axis="z"):
    """
    Solve the Lindblad master equation with QuTip and return mesolve result.

    Parameters
    ----------
    spinsys : SpinSys
        Spin system with Hamiltonian, eigenstates, and spin operators.
    psi_0 : Qobj
        Initial state (ket) for the time evolution.
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


def plot_time_evolution(spinsys, psi_0, times=np.linspace(0, 1e-7, 500), spin_idx=[0], axis="z"):
    """
    Plot expectation values of a spin component over time.

    Parameters
    ----------
    spinsys : SpinSys
        Spin system with Hamiltonian, eigenstates, and spin operators.
    psi_0 : Qobj
        Initial state (ket or density matrix) for the time evolution.
    times : array_like, optional
        Time points (in seconds) at which to evaluate the evolution.
        Default is ``np.linspace(0, 1e-7, 500)``.
    spin_idx : list[int], optional
        Indices of spins for which expectation values are plotted.
        Default is ``[0]``.
    axis : {"x", "y", "z"}, optional
        Spin component to plot.
        Default is ``"z"``.

    Returns
    -------
    dict
        Dictionary containing:
        ``fig`` : matplotlib.figure.Figure
            The created figure.
        ``time_values`` : ndarray
            Time values used for the plot.
        ``expectation_values`` : list[ndarray]
            Expectation values for each requested spin index.
        ``state_values`` : list[Qobj]
            Time-evolved states from the solver.

    See Also
    --------
    calc_time_evolution : Compute the time evolution without plotting.
    """

    result = calc_time_evolution(spinsys, psi_0, times=times, spin_idx=spin_idx, axis=axis)

    fig, ax = plt.subplots()

    for spin in spin_idx:
        ax.plot(times, result.expect[spin], label=f'n={spin}')

    ax.legend()
    ax.set_ylim([-1, 1])
    ax.set_xlabel('t/s')
    ax.set_ylabel(f'<S{axis}>')
    plt.show()

    return_dict = {
        'fig': fig,
        'time_values': times,
        'expectation_values': result.expect,
        'state_values': result.states
        }
    
    return return_dict



def plot_decay_tipfieldsweep(spinsys, psi_0, times=np.linspace(0, 1e-7, 500), BTip_values = np.linspace(-0.1, 0.1, 100), Baxis="z", spinidx=0):
        """
        Plots a colormap of <Sz>(t) at different tip fields. 
        Sweeps through one tip field component over a specified range, computes the expectation value of Sz of a specified spin
        at each time step.
        
        Parameters
        ---------
        spinsys : SpinSys
            The spin system object 
        psi_0 : Qobj
            Initial state of spinsys.
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
        np.array
            2D array of shape (len(BTip_values), len(times)) containing ⟨Sz⟩(t) for each time and tip field value.

        """
        # TODO check which spinsys attributes have to be recalculated and where
        # TODO check factor 2 in E_z
        BTip_initial = spinsys.BTip   # save for later

        # select axis to sweep through (convert to numerical index)
        if Baxis=="z":
            Bidx = 2
        elif Baxis=="y":
            Bidx = 1
        elif Baxis=="x":
            Bidx = 0

        hamil.calcAngMomMatrices(spinsys.ReadoutSpin)  # calculate Sz
        E_z = Qobj(2*spinsys.Sz[spinidx])   # define Sz of the chosen spin of which the expectation value will be calculated

        resulti = []  # initialize array to store <Sz> results

        # get decay values for spin spinidx. Store result of each tip field step in resulti
        for BTip_value in BTip_values:
            spinsys.BTip[Bidx] = BTip_value
            hamil.calcEigEnergies(spinsys)  # calculate new Hamiltonian and eigenenergies
    
            L_operators = collapse_operators(spinsys)  # calculate new collapse operators with new Hamiltonian

            meV_to_J = 1e-3 * const.e
            H_q = Qobj(spinsys.H * (meV_to_J / const.hbar))  # convert from meV to rad/s

            result = mesolve(H_q, psi_0, times, L_operators, [E_z])  # calculate Sz expectation values for given tip field

            resulti.append(result.expect[0])  # add expectation values with BTip_value to resulti
        
        resulti = np.stack(resulti)

        # Plot as a 2D colormap: x=time, y=tip field values
        fig, ax = plt.subplots()
        im = ax.imshow(
            resulti,
            aspect="auto",
            origin="lower",
            extent=[times[0], times[-1], BTip_values[0], BTip_values[-1]],
        )
        ax.set_xlabel("time/s")
        ax.set_ylabel(f"BTip_{Baxis}/T")
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(r'$\langle S_z \rangle$')
        plt.tight_layout()
        plt.show()

        # Restore original BTip
        spinsys.BTip = BTip_initial

        return resulti








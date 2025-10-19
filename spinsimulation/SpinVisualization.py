import numpy as np
import scipy as sp
import scipy.constants as const
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.colors

from spinsimulation.SpinSys import SpinSys
import spinsimulation.SpinHamiltonian as hamil 
import spinsimulation.calcBath as bath
import spinsimulation.calcdIdV as dIdV
import spinsimulation.calcESR as ESR
import spinsimulation.calcdIdV3 as dIdV3

# region Hamiltonian

def plotEigenMatrix(spinsys: SpinSys, plot_type: str='p'):
        """
        Visualizes the eigenstate matrix of the spin system as a heatmap.

        Parameters
        ----------
        spinsys : SpinSys
            The spin system object containing eigenvectors, basis states, and other relevant properties.
        plot_type : str, optional
            Type of plot to generate:
            - 'p': Probability (default), displays the probability matrix using the 'hot' colormap.
            - 'a': Amplitude, displays the amplitude matrix using the 'coolwarm' colormap.

        Returns
        -------
        None
            This function generates a matplotlib figure and does not return any value.

        Raises
        ------
        ValueError
            If `plot_type` is not 'p' or 'a'.
            
        Notes
        -----
        The color limits are set to [0, 1] for probability and [-1, 1] for amplitude.
        """

        # Check if eigVectors are calculated, if not calculate them. They are needed for the projection matrix.
        if not hasattr(spinsys, 'eigVectors') or spinsys.eigVectors is None:
            hamil.calcEigStates(spinsys)
               
        projection = np.zeros([spinsys.dimensionOfMatrix,spinsys.dimensionOfMatrix],dtype=complex)
        for i in range(spinsys.dimensionOfMatrix):
            for j in range(spinsys.dimensionOfMatrix):
                projection[i, j] = np.dot(spinsys.eigVectors[:, i].conj().T, np.identity(spinsys.dimensionOfMatrix)[:, j])


        if plot_type == 'p':
            m = abs(np.conjugate(projection) * projection)
            name0, colormap = 'Probability', 'hot'
        elif plot_type == 'a':
            m = np.real(projection) + np.imag(projection)
            name0, colormap = 'Amplitude', 'coolwarm'
        else:
            raise ValueError("Invalid plot_type. Use 'p' for probability or 'a' for amplitude.")

        
        # Generate the figure and axes
        fig, ax = plt.subplots(figsize=(16, 6), num=110)
        im = ax.imshow(np.flipud(m), cmap=colormap, aspect='auto')
        c = fig.colorbar(im, ax=ax)
        ax.set_xlabel('Basis States')
        ax.set_ylabel('Eigenstates')
        ax.set_title(f"System Eigenstates, B_Ext = [{spinsys.B[0]}, {spinsys.B[1]}, {spinsys.B[2]}]")
        c.set_label(name0)
        ax.set_xticks(np.arange(spinsys.dimensionOfMatrix))
        ax.set_xticklabels(spinsys.basisStates)
        ax.set_yticks(np.arange(spinsys.dimensionOfMatrix))
        ax.set_yticklabels(spinsys.statesOnlyLHS[::-1])
        # Set color limits
        if plot_type == 'p':
            im.set_clim(0, 1)
        else:
            im.set_clim(-1, 1)
        # Do not call plt.show() here
        return fig, ax, im, c
    
def plotZeemanDiagramm(spinsys: SpinSys,Brange: list=[0,1], N: int=200, energyscale: str='meV'):
    """
    Plots the Zeeman diagram for a given spin system over a specified magnetic field range.

    Parameters
    ----------
    spinsys : SpinSys
        The spin system object containing all relevant parameters and methods, including the magnetic field vector and energy levels.
    Brange : list of float, optional
        The range of magnetic field values (in Tesla) to sweep, given as [B_min, B_max]. Default is [0, 1].
    N : int, optional
        The number of points in the magnetic field sweep. Default is 200.

    Returns
    -------
    None
        This function does not return a value. It displays a plot of the Zeeman diagram.

    Notes
    -----
    - The function temporarily modifies the magnetic field vector of `spinsys` during the calculation and restores it to its original value afterwards.
    """

    # Save the initial B-Vector
    Binitial = spinsys.B
    B = np.linspace(Brange[0],Brange[1],N)
    E = np.zeros((N,spinsys.dimensionOfMatrix))
    
    for i in range(N):
        spinsys.B = [0,0,B[i]]
        hamil.calcEigEnergies(spinsys)
        E[i,:] = spinsys.E_sort

    # Resotre the initial solution
    spinsys.B = Binitial
    hamil.calcEigEnergies(spinsys)
        
    fig, ax = plt.subplots()
    for i in range(spinsys.dimensionOfMatrix):
        if energyscale == 'GHz':
            ax.plot(B, E[:, i] * spinsys.meVtoGHzConversion, label=f"E_{i}")
        else:
            ax.plot(B, E[:, i], label=f"E_{i}")
    ax.set_xlabel('Magnetic Field (T)')
    ax.set_ylabel(f'Energy ({energyscale})')
    ax.set_title('Zeeman Diagramm')
    ax.legend()
    # Do not call plt.show() here
    return fig, ax

def plotEnergyVsS(spinsys: SpinSys, type: str='z',energyscale: str='meV'):
    """
    Plots the energy eigenvalues of a spin system versus the expectation values of a spin operator (Sx, Sy, or Sz).

    Parameters
    ----------
    spinsys : SpinSys
        The spin system object containing Hamiltonian, eigenvectors, and energies.
    type : str, optional
        The spin operator to use for the expectation value ('x', 'y', or 'z'). Default is 'z'.

    Returns
    -------
    None
        This function displays a scatter plot and does not return any value.
        
    Notes
    -----
    If the eigenvectors of the spin system are not already calculated, the function will compute them.
    The expectation value of the selected spin operator is calculated for each eigenstate and plotted against its energy.
    """
    
    if not hasattr(spinsys, 'eigVectors') or spinsys.eigVectors is None:
        hamil.calcEigEnergies(spinsys)

    if type == 'z':
        S_Operator = np.sum(spinsys.Sz, axis=0)
    elif type == 'x':
        S_Operator = np.sum(spinsys.Sx, axis=0)
    elif type == 'y':
        S_Operator = np.sum(spinsys.Sy, axis=0)

    # Calculate the Sz matrix elements for each state
    S_elements = np.zeros(spinsys.dimensionOfMatrix)
    for i in range(spinsys.dimensionOfMatrix):
        S_elements[i] = np.real(spinsys.eigVectors[:, i].T.conj() @ S_Operator @ spinsys.eigVectors[:, i])

    # Prepare the energy values
    if energyscale == 'GHz':
        energies = spinsys.E_All_inGHz
    elif energyscale == 'meV':
        energies = spinsys.E_All
    else:
        raise ValueError("energyscale must be 'meV' or 'GHz'")

    # Create the figure and axes
    fig, ax = plt.subplots()
    ax.scatter(S_elements, energies, c='k', marker='_', s=1000, linewidths=5)
    ax.set_xlabel('m_' + type)
    ax.set_ylabel(f'Energy ({energyscale})')
    ax.set_title('Energy vs S_' + type + ' Matrix Elements')
    ax.grid(True)
    # Do not call plt.show() here
    return fig, ax

    return plt

def plotAll(spinsys: SpinSys, type: str='z', energyscale: str='meV'):
    """
    Plots all relevant information for the spin system, including energy levels, Zeeman diagram, and energy vs spin expectation values.

    Parameters
    ----------
    spinsys : SpinSys
        The spin system object containing Hamiltonian, eigenvectors, and energies.
    type : str, optional
        The spin operator to use for the expectation value ('x', 'y', or 'z'). Default is 'z'.
    energyscale : str, optional
        The energy scale to use for the plots ('meV' or 'GHz'). Default is 'meV'.

    Returns
    -------
    None
        This function does not return a value. It displays several plots.
    """
    # make them all plot next to each other
    plotEigenMatrix(spinsys, plot_type='p')
    plotEnergyVsS(spinsys, type=type, energyscale=energyscale)
    plotZeemanDiagramm(spinsys, energyscale=energyscale)
    plotPopulations(spinsys, AllowPumping=True, Approach="Loth")
# endregion Hamiltonian

# region Bath
def plotPopulations(spinsys: SpinSys, AllowPumping: bool = True, Approach: str = "Loth"):
    """
    Plots the thermal populations of the spin system's states.

    Parameters
    ----------
    spinsys : SpinSys
        The spin system object containing the necessary parameters and methods.
    AllowPumping : bool, optional
        Whether to allow pumping effects in the calculation (default is True).
    Approach : str, optional
        The approach to use for calculating populations (default is "Loth").

    Returns
    -------
    None
        This function generates a plot of the populations but does not return any value.
    
    Notes
    -----
    - The function assumes that the `spinsys` object has the necessary attributes and methods for rate calculations.
    """
    
    if not hasattr(spinsys, 'Populations') or spinsys.Populations is None:
            bath.calcPopulations(spinsys,AllowPumping=AllowPumping, Approach=Approach)
    if not hasattr(spinsys, 'basisSates') or spinsys.basisStates is None:
            hamil.calcEigStates(spinsys)
    
    x = np.arange(len(spinsys.Populations))
    width = 0.35

    fig, ax = plt.subplots()
    ax.barh(x + width/2, spinsys.pop_0, height=width, color='blue', alpha=0.5, label='Boltzmann Populations')
    ax.barh(x - width/2, spinsys.Populations, height=width, color='red', alpha=0.5, label='Tunneling Populations')
    ax.set_yticks(x)
    ax.set_yticklabels(spinsys.statesWithoutE)
    ax.set_xlabel('Population')
    ax.set_ylabel('Basis States')
    ax.set_title('Population of Spin States')
    ax.legend()
    plt.tight_layout()
    plt.show()

    return

def plotRateContributions(spinsys: SpinSys, yscale: str='linear', N=None, ylim: list=None):
    """
    Plot the contributions of transition rates between spin states.

    Parameters
    ----------
    spinsys : SpinSys
        The spin system object containing the states, rates, and matrix dimension.
        It is expected to have the following attributes:
            - dimensionOfMatrix: int, the number of spin states.
            - statesWithoutE: list of str, labels for the spin states.
            - Rates: list of 2D numpy arrays, each representing the rate matrix for a process.
    yscale : str, optional
        The scale for the y-axis ('linear' or 'log'). Default is 'linear'.
    N : int, optional
        The number of highest rate transitions to display. If None, all transitions are shown (default is None).
    ylim : list of float, optional
        The limits for the y-axis as [ymin, ymax]. If None, no limits are set (default is None).

    Returns
    -------
    None
        This function displays a matplotlib plot and does not return any value.

    Notes
    -----
    The function assumes that `bath.calcRates(spinsys)` populates the `spinsys.Rates`
    attribute with the required rate matrices. The plot shows stacked bars for each
    transition, with each color representing a different rate contribution.
    """
    hamil.calcEigStates(spinsys)
    bath.calcRates(spinsys)

    labels = ['ts', 'st', 'ss','tt']
    colors = ['blue', 'orange', 'green', 'red']

    if N is None:
        fig, ax = plt.subplots(figsize=(12, 5))
        for i in range(spinsys.dimensionOfMatrix):
            for f in range(spinsys.dimensionOfMatrix):
                if i != f:
                    bottom=0
                    for idx in range(4):
                        ax.bar(f'{spinsys.statesWithoutE[i]} → {spinsys.statesWithoutE[f]}', spinsys.Rates[idx][i, f], bottom=bottom, alpha=0.5, label=labels[idx], color=colors[idx])
                        bottom += spinsys.Rates[idx][i, f]
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, fontsize=10)
        ax.set_xlabel('Transitions')
        ax.set_ylabel('Rate')
        ax.set_title('Rate Contributions')
        ax.set_yscale(yscale)
        if ylim is not None:
            ax.set_ylim(ylim)
        plt.grid(True, which="both", ls="--", linewidth=0.5)
        # Only show legend for the first bar (Rate_exc)
        handles, _ = ax.get_legend_handles_labels()
        ax.legend(handles, labels)
        plt.show()
    
    # Limit the number of displayed transitions to N highest entries (ignoring nan)
    else:
        # Collect all transitions and their total rates
        transitions = []
        for i in range(spinsys.dimensionOfMatrix):
            for f in range(spinsys.dimensionOfMatrix):
                if i != f:
                    total_rate = sum([spinsys.Rates[idx][i, f] for idx in range(4)])
                    if not np.isnan(total_rate):
                        transitions.append(((i, f), total_rate))
        # Sort by total_rate descending and take top N
        transitions = sorted(transitions, key=lambda x: x[1], reverse=True)[:N]
        selected = set([t[0] for t in transitions])

        fig, ax = plt.subplots(figsize=(12, 5))
        for (i, f), _ in transitions:
            bottom = 0
            for idx in range(4):
                ax.bar(f'{spinsys.statesWithoutE[i]} → {spinsys.statesWithoutE[f]}', spinsys.Rates[idx][i, f], bottom=bottom, alpha=0.5, label=labels[idx], color=colors[idx])
                bottom += spinsys.Rates[idx][i, f]
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, fontsize=10)
        ax.set_xlabel('Transitions')
        ax.set_ylabel('Rate')
        ax.set_title(f'Top {N} Rate Contributions')
        ax.set_yscale(yscale)
        if ylim is not None:
            ax.set_ylim(ylim)
        plt.grid(True, which="both", ls="--", linewidth=0.5)
        handles, _ = ax.get_legend_handles_labels()
        ax.legend(handles, labels)
        plt.show()


    return

def plotRatesTo(spinsys: SpinSys, state: int, yscale: str='linear', N: int=None):
    """
    Plots the rate contributions to a specified state in a spin system.
    This function calculates and visualizes the transition rates from all other states to a given target state
    in the provided spin system. The rates are displayed as stacked bar plots, with each bar representing
    the contributions from different rate components (e.g., 'ts', 'st', 'ss', 'tt').

    Parameters
    ----------
    spinsys : SpinSys
        The spin system object containing state information and rate matrices.
    state : int
        The index of the target state to which rates are plotted.
    yscale : str, optional
        The scale for the y-axis of the plot. Can be 'linear', 'log', etc. Default is 'linear'.
    N : int, optional
        The number of highest rate transitions to display. If None, all transitions are shown (default is None).

    Returns
    -------
    None
        This function displays a plot and does not return any value.

    Notes
    -----
    - The function assumes that `spinsys` has attributes `dimensionOfMatrix`, `statesWithoutE`, and `Rates`.
    - The function also assumes the existence of a global `bath` object with a `calcRates` method.
    - The legend shows the labels for the different rate contributions.
    """
    labels = ['ts', 'st', 'ss','tt']
    colors = ['blue', 'orange', 'green', 'red']

    hamil.calcEigStates(spinsys)
    if N is not None:
        # Collect all transitions and their total rates to the target state
        transitions = []
        for i in range(spinsys.dimensionOfMatrix):
            if i != state:
                total_rate = sum([spinsys.Rates[idx][i, state] for idx in range(4)])
                if not np.isnan(total_rate):
                    transitions.append(((i, state), total_rate))
        # Sort by total_rate descending and take top N
        transitions = sorted(transitions, key=lambda x: x[1], reverse=True)[:N]
        selected = set([t[0][0] for t in transitions])
        fig, ax = plt.subplots(figsize=(8, 5))
        for (i, _), _ in transitions:
            bottom = 0
            for idx in range(4):
                ax.bar(f'{spinsys.statesWithoutE[i]} → {spinsys.statesWithoutE[state]}', spinsys.Rates[idx][i, state], bottom=bottom, alpha=0.5, label=labels[idx], color=colors[idx])
                bottom += spinsys.Rates[idx][i, state]
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, fontsize=10)
        ax.set_xlabel('Transitions')
        ax.set_ylabel('Rate')
        ax.set_title(f'Top {N} Rate Contributions to {spinsys.statesWithoutE[state]}')
        ax.set_yscale(yscale)
        plt.grid(True, which="both", ls="--", linewidth=0.5)
        handles, _ = ax.get_legend_handles_labels()
        ax.legend(handles, labels)
        plt.show()
        return
    else:
        hamil.calcEigStates(spinsys)
        bath.calcRates(spinsys)
        
        
        fig, ax = plt.subplots(figsize=(8, 5))
        for i in range(spinsys.dimensionOfMatrix):
            if i != state:
                bottom=0
                for idx in range(4):
                    ax.bar(f'{spinsys.statesWithoutE[i]} → {spinsys.statesWithoutE[state]}', spinsys.Rates[idx][i, state], bottom=bottom, alpha=0.5, label=labels[idx], color=colors[idx])
                    bottom += spinsys.Rates[idx][i, state]
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, fontsize=10)
        ax.set_xlabel('Transitions')
        ax.set_ylabel('Rate')
        ax.set_title(f'Rate Contributions to {spinsys.statesWithoutE[state]}')
        ax.set_yscale(yscale)
        plt.grid(True, which="both", ls="--", linewidth=0.5)
        # Only show legend for the first bar (Rate_exc)
        handles, _ = ax.get_legend_handles_labels()
        ax.legend(handles, labels)
        plt.show()

    return

def plotRatesFrom(spinsys: SpinSys, state: int, yscale: str='linear', N: int=None):
    """
    Plots all transition rates originating from a specified initial state.

    Parameters
    ----------
    spinsys : SpinSys
        The spin system object containing the states, rates, and matrix dimension.
        It is expected to have the following attributes:
            - dimensionOfMatrix: int, the number of spin states.
            - statesWithoutE: list of str, labels for the spin states.
            - Rates: list of 2D numpy arrays, each representing the rate matrix for a process.
    state : int
        The index of the initial state from which transitions originate.
    yscale : str, optional
        The scale for the y-axis ('linear' or 'log'). Default is 'linear'.
    N : int, optional
        The number of highest rate transitions to display. If None, all transitions are shown (default

    Returns
    -------
    None
        This function displays a matplotlib plot and does not return any value.

    Notes
    -----
    The function assumes that `bath.calcRates(spinsys)` populates the `spinsys.Rates`
    attribute with the required rate matrices. The plot shows stacked bars for each
    transition, with each color representing a different rate contribution.
    """

    bath.calcRates(spinsys)

    labels = ['ts', 'st', 'ss','tt']
    colors = ['blue', 'orange', 'green', 'red']
    fig, ax = plt.subplots(figsize=(8, 5))

    if N is not None:
        # Collect all transitions and their total rates from the given state
        transitions = []
        for f in range(spinsys.dimensionOfMatrix):
            if f != state:
                total_rate = sum([spinsys.Rates[idx][state, f] for idx in range(4)])
                if not np.isnan(total_rate):
                    transitions.append(((state, f), total_rate))
        # Sort by total_rate descending and take top N
        transitions = sorted(transitions, key=lambda x: x[1], reverse=True)[:N]
        for (s, f), _ in transitions:
            bottom = 0
            for idx in range(4):
                ax.bar(f'{spinsys.statesWithoutE[s]} → {spinsys.statesWithoutE[f]}', spinsys.Rates[idx][s, f], bottom=bottom, alpha=0.5, label=labels[idx], color=colors[idx])
                bottom += spinsys.Rates[idx][s, f]
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, fontsize=10)
        ax.set_xlabel('Transitions')
        ax.set_ylabel('Rate')
        ax.set_title(f'Top {N} Rate Contributions from {spinsys.statesWithoutE[state]}')
        ax.set_yscale(yscale)
        plt.grid(True, which="both", ls="--", linewidth=0.5)
        handles, _ = ax.get_legend_handles_labels()
        ax.legend(handles, labels)
        plt.show()
        
    else:

        for f in range(spinsys.dimensionOfMatrix):
            if f != state:
                bottom=0
                for idx in range(4):
                    ax.bar(f'{spinsys.statesWithoutE[state]} → {spinsys.statesWithoutE[f]}', spinsys.Rates[idx][state, f], bottom=bottom, alpha=0.5, label=labels[idx], color=colors[idx])
                    bottom += spinsys.Rates[idx][state, f]
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, fontsize=10)
        ax.set_xlabel('Transitions')
        ax.set_ylabel('Rate')
        ax.set_title(f'Rate Contributions from {spinsys.statesWithoutE[state]}')
        ax.set_yscale(yscale)
        plt.grid(True, which="both", ls="--", linewidth=0.5)
        # Only show legend for the first bar (Rate_exc)
        handles, _ = ax.get_legend_handles_labels()
        ax.legend(handles, labels)
        plt.show()

    return
# endregion Bath

# region dIdV

def plotIETS(spinsys: SpinSys, Vrange: list=[-30, 30], N: int=100, norm: bool=False, order: str='2nd', plotAllSpins: bool=False):
    """
    Plot the Inelastic Electron Tunneling Spectroscopy (IETS) spectrum for a given spin system.

    Parameters
    ----------
    spinsys : SpinSys
        The spin system object for which the IETS spectrum is to be calculated.
    Vrange : list, optional
        The voltage range [min, max] in millivolts over which to calculate the spectrum (default is [-30, 30]).
    N : int, optional
        Number of points in the voltage range for the spectrum calculation (default is 400).
    norm : bool, optional
        If True, normalize the calculated spectrum (default is False).
    order : str, optional
        The order of the IETS calculation. Currently, only '2nd' order is supported (default is '2nd').

    Raises
    ------
    ValueError
        If an unsupported order is specified.

    Returns
    -------
    None
        This function creates a plot of the IETS spectrum but does not return any value.
    """
    if plotAllSpins==False:
        # Calculate the IETS spectrum for the specified order
        if order == '2nd':
            x, y = dIdV.calcIETS_2nd(spinsys, Vrange=Vrange, N=N, norm=norm)
        elif order == '2ndR':
            x, y, _ = dIdV.calcIETS_2ndR(spinsys, Vrange=Vrange, N=N, norm=norm)
        elif order == '3rd':
            x, y = dIdV3.calcIETS_3rd(spinsys, Vrange=Vrange, N=N, norm=norm)
        elif order == '3rdR':
            x, y = dIdV3.calcIETS_3rdR(spinsys, Vrange=Vrange, N=N, norm=norm)
        else:
            raise ValueError("Unsupported order. Currently only '2nd' order is implemented.")
    
        # Plot the IETS spectrum
        plt.figure() 
        plt.plot(x, y, label='IETS Spectrum')
        plt.xlabel('Bias Voltage (mV)')
        plt.ylabel('dI/dV (a.u.)')
        plt.ylim([0, np.max(y) * 1.1])  # Set y-axis limits to avoid cutting off the top of the plot
        plt.title('IETS Spectrum ' + f'({order} Order)')
    elif plotAllSpins==True:
        fig, ax = plt.subplots()
        for i in range(len(spinsys.Spins)):
            spinsys.ReadoutSpin = i
            #print(f"Calculating IETS for Spin {i} with ReadoutSpin={spinsys.ReadoutSpin}")
            if order == '2nd':
                x, y = dIdV.calcIETS_2nd(spinsys, Vrange=Vrange, N=N, norm=norm)
            elif order == '2ndR':
                x, y, _ = dIdV.calcIETS_2ndR(spinsys, Vrange=Vrange, N=N, norm=norm)
            elif order == '3rd':
                x, y = dIdV3.calcIETS_3rd(spinsys, Vrange=Vrange, N=N, norm=norm)
            elif order == '3rdR':
                x, y = dIdV3.calcIETS_3rdR(spinsys, Vrange=Vrange, N=N, norm=norm)
            else:
                raise ValueError("Unsupported order. Currently only '2nd' '2ndR' and '3rd' order are implemented.")
            
            # Plot all IETS spectra in the same figure on top of each other
            # Only create the figure once, before the loop
            #if i == 0:
            ax.plot(x, y, label=f'Spin {i}')

        # Set labels and title for the plot
        ax.set_xlabel('Bias Voltage (mV)')
        ax.set_ylabel('dI/dV (a.u.)')
        ax.set_title(f'IETS Spectrum ({order} Order) for All Spins')
        ax.legend()
        ax.set_ylim(bottom=0)  # start from 0

        plt.tight_layout()
        plt.show()
            
        
    return 

# endregion dIdV

# region ESR

def plotESR_FreqSweep_Benjamin(spinsys: SpinSys, FreqRange: list = [10, 20], AllowPumping: bool = False, N: int = 1000, lw: float = 0.02, norm: bool = True, output: bool = False):
    """
    Plot the ESR (Electron Spin Resonance) spectrum for a given spin system over a specified frequency range.
    This function computes the ESR spectrum using the provided spin system and plots the resulting signal.
    It also marks the resonance frequencies corresponding to recorded transitions in the spin system.

    Parameters
    ----------
    spinsys : SpinSys
        The spin system object containing resonance frequencies and recorded transitions.
    FreqRange : list of float, optional
        The frequency range [start, stop] in GHz over which to compute the ESR spectrum (default is [10, 20]).
    AllowPumping : bool, optional
        Whether to allow pumping effects in the ESR calculation (default is False).
    N : int, optional
        Number of frequency points in the sweep (default is 1000).
    lw : float, optional
        Linewidth for the ESR calculation (default is 0.02).
    norm : bool, optional
        Whether to normalize the ESR signal (default is True).

    Returns
    -------
    fig : plotly.graph_objects.Figure
        The plotly figure object containing the ESR spectrum plot.

    Notes
    -----
    - Resonance frequencies and transitions are marked on the plot with different colors.
    - The ESR spectrum is calculated using `ESR.calcESR_Benjamin`.
    """
    
    # calculate the ESR spectrum
    freq_array, ESRsignal = ESR.calcESR_FreqSweep_Benjamin(spinsys, FreqRange=FreqRange, AllowPumping=AllowPumping,N=N, lw=lw, norm=norm, output=output)

    # Make the plot
    color_list = plotly.colors.qualitative.Plotly[:len(spinsys.ResonanceFrequencies)]

    # Create the main ESR signal trace
    fig = go.Figure()
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    fig.add_trace(go.Scatter(
        x=freq_array,
        y=ESRsignal,
        mode='lines',
        line=dict(color='black', width=2),
        name='ESR Signal'
    ))

    # Mark resonances
    for i, freq_res in enumerate(spinsys.ResonanceFrequencies):
        index = np.argmin(np.abs(freq_array - freq_res))
        amp = ESRsignal[index]
        if amp > 0.01 * np.max(ESRsignal):  # Filter very small peaks
            transition_str = (f"|{spinsys.RecordedTransitions[i][0]}> → |{spinsys.RecordedTransitions[i][1]}>; "
                             f"f = {np.round(freq_res,4)} GHz")
            fig.add_trace(go.Scatter(
                x=[freq_res],
                y=[amp],
                mode='markers',
                marker=dict(color=color_list[i], size=12),
                name=transition_str
            ))

    fig.update_layout(
        xaxis_title='Freq (GHz)',
        yaxis_title='ΔI (a.u.)',
        title='ESR-Simulation',
        legend_title='Transitions',
        showlegend=True
    )
    fig.update_xaxes(range=[freq_array[0], freq_array[-1]])
    fig.show()

    return fig

def plotESR_FreqSweep_ESRrate(spinsys: SpinSys, FreqRange = [10, 20], N: int = 200, norm: bool = True):
    """
    This function computes the ESR spectrum based on an ESR driving rate for a given spin system over a specified frequency range
    and plots the resulting ESR signal. It also marks the resonance frequencies corresponding to allowed transitions in the spin
    system.

    Parameters
    ----------
    spinsys : SpinSys
        The spin system object containing energy levels and transition information.
    FreqRange : list of float, optional
        The frequency range [start, stop] in GHz over which to perform the sweep (default is [10, 20]).
    N : int, optional
        Number of frequency points in the sweep (default is 1000).
    norm : bool, optional
        Whether to normalize the ESR signal (default is True).

    Returns
    -------
    fig : plotly.graph_objects.Figure
        The plotly figure object containing the ESR spectrum plot.
        
    Notes
    -----
    - The function updates `spinsys.RecordedTransitions` and `spinsys.ResonanceFrequencies` with the
      calculated transitions and their corresponding resonance frequencies.
    - Each resonance is marked on the plot with a unique color and labeled with the transition information.
    """

    # Calculate the ESR spectrum using the specified parameters
    freq_array, ESRsignal = ESR.calcESR_FreqSweep_ESRrate(spinsys, FreqRange=FreqRange, N=N, norm=norm)
    
    # Record the transitions
    spinsys.RecordedTransitions = ESR.calcRecordedTransitions(spinsys,freq_array[0],freq_array[-1])
    spinsys.ResonanceFrequencies = np.zeros(len(spinsys.RecordedTransitions))
    for i in range(len(spinsys.RecordedTransitions)):
        spinsys.ResonanceFrequencies[i] = abs(spinsys.E_All_inGHz[spinsys.RecordedTransitions[i][1]] - 
                                        spinsys.E_All_inGHz[spinsys.RecordedTransitions[i][0]])

    # Make the plot
    color_list = plotly.colors.qualitative.Plotly[:len(spinsys.ResonanceFrequencies)]

    # Create the main ESR signal trace
    fig = go.Figure()
    fig.update_layout(
    plot_bgcolor='white',
    paper_bgcolor='white'
    )
    fig.add_trace(go.Scatter(
        x=freq_array,
        y=ESRsignal,
        mode='lines',
        line=dict(color='black', width=2),
        name='ESR Signal'
    ))

    # Mark resonances
    for i, freq_res in enumerate(spinsys.ResonanceFrequencies):
        index = np.argmin(np.abs(freq_array - freq_res))
        amp = ESRsignal[index]
        if amp > 0.01 * np.max(ESRsignal):  # Filter very small peaks
            transition_str = (f"|{spinsys.RecordedTransitions[i][0]}> → |{spinsys.RecordedTransitions[i][1]}>; "
                             f"f = {np.round(freq_res,4)} GHz")
            fig.add_trace(go.Scatter(
                x=[freq_res],
                y=[amp],
                mode='markers',
                marker=dict(color=color_list[i], size=12),
                name=transition_str
            ))

    fig.update_layout(
        xaxis_title='Freq (GHz)',
        yaxis_title='ΔI (a.u.)',
        title='ESR-Simulation',
        legend_title='Transitions',
        showlegend=True
    )
    fig.update_xaxes(range=[freq_array[0], freq_array[-1]])
    fig.show()

    return fig

def plotESR_Tipfieldsweep(spinsys: SpinSys, FreqRange: list = [10, 20], N_Freq: int = 500, lw: float = 0.02, norm: bool = True, BTipRange: list = [-0.05, 0.05], N_B: int = 100, AllowPumping: bool = False, Angle: float = 0):  
    """
    Plot ESR spectra as a function of tip magnetic field for a given spin system.
    
    Parameters
    ----------
    spinsys : SpinSys
        The spin system object containing all relevant spin and field parameters.
    FreqRange : list of float, optional
        Frequency range [min, max] in GHz for the ESR spectrum (default is [10, 20]).
    N_Freq : int, optional
        Number of frequency points in the ESR spectrum (default is 500).
    lw : float, optional
        Linewidth for the ESR peaks (default is 0.02).
    norm : bool, optional
        Whether to normalize the ESR signal (default is True).
    BTipRange : list of float, optional
        Range [min, max] of tip magnetic field values in Tesla (default is [-0.05, 0.05]).
    N_B : int, optional
        Number of tip field values to compute (default is 100).
    AllowPumping : bool, optional
        Whether to allow pumping effects in the ESR calculation (default is False).
    Angle : float, optional
        Angle (in degrees) of the tip field with respect to the z-axis (default is 0).

    Returns
    -------
    fig : plotly.graph_objects.Figure
        The plotly figure object containing the ESR spectra as a function of tip field.
        
    Notes
    -----
    The function visualizes the ESR signal as a 2D color plot with frequency on the x-axis
    and tip field on the y-axis. A secondary y-axis shows the corresponding detuning in GHz.
    """
    # Save previous BTip value
    BTip_initial = spinsys.BTip.copy()

    # Predefine Arrays
    BTip_array = np.linspace(BTipRange[0],BTipRange[1],N_B,dtype=float)
    Freq_array = np.linspace(FreqRange[0],FreqRange[1],N_Freq,dtype=float)

    ESRsignal = np.zeros((N_B, N_Freq), dtype=float)
    
    # calculating the ESR signal for each Field 
    for i in range(N_B):
        spinsys.BTip[2]=BTip_array[i]*np.cos(Angle*np.pi/180) # BTip[2] is the z-component of the tip field
        spinsys.BTip[0]=BTip_array[i]*np.sin(Angle*np.pi/180)+1e-6 # BTip[0] is the x-component of the tip field
        spinsys.BTip[1]=0  # BTip[0] is the y-component of the tip field
        hamil.calcEigEnergies(spinsys)
        f , ESRsignal[i][:] = ESR.calcESR_FreqSweep_Benjamin(spinsys, norm=norm,N=N_Freq,lw=lw,FreqRange=FreqRange,AllowPumping=AllowPumping, output=False)
        

    # Calculating the Detuning from the Tipfield
    Detuning=spinsys.gfactorvector[spinsys.ReadoutSpin] * spinsys.muB * (BTip_array + spinsys.B[2]) * spinsys.meVtoGHzConversion - spinsys.g * spinsys.muB * spinsys.B[2] * spinsys.meVtoGHzConversion


    # Plot the ESR Signal as a color plot
    fig = go.Figure(data=go.Heatmap(
        z=ESRsignal,
        x=Freq_array,
        y=BTip_array * 1e3,  # Convert to mT for y-axis
        colorscale='Blues',
        colorbar=dict(title='ESR Signal'),
        zsmooth=False
    ))

    # Add labels and title
    fig.update_layout(
        height=600,
        width=800,
        xaxis_title='Freq (GHz)',
        yaxis_title='B_Tip (mT)',
        title=f'Tipfield dependent ESR (Angle = {Angle}°)'
    )

    # Add secondary y-axis for Detuning (GHz)
    # We'll add a custom axis on the right with detuning values
    # Calculate detuning for each BTip value
    detuning_labels = np.round(Detuning, 2)
    fig.update_layout(
        yaxis=dict(
            title='B_Tip (mT)',
            side='left'
        ),
        yaxis2=dict(
            title='Detuning (GHz)',
            overlaying='y',
            side='right',
            tickvals=BTip_array * 1e3,
            ticktext=detuning_labels
        )
    )

    fig.show()

    # Restore the initial BTip value
    spinsys.BTip = BTip_initial
    hamil.calcEigEnergies(spinsys)

    return fig, ESRsignal, Freq_array, BTip_array

def plotPopulations_ESR(spinsys: SpinSys, freq: float = 14, filter: float = None):
    """
    Plot the populations of spin states for a given spin system under ESR conditions.
    This function calculates and visualizes the populations of spin states, including Boltzmann, tunneling, 
    and ESR-driven populations, for a provided spin system. The populations are displayed as horizontal bar plots 
    for easy comparison.

    Parameters
    ----------
    spinsys : SpinSys
        The spin system object containing the relevant state and population information.
    freq : float, optional
        The frequency of the ESR field (in the same units as used in the simulation), by default 14.
    filter : float, optional
        If provided, only states with populations greater than this value in any of the three population arrays
        will be displayed. Default is None, which means all states are shown.

    Returns
    -------
    None
        This function does not return any value. It displays a matplotlib plot of the populations.

    Notes
    -----
    - The function assumes that the `spinsys` object has attributes `pop_0`, `Populations`, `Populations_ESR`, and `basisStates`.
    - If the eigenstates are not already calculated, the function will compute them.
    - Requires matplotlib and numpy for plotting and array operations.
    """

    ESR.calcPopulations_ESR(spinsys, freq=freq)
    if not hasattr(spinsys, 'basisSates') or spinsys.basisStates is None:
            hamil.calcEigStates(spinsys)
    
    x = np.arange(len(spinsys.Populations))
    width = 0.2

    if filter is None:
        fig, ax = plt.subplots()
        ax.barh(x, spinsys.pop_0, height=width, color='blue', alpha=0.5, label='Boltzmann Populations')
        ax.barh(x + width, spinsys.Populations, height=width, color='red', alpha=0.5, label='Tunneling Populations')
        ax.barh(x - width, spinsys.Populations_ESR, height=width, color='green', alpha=0.5, label='ESR Populations')
        ax.set_yticks(x)
        ax.set_yticklabels(spinsys.statesWithoutE)
        ax.set_xlabel('Population')
        ax.set_ylabel('Basis States')
        ax.set_title('Population of Spin States')
        ax.legend()
        plt.tight_layout()
        plt.show()
    else:
        # Only display states with population > filter in any of the three population arrays
        
        pop_arrays = [spinsys.pop_0, spinsys.Populations, spinsys.Populations_ESR]
        mask = np.any([np.array(p) >= filter for p in pop_arrays], axis=0)

        filtered_indices = np.where(mask)[0]
        if len(filtered_indices) == 0:
            print(f"No states with population greater than {filter*100}%.")
        else:
            fig, ax = plt.subplots()
            ax.barh(np.arange(len(filtered_indices)), spinsys.pop_0[mask], height=width, color='blue', alpha=0.5, label='Boltzmann Populations')
            ax.barh(np.arange(len(filtered_indices)) + width, spinsys.Populations[mask], height=width, color='red', alpha=0.5, label='Tunneling Populations')
            ax.barh(np.arange(len(filtered_indices)) - width, spinsys.Populations_ESR[mask], height=width, color='green', alpha=0.5, label='ESR Populations')
            ax.set_yticks(np.arange(len(filtered_indices)))
            ax.set_yticklabels(np.array(spinsys.statesWithoutE)[mask])
            ax.set_xlabel('Population')
            ax.set_ylabel('Basis States')
            ax.set_title(f'Population of Spin States (> {filter*100}%)')
            ax.legend()
            plt.tight_layout()
            plt.show()

    return
# endregion ESR

# region Magnetization over Voltage 

def plotMagnetization_Voltage(spinsys: SpinSys, Vrange: list=[-200, 200], N: int=100, type: str='z', spins: list=[0], norm: bool=True, total: bool=True):
    """
    Plots the magnetization of specified spins and the total magnetization as a function of voltage.

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
    None
        This function displays a plot and does not return any value.

    Notes
    -----
    The function temporarily modifies the `V_DC` attribute of `spinsys` during the voltage sweep,
    and restores its original value after plotting. The calculations rely on external `bath` functions
    for rates, populations, and magnetization.
    """
    
    # Save the old voltage value
    V_initial = spinsys.V_DC

    if len(spins) > spinsys.NSpins:
        print("Error: Too many spins specified in the 'spin' argument.")
        return
    
    # Initialize arrays to store voltage and magnetization data
    V_array = np.linspace(Vrange[0], Vrange[1], N)
    M_array = np.zeros((N, spinsys.NSpins+1)) 

    for idx, V in enumerate(V_array):
        spinsys.V_DC = V
        bath.calcRates(spinsys)
        bath.calcPopulations(spinsys, AllowPumping=True, Approach="Loth")
        bath.calcMagnetization(spinsys, type=type, AllowPumping=True, Approach= "Loth")
        for i in range(spinsys.NSpins):
            if norm:
                M_array[idx, i] = spinsys.Mag[i]/spinsys.Mag_max[i]
            else:
                M_array[idx, i] = spinsys.Mag[i]
        
        if norm:
            M_array[idx, -1] = spinsys.Mag_tot/spinsys.Mag_tot_max
        else:
            M_array[idx, -1] = spinsys.Mag_tot

    # Plotting
    plt.figure(figsize=(8, 5))
    for idx, spin in enumerate(spins):
        plt.plot(V_array, M_array[:, spin], label=f'Spin {spin} Magnetization')
    if total:
        plt.plot(V_array, M_array[:, -1], label='Total Magnetization', linestyle='-', color='red')
    if norm:
        plt.ylim([-1, 1])
        plt.plot(Vrange,[spinsys.TipPolarization[2],spinsys.TipPolarization[2]],'--k', label='Tip Polarization', alpha=0.5)
        plt.plot(Vrange,[-spinsys.TipPolarization[2],-spinsys.TipPolarization[2]],'--k', label='', alpha=0.5)
    plt.xlabel('Voltage (mV)')
    plt.ylabel('Magnetization')
    plt.title('Magnetization vs Voltage')
    plt.legend()
    plt.show()

    # Restore the initial voltage value
    spinsys.V_DC = V_initial
    bath.calcRates(spinsys)
    bath.calcPopulations(spinsys, AllowPumping=True, Approach="Loth")
    bath.calcMagnetization(spinsys, type=type, AllowPumping=True, Approach= "Loth")
    
    return 

def plotPopulations_Voltage(
        spinsys: SpinSys,
        Vrange: list = [-200, 200],
        N: int = 200):
    """
    Plots the populations of spin states as a function of applied voltage.
    
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
    None
        This function does not return a value; it displays a plot of populations vs voltage.

    Notes
    -----
    - The function temporarily modifies `spinsys.V_DC` during the sweep and restores its original value at the end.
    - The bath object and its methods `calcRates` and `calcPopulations` must be accessible in the current scope.
    """

    # Save the old voltage value
    V_initial = spinsys.V_DC

    # Initialize arrays to store voltage and population data
    V_array = np.linspace(Vrange[0], Vrange[1], N)
    P_array = np.zeros((N, spinsys.dimensionOfMatrix))
    for idx, V in enumerate(V_array):
        spinsys.V_DC = V
        bath.calcRates(spinsys)
        bath.calcPopulations(spinsys, AllowPumping=True, Approach="Loth")
        P_array[idx, :] = spinsys.Populations

    # Plotting
    plt.figure(figsize=(8, 5))
    for idx in range(spinsys.dimensionOfMatrix):
        plt.plot(V_array, P_array[:, idx], label=f'State {idx}')
    plt.xlabel('Voltage (mV)')
    plt.ylabel('Populations')
    plt.title('Populations vs Voltage')
    plt.legend()
    plt.yscale('log')
    plt.show()

    # Restore the initial voltage value
    spinsys.V_DC = V_initial
    bath.calcRates(spinsys)
    bath.calcPopulations(spinsys, AllowPumping=True, Approach="Loth")

    return

def plotRates_Voltages(spinsys: SpinSys, Vrange: list=[-200, 200], N: int=100):
    """
    Plots the transition rates between spin states as a function of applied voltage.
    
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
    None
        This function does not return a value. It displays a plot of the transition rates.
        
    Notes
    -----
    - The function temporarily modifies `spinsys.V_DC` during the sweep and restores its original
      value at the end.
    - The function assumes that `bath.calcRates(spinsys)` updates `spinsys.Rates_Summed` in-place.
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
                    R_array[idx, i, f] = spinsys.Rates_Summed[i,f]

    # Plotting
    plt.figure(figsize=(10, 6))
    for i in range(spinsys.dimensionOfMatrix):
        for f in range(spinsys.dimensionOfMatrix):
            if i != f:
                plt.plot(V_array, R_array[:, i, f], label=f'Rate {i}→{f}')
    
    plt.xlabel('Voltage (mV)')
    plt.ylabel('Transition Rates')
    plt.title('Transition Rates vs Voltage')
    plt.yscale('log')
    plt.legend()
    plt.show()

    # Restore the initial voltage value
    spinsys.V_DC = V_initial
    bath.calcRates(spinsys)

    return

def plotSplitMagnetization_Voltage(spinsys: SpinSys, subspin: SpinSys, Vrange: list=[-200,200], N: int=100, type: str='z', normVal: float=None):

    # Save the old voltage value
    V_initial = spinsys.V_DC

    # Initialize arrays to store voltage and rate data
    V_array = np.linspace(Vrange[0], Vrange[1], N)
    Mag_A = np.zeros(N)
    Mag_B = np.zeros(N)

    for idx, V in enumerate(V_array):
        spinsys.V_DC = V
        mA, mB = bath.split_Magnetization(spinsys, subspin, type=type)
        Mag_A[idx] = mA
        Mag_B[idx] = mB
    
    if normVal is not None:
        Mag_A = Mag_A/normVal
        Mag_B = Mag_B/normVal

    # Plotting
    plt.figure(figsize=(8, 5))
    plt.plot(V_array, Mag_A, label='Magnetization A')
    plt.plot(V_array, Mag_B, label='Magnetization B')
    plt.xlabel('Voltage (mV)')
    plt.ylabel('Magnetization')
    plt.title('Split Magnetization vs Voltage')
    plt.legend()
    plt.show()

    # Restore the initial voltage value
    spinsys.V_DC = V_initial
    bath.calcRates(spinsys)
    bath.calcPopulations(spinsys, AllowPumping=True)

    return

def plotSplitPopulations_Voltage(spinsys: SpinSys, subspin: SpinSys, Vrange: list=[-200,200], N: int=100):

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
        pA, pB = bath.split_Populations(spinsys, subspin)
        Populations_A[idx, :] = pA
        Populations_B[idx, :] = pB

    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for idx in range(subspin.dimensionOfMatrix):
        ax1.plot(V_array, Populations_A[:, idx], label=f'{subspin.statesWithoutE[idx]}')
        ax2.plot(V_array, Populations_B[:, idx], label=f'{subspin.statesWithoutE[idx]}')
    ax1.set_xlabel('Voltage (mV)')
    ax1.set_ylabel('Populations')
    ax1.set_title('Populations A vs Voltage')
    ax1.legend()
    ax1.set_ylim([0,1])

    ax2.set_xlabel('Voltage (mV)')
    ax2.set_title('Populations B vs Voltage')
    ax2.legend()
    ax2.set_ylim([0,1])

    # Restore the initial voltage value
    spinsys.V_DC = V_initial
    bath.calcRates(spinsys)
    bath.calcPopulations(spinsys, AllowPumping=True)

    return

def plotSplitPopulations(spinsys: SpinSys, subspin: SpinSys):

    Populations_A = np.zeros((subspin.dimensionOfMatrix))
    Populations_B = np.zeros((subspin.dimensionOfMatrix))

    
    Populations_A, Populations_B = bath.split_Populations(spinsys, subspin)
    
    # Plotting
    x = np.arange(subspin.dimensionOfMatrix)
    width = 0.4
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

    # Plot Populations_A
    ax1.barh(x, Populations_A, height=width, color='blue', alpha=0.6, label='Populations A')
    ax1.set_yticks(x)
    ax1.set_yticklabels(subspin.statesWithoutE)
    ax1.set_xlabel('Population')
    ax1.set_ylabel('States')
    ax1.set_title('Population of Spin A')
    ax1.legend()
    ax1.set_xlim([0,1])

    # Plot Populations_B
    ax2.barh(x, Populations_B, height=width, color='green', alpha=0.6, label='Populations B')
    ax2.set_yticks(x)
    ax2.set_yticklabels(subspin.statesWithoutE)
    ax2.set_xlabel('Population')
    ax2.set_title('Population of Spin B')
    ax2.legend()
    ax2.set_xlim([0,1])

    plt.tight_layout()
    plt.show()

    return
# endregion Magnetization over Voltage